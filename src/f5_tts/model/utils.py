from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import torch
from torch.nn.utils.rnn import pad_sequence

import jieba
from pypinyin import lazy_pinyin, Style


# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:  # noqa: F722
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


# Get tokenizer

import os
import tempfile
import boto3
from importlib.resources import files

def download_from_s3(s3_path: str, local_path: str, region="ap-northeast-2"):
    """s3://bucket/path/to/file 형식의 s3 경로에서 로컬 파일로 다운로드합니다."""
    assert s3_path.startswith("s3://"), "Not a valid s3 path"
    parts = s3_path.split("/")
    bucket = parts[2]
    key = "/".join(parts[3:])
    s3 = boto3.client("s3", region_name=region)
    s3.download_file(bucket, key, local_path)

def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" : 중국어 문자에 대해 g2p를 수행 (vocab.txt 필요)
                  - "char"   : 문자 단위 토크나이저 (vocab.txt 필요)
                  - "byte"   : utf-8 토크나이저 (vocab_size=256)
                  - "custom" : 사용자가 지정한 vocab.txt 경로 직접 지정
    vocab_size  - "pinyin"인 경우, 사용 가능한 모든 pinyin 및 기호 등
                  - "char"인 경우, 데이터셋에서 추출된 문자 및 기호 개수
                  - "byte"인 경우 256 (유니코드 바이트 범위)
    """
    # pinyin과 char 모드: 파일은 기본적으로 ../../data/{dataset_name}_{tokenizer}/vocab.txt에 있다고 가정
    if tokenizer in ["pinyin", "char"]:
        # 기본 경로 (로컬)
        base_path = os.path.join(files("f5_tts").joinpath("../../data"))
        tokenizer_path = os.path.join(base_path, f"{dataset_name}_{tokenizer}", "vocab.txt")
        # S3 경로 지원: dataset_name이 s3://로 시작하면
        if tokenizer_path.startswith("s3://"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                local_vocab_path = tmp_file.name
            download_from_s3(tokenizer_path, local_vocab_path)
            vocab_file = local_vocab_path
        else:
            vocab_file = tokenizer_path

        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char.rstrip("\n")] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map.get(" ") == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

        # S3 임시 파일을 사용한 경우 삭제
        if tokenizer_path.startswith("s3://"):
            os.remove(local_vocab_path)

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        # custom 모드에서는 dataset_name이 직접 vocab.txt 파일의 경로여야 함.
        if dataset_name.startswith("s3://"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                local_vocab_path = tmp_file.name
            download_from_s3(dataset_name, local_vocab_path)
            vocab_file = local_vocab_path
        else:
            vocab_file = dataset_name

        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char.rstrip("\n")] = i
        vocab_size = len(vocab_char_map)

        if dataset_name.startswith("s3://"):
            os.remove(local_vocab_path)

    return vocab_char_map, vocab_size


# convert char to pinyin


def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False
