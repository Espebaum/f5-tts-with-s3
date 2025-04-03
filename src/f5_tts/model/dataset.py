import json
from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default

# ---
import os
import io
import boto3
import tempfile
import shutil
from typing import Optional

class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset_,
        target_sample_rate=24000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate
        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))
        audio_tensor = torch.from_numpy(audio).float()
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)
        audio_tensor = audio_tensor.unsqueeze(0)
        mel_spec = self.mel_spectrogram(audio_tensor).squeeze(0)
        text = row["text"]
        return dict(mel_spec=mel_spec, text=text)

def load_audio_from_s3(bucket_name, key, chunk_size=1024):
    """
    S3 버킷에서 지정한 음성 파일을 스트리밍 방식으로 읽어 torchaudio로 로드합니다.
    """
    s3 = boto3.client('s3', region_name='ap-northeast-2')  # 버킷 리전에 맞게 수정
    response = s3.get_object(Bucket=bucket_name, Key=key)
    streaming_body = response['Body']
    
    audio_buffer = io.BytesIO()
    while True:
        chunk = streaming_body.read(chunk_size)
        if not chunk:
            break
        audio_buffer.write(chunk)
    
    audio_buffer.seek(0)
    waveform, sample_rate = torchaudio.load(audio_buffer)
    return waveform, sample_rate

class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset,
        durations=None,
        target_sample_rate=24000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
        s3_bucket: str | None = None,  # 추가: S3 버킷 이름
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel
        self.s3_bucket = s3_bucket

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if self.durations is not None:
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 반복하여 유효한 항목을 찾음 (duration 필터링)
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]

            if 0.3 <= duration <= 30:
                break
            index = (index + 1) % len(self.data)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            # audio_path가 S3 경로(s3://...)이면, 스트리밍으로 로드
            if audio_path.startswith("s3://"):
                # s3://your-bucket-name/경로... 형식이라 가정
                parts = audio_path.split("/")
                bucket_name = parts[2]
                key = "/".join(parts[3:])
                audio, source_sample_rate = load_audio_from_s3(bucket_name, key)
            else:
                audio, source_sample_rate = torchaudio.load(audio_path)

            # 멀티채널인 경우 mono로 변환
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # 리샘플링
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            # 멜 스펙트로그램 변환
            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)

        return {"mel_spec": mel_spec, "text": text}

# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


# Load dataset
# --- 수정된 load_dataset 함수 ---
def parse_s3_path(s3_path: str):
    assert s3_path.startswith("s3://"), "Not a valid S3 path"
    parts = s3_path.split("/")
    bucket = parts[2]
    key = "/".join(parts[3:])
    return bucket, key

def download_from_s3(s3_path: str, local_path: str, region="ap-northeast-2"):
    bucket, key = parse_s3_path(s3_path)
    s3 = boto3.client("s3", region_name=region)
    s3.download_file(bucket, key, local_path)

from datasets import Dataset
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

def load_arrow_as_dataset(arrow_path: str) -> Dataset:
    """
    raw.arrow 파일을 로드하여 HuggingFace Dataset 객체로 변환합니다.
    from_arrow()을 지원하지 않는 경우 from_pandas()로 우회합니다.
    """
    with open(arrow_path, "rb") as f:
        reader = pa.ipc.RecordBatchFileReader(f)
        table = reader.read_all()
        df = table.to_pandas()
    return Dataset.from_pandas(df)

def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDatasetPath",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """
    print("Loading dataset ...")
    
    # S3 경로 지원: dataset_name이 S3 URL("s3://")으로 시작하면
    if dataset_name.startswith("s3://"):
        # 임시 폴더를 생성하여 S3 파일들을 다운로드합니다.
        temp_dir = tempfile.mkdtemp()
        try:
            # raw.arrow 파일 다운로드
            raw_s3_path = os.path.join(dataset_name, "raw.arrow")
            local_raw = os.path.join(temp_dir, "raw.arrow")
            download_from_s3(raw_s3_path, local_raw)
            # duration.json 파일 다운로드
            duration_s3_path = os.path.join(dataset_name, "duration.json")
            local_duration = os.path.join(temp_dir, "duration.json")
            download_from_s3(duration_s3_path, local_duration)
            
            # 데이터셋 로드: Arrow 파일이 Dataset 폴더 형식이 아닐 경우 from_file()를 사용
            try:
                train_dataset = load_from_disk(local_raw)
            except Exception as e:
                print(f"load_from_disk failed: {e}. Falling back to Dataset_.from_file")
                # train_dataset = Dataset_.from_file(local_raw)
                train_dataset = load_arrow_as_dataset(local_raw)

            with open(local_duration, "r", encoding="utf-8") as f:
                data_dict = json.load(f)
            durations = data_dict["duration"]
            
            # S3 경로를 사용한 경우에도 CustomDataset으로 Wrapping
            # preprocessed_mel 여부는 audio_type에 따라 결정합니다.
            preprocessed_mel = (audio_type == "mel")
            train_dataset = CustomDataset(
                train_dataset,
                durations=durations,
                preprocessed_mel=preprocessed_mel,
                mel_spec_module=mel_spec_module,
                **mel_spec_kwargs,
            )
        finally:
            # 임시 폴더는 사용 후 삭제합니다.
            shutil.rmtree(temp_dir)

    else:
        # 로컬 경로인 경우 기존 로직 사용
        if dataset_type == "CustomDataset":
            rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
            if audio_type == "raw":
                try:
                    train_dataset = load_from_disk(f"{rel_data_path}/raw")
                except:  # noqa: E722
                    train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
                preprocessed_mel = False
            elif audio_type == "mel":
                train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
                preprocessed_mel = True
            with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
                data_dict = json.load(f)
            durations = data_dict["duration"]
            train_dataset = CustomDataset(
                train_dataset,
                durations=durations,
                preprocessed_mel=preprocessed_mel,
                mel_spec_module=mel_spec_module,
                **mel_spec_kwargs,
            )
        elif dataset_type == "CustomDatasetPath":
            try:
                train_dataset = load_from_disk(f"{dataset_name}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")
            with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
                data_dict = json.load(f)
            durations = data_dict["duration"]
            train_dataset = CustomDataset(
                train_dataset, durations=durations, preprocessed_mel=False, **mel_spec_kwargs
            )
        elif dataset_type == "HFDataset":
            print(
                "Should manually modify the path of huggingface dataset to your need.\n"
                + "May also the corresponding script cuz different dataset may have different format."
            )
            pre, post = dataset_name.split("_")
            train_dataset = HFDataset(
                load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("f5_tts").joinpath("../../data")))
            )
    return train_dataset


# collation

def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
    )
