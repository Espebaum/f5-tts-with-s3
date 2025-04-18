import os
import sys
import signal
import subprocess  # For invoking ffprobe
import shutil
import concurrent.futures
import multiprocessing
from contextlib import contextmanager
import chardet

sys.path.append(os.getcwd())

import argparse
import csv
import json
from importlib.resources import files
from pathlib import Path
import boto3

import torchaudio
from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter

from f5_tts.model.utils import (
    convert_char_to_pinyin,
)

# PRETRAINED_VOCAB_PATH = files("f5_tts").joinpath("../../data/Emilia_ZH_EN_pinyin/vocab.txt")
PRETRAINED_VOCAB_PATH = Path("/mnt/e/home/gyopark/F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt")

# ---
from datasets.features import Features, Value

def merge_vocab_and_save(pretrained_s3_path, current_vocab_path, output_s3_path):
    print(f"☁️ Merging vocab from {pretrained_s3_path} and {current_vocab_path} to {output_s3_path}")

    def parse_s3_path(s3_path):
        assert s3_path.startswith("s3://")
        parts = s3_path.replace("s3://", "").split("/", 1)
        return parts[0], parts[1]

    def read_vocab_from_s3(s3_path):
        bucket, key = parse_s3_path(s3_path)
        s3 = boto3.client("s3", region_name="ap-northeast-2")
        response = s3.get_object(Bucket=bucket, Key=key)
        return [line.strip() for line in response["Body"].read().decode("utf-8").splitlines() if line.strip()]

    def read_vocab_from_local(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def write_vocab_to_s3(vocab_list, s3_path):
        bucket, key = parse_s3_path(s3_path)
        body = "\n" + "\n".join(vocab_list) + "\n"  # ← 앞에 개행 추가
        boto3.client("s3", region_name="ap-northeast-2").put_object(
            Bucket=bucket, Key=key, Body=body.encode("utf-8")
        )
    # Load vocabs
    pretrained_vocab = read_vocab_from_s3(pretrained_s3_path)
    new_vocab = read_vocab_from_local(current_vocab_path)

    # Build merged vocab (preserve order)
    pretrained_set = set(pretrained_vocab)
    final_vocab = pretrained_vocab + [tok for tok in new_vocab if tok not in pretrained_set]

    # Save
    write_vocab_to_s3(final_vocab, output_s3_path)
    print(f"☁️ Merged vocab saved to {output_s3_path}")
    print("수정 완료")
    return len(final_vocab)


def parse_s3_path(s3_path: str):
    """
    s3://bucket-name/path/to/file 형태의 경로에서 버킷 이름과 key를 반환합니다.
    """
    assert s3_path.startswith("s3://")
    parts = s3_path.split("/")
    bucket = parts[2]
    key = "/".join(parts[3:])
    return bucket, key

def s3_object_exists(s3_path: str, region="ap-northeast-2"):
    """
    주어진 s3 경로의 객체가 존재하는지 확인합니다.
    """
    bucket, key = parse_s3_path(s3_path)
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False

def s3_prefix_exists(s3_path: str, region="ap-northeast-2"):
    """
    주어진 s3 경로(접두사)에 해당하는 객체가 하나라도 존재하는지 확인합니다.
    """
    bucket, prefix = parse_s3_path(s3_path)
    s3 = boto3.client("s3", region_name=region)
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return "Contents" in response



from pathlib import Path

def is_csv_wavs_format(input_dir):
    """
    입력 디렉토리가 csv_wavs 포맷인지 확인합니다.
    로컬 경로인 경우 metadata.csv 파일과 wavs 폴더의 존재 여부를 체크하고,
    S3 경로인 경우 boto3를 사용해 해당 객체들이 존재하는지 확인합니다.
    """
    if input_dir.startswith("s3://"):
        # S3 경로 처리: 메타데이터 CSV와 wavs 접두사가 존재하는지 확인
        metadata_path = input_dir.rstrip("/") + "/metadata.csv"
        wavs_prefix = input_dir.rstrip("/") + "/wavs/"
        if not s3_object_exists(metadata_path):
            return False
        if not s3_prefix_exists(wavs_prefix):
            return False
        return True
    else:
        # 로컬 경로 처리
        fpath = Path(input_dir)
        metadata = fpath / "metadata.csv"
        wavs = fpath / "wavs"
        return metadata.exists() and metadata.is_file() and wavs.exists() and wavs.is_dir()

# Configuration constants
BATCH_SIZE = 100  # Batch size for text conversion
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
THREAD_NAME_PREFIX = "AudioProcessor"
CHUNK_SIZE = 100  # Number of files to process per worker batch

executor = None  # Global executor for cleanup


@contextmanager
def graceful_exit():
    """Context manager for graceful shutdown on signals"""

    def signal_handler(signum, frame):
        print("\nReceived signal to terminate. Cleaning up...")
        if executor is not None:
            print("Shutting down executor...")
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        yield
    finally:
        if executor is not None:
            executor.shutdown(wait=False)


def process_audio_file(audio_path, text, polyphone):
    """Process a single audio file by checking its existence and extracting duration."""
    if not Path(audio_path).exists():
        print(f"audio {audio_path} not found, skipping")
        return None
    try:
        audio_duration = get_audio_duration(audio_path)
        if audio_duration <= 0:
            raise ValueError(f"Duration {audio_duration} is non-positive.")
        return (audio_path, text, audio_duration)
    except Exception as e:
        print(f"Warning: Failed to process {audio_path} due to error: {e}. Skipping corrupt file.")
        return None


def batch_convert_texts(texts, polyphone, batch_size=BATCH_SIZE):
    """Convert a list of texts to pinyin in batches."""
    converted_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        converted_batch = convert_char_to_pinyin(batch, polyphone=polyphone)
        converted_texts.extend(converted_batch)
    return converted_texts

def get_audio_duration_s3(s3_path: str, region="ap-northeast-2"):
    """
    S3 오디오 파일을 스트리밍으로 불러와 duration 계산
    """
    assert s3_path.startswith("s3://")
    bucket, key = parse_s3_path(s3_path)

    s3 = boto3.client("s3", region_name=region)
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response['Body'].read()

    audio_bytes = io.BytesIO(body)
    waveform, sample_rate = torchaudio.load(audio_bytes)
    duration = waveform.shape[1] / sample_rate
    return duration

def prepare_csv_wavs_dir(input_dir, wav_root=None, num_workers=None):
    """
    새로운 형식의 metadata (id,wav,text,duration,...)를 처리하는 함수.
    - input_dir: metadata 파일들이 들어있는 S3 prefix (e.g. s3://bucket/path/to/metadata/)
    - wav_root: 실제 wav/mp3 파일들이 있는 S3 prefix (e.g. s3://bucket/Emilia-dataset/YODAS/KO)
    """
    assert input_dir.startswith("s3://"), "Only S3 input is currently supported."
    s3 = boto3.client("s3", region_name="ap-northeast-2")
    bucket, prefix = parse_s3_path(input_dir)

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    metadata_keys = [content["Key"] for content in response.get("Contents", []) if content["Key"].endswith(".csv")]

    print(f"📦 Total metadata files: {len(metadata_keys)}")

    audio_text_pairs = []
    for key in metadata_keys:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8-sig").splitlines()
        reader = csv.DictReader(content)

        for row in reader:
            file_path = row["id"]
            text = row["text"].strip()
            wav_subfolder = row.get("sub", "").strip()
            duration_str = row.get("duration", "").strip()

            # 경로 확장자 보장
            if not file_path.endswith(".mp3"):
                file_path += ".mp3"

            # S3 오디오 경로 구성
            if wav_root:
                audio_path = os.path.join(wav_root.rstrip("/"), wav_subfolder, file_path)
            else:
                audio_path = f"s3://{bucket}/{file_path}"

            try:
                duration = float(duration_str)
                audio_text_pairs.append((audio_path, text, duration))
            except ValueError:
                print(f"⚠️ Skipping {audio_path}: invalid duration '{duration_str}'")
                continue

    print(f"🎧 Total valid audio-text-duration triples: {len(audio_text_pairs)}")

    results = []
    durations = []
    vocab_set = set()

    for audio_path, text, duration in tqdm(audio_text_pairs, desc="Processing audio metadata"):
        try:
            results.append({
                "audio_path": audio_path,
                "text": text,
                "duration": duration
            })
            durations.append(duration)
            vocab_set.update(list(text))
        except Exception as e:
            print(f"❌ Error handling row {audio_path}: {e}")

    print(f"✅ Final result size: {len(results)}")
    return results, durations, vocab_set



def get_audio_duration(audio_path, timeout=5):
    """
    Get the duration of an audio file in seconds using ffmpeg's ffprobe.
    Falls back to torchaudio.load() if ffprobe fails.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=timeout
        )
        duration_str = result.stdout.strip()
        if duration_str:
            return float(duration_str)
        raise ValueError("Empty duration string from ffprobe.")
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
        print(f"Warning: ffprobe failed for {audio_path} with error: {e}. Falling back to torchaudio.")
        try:
            audio, sample_rate = torchaudio.load(audio_path)
            return audio.shape[1] / sample_rate
        except Exception as e:
            raise RuntimeError(f"Both ffprobe and torchaudio failed for {audio_path}: {e}")


from pathlib import Path
import chardet

def read_csv_from_s3(s3_path):
    assert s3_path.startswith("s3://"), f"Invalid S3 path: {s3_path}"
    parts = s3_path.split("/")
    bucket_name = parts[2]
    key = "/".join(parts[3:])
    
    s3 = boto3.client("s3", region_name="ap-northeast-2")
    response = s3.get_object(Bucket=bucket_name, Key=key)
    raw_bytes = response["Body"].read()

    # ✅ Step 1: Try utf-8-sig
    try:
        return raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        print("❗ utf-8-sig decoding failed, trying cp949...")
    
    # ✅ Step 2: Try cp949 (Windows Korean)
    try:
        return raw_bytes.decode("cp949")
    except UnicodeDecodeError:
        raise RuntimeError(f"❌ Failed to decode {s3_path} with utf-8-sig and cp949.")

import io
import csv

def read_audio_text_pairs(csv_file_path):
    print("ASDFASDF")
    audio_text_pairs = []
    if csv_file_path.startswith("s3://"):
        bucket, key = parse_s3_path(csv_file_path)
        s3 = boto3.client("s3", region_name="ap-northeast-2")
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8-sig")
        lines = content.splitlines()
    else:
        fpath = Path(csv_file_path)
        with open(fpath, mode="r", newline="", encoding="utf-8-sig") as csvfile:
            lines = csvfile.read().splitlines()

    print(f"Read {len(lines)} lines from {csv_file_path}")  # 디버깅용 출력

    reader = csv.reader(lines, delimiter="|")
    try:
        header = next(reader)
        print("Header:", header)
    except StopIteration:
        return audio_text_pairs
    for row in reader:
        if len(row) >= 2:
            audio_file = row[0].strip()
            text = row[1].strip()
            audio_text_pairs.append((audio_file, text))
    print(f"Parsed {len(audio_text_pairs)} audio-text pairs")
    return audio_text_pairs


from datasets.arrow_writer import ArrowWriter
from pathlib import Path
import numpy as np

import pyarrow as pa
import pyarrow.ipc as ipc

def save_arrow_file_with_pyarrow(records: list, path: str):
    """
    List of dictionaries → Apache Arrow 포맷으로 저장
    """
    schema = pa.schema([
        ("audio_path", pa.string()),
        ("text", pa.string()),
        ("duration", pa.float32()),
    ])
    table = pa.Table.from_pylist(records, schema=schema)

    with open(path, "wb") as f:
        with ipc.RecordBatchFileWriter(f, schema) as writer:
            writer.write_table(table)

    print(f"✅ Saved Arrow file to: {path}")

def save_prepped_dataset(out_dir, result, duration_list, text_vocab_set, is_finetune):
    if not result:
        raise ValueError("❌ result is empty before Arrow write")

    is_s3 = str(out_dir).startswith("s3://")
    temp_dir = Path("/tmp/prepped_dataset")
    temp_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n✅ Saving to temporary path: {temp_dir}")

    cleaned_result = [
        {
            "audio_path": r["audio_path"],
            "text": r["text"],
            "duration": np.float32(r["duration"]),
        }
        for r in result
        if is_valid_arrow_row(r)
    ]

    if not cleaned_result:
        raise RuntimeError("❌ No valid entries after cleaning result for ArrowWriter")

    print(f"📋 Cleaned result count: {len(cleaned_result)}")
    print("🔍 Sample of cleaned_result:", cleaned_result[:2])

    # File paths
    raw_arrow_path = temp_dir / "raw.arrow"
    dur_json_path = temp_dir / "duration.json"
    voca_out_path = temp_dir / "vocab.txt"

    # Save Arrow
    save_arrow_file_with_pyarrow(cleaned_result, raw_arrow_path)
    if raw_arrow_path.stat().st_size == 0:
        raise RuntimeError("❌ raw.arrow file is 0 bytes. Something went wrong.")

    # Save duration
    with open(dur_json_path, "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # Save vocab
    if is_finetune:
        shutil.copy2(PRETRAINED_VOCAB_PATH, voca_out_path)
        merged_vocab_size = len(set(open(PRETRAINED_VOCAB_PATH).read().splitlines()))
    else:
        with open(voca_out_path, "w", encoding="utf-8") as f:
            for vocab in sorted(text_vocab_set):
                f.write(vocab + "\n")
        merged_vocab_size = len(text_vocab_set)

    if is_s3:
        # Merge and upload vocab
        merged_vocab_s3_path = "s3://kmpark-seoul/vocab.txt"
        if not is_finetune:
            merged_vocab_size = merge_vocab_and_save(
                pretrained_s3_path="s3://kmpark-seoul/pretrained/vocab.txt",
                current_vocab_path=str(voca_out_path),
                output_s3_path=merged_vocab_s3_path,
            )
            print(f"☁️ Uploaded merged vocab to: {merged_vocab_s3_path} (✅ size: {merged_vocab_size})")
        else:
            print(f"☁️ Uploaded vocab from fine-tuned base (no merge). Size: {merged_vocab_size}")

        # Upload arrow and duration
        s3 = boto3.client("s3", region_name="ap-northeast-2")
        bucket, prefix = parse_s3_path(str(out_dir))
        for file in [raw_arrow_path, dur_json_path]:
            key = f"{prefix.rstrip('/')}/{file.name}" if prefix.strip() else file.name
            print(f"☁️ Uploading {file.name} to s3://{bucket}/{key}")
            s3.upload_file(str(file), bucket, key)
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(raw_arrow_path), out_dir / "raw.arrow")
        shutil.move(str(dur_json_path), out_dir / "duration.json")
        shutil.move(str(voca_out_path), out_dir / "vocab.txt")

    print(f"\n✅ Dataset saved to: {out_dir}")
    print(f"🔢 Sample count: {len(result)}")
    print(f"🔤 Current dataset vocab size: {len(text_vocab_set)}")
    print(f"🔀 Merged vocab size (saved): {merged_vocab_size}")
    print(f"⏱ Total hours: {sum(duration_list)/3600:.2f}")


from datasets import Dataset
import unicodedata

def sanitize_text(text):
    if not isinstance(text, str):
        return ""
    try:
        # normalize + UTF-8 strict encoding → 디코딩 가능한 문자만 필터링
        text = unicodedata.normalize("NFKC", text)
        text = text.encode("utf-8", "strict").decode("utf-8")  # 강제 유효성 검사
    except UnicodeEncodeError as e:
        print(f"⚠️ Invalid UTF-8 text filtered: {repr(text)} ({e})")
        return ""
    text = text.replace("\x00", "")  # null 문자 제거
    return text.strip()


def is_valid_arrow_row(row: dict):
    try:
        return (
            isinstance(row.get("audio_path"), str) and row.get("audio_path").strip() != "" and
            isinstance(row.get("text"), str) and row.get("text").strip() != "" and
            isinstance(row.get("duration"), (int, float)) and not (row["duration"] != row["duration"])  # NaN
        )
    except:
        return False

def prepare_and_save_set(inp_dir, out_dir, wav_root=None, is_finetune: bool = True, num_workers: int = None):
    if is_finetune:
        assert PRETRAINED_VOCAB_PATH.exists(), f"pretrained vocab.txt not found: {PRETRAINED_VOCAB_PATH}"

    # Step 1: Load
    raw_results, durations, vocab_set = prepare_csv_wavs_dir(inp_dir, wav_root=wav_root, num_workers=num_workers)

    # Step 2: Validate & sanitize
    valid_results = []
    valid_durations = []

    for r in raw_results:
        try:
            sanitized_text = sanitize_text(r.get("text", ""))
            if (
                isinstance(r, dict) and
                isinstance(r.get("audio_path"), str) and r["audio_path"].strip() and
                sanitized_text and
                isinstance(r.get("duration"), (float, int)) and not (r["duration"] != r["duration"])  # NaN check
            ):
                valid_results.append({
                    "audio_path": r["audio_path"].strip(),
                    "text": sanitized_text,
                    "duration": float(r["duration"]),
                })
                valid_durations.append(float(r["duration"]))
            else:
                print(f"⚠️ Skipped invalid record: {r}")
        except Exception as e:
            print(f"❌ Error validating record {r}: {e}")

    if not valid_results:
        raise ValueError("❌ No valid results found to write to arrow file. Check CSV contents.")

    # Step 3: Save
    save_prepped_dataset(out_dir, valid_results, valid_durations, vocab_set, is_finetune)


def cli():
    try:
        # Before processing, check if ffprobe is available.
        if shutil.which("ffprobe") is None:
            print(
                "Warning: ffprobe is not available. Duration extraction will rely on torchaudio (which may be slower)."
            )

        # Usage examples in help text
        parser = argparse.ArgumentParser(
            description="Prepare and save dataset.",
            epilog="""
Examples:
    # For fine-tuning (default):
    python prepare_csv_wavs.py /input/dataset/path /output/dataset/path
    
    # For pre-training:
    python prepare_csv_wavs.py /input/dataset/path /output/dataset/path --pretrain
    
    # With custom worker count:
    python prepare_csv_wavs.py /input/dataset/path /output/dataset/path --workers 4
            """,
        )
        parser.add_argument("inp_dir", type=str, help="Input directory containing the data.")
        parser.add_argument("out_dir", type=str, help="Output directory to save the prepared data.")
        parser.add_argument("--wav-root", type=str, help="Optional different root S3 path for .wav files.")
        parser.add_argument("--pretrain", action="store_true", help="Enable for new pretrain, otherwise is a fine-tune")
        parser.add_argument("--workers", type=int, help=f"Number of worker threads (default: {MAX_WORKERS})")
        args = parser.parse_args()

        prepare_and_save_set(
            args.inp_dir,
            args.out_dir,
            wav_root=args.wav_root,   # ← 이거 추가
            is_finetune=args.pretrain,
            num_workers=args.workers
        )
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Cleaning up...")
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()