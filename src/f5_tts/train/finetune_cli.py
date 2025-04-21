import argparse
import os
import shutil
from importlib.resources import files

from cached_path import cached_path

from f5_tts.model import CFM, UNetT, DiT, Trainer
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.dataset import load_dataset

from finetune_gradio import vocab_check
from finetune_gradio import vocab_extend
from finetune_gradio import expand_model_embeddings
from finetune_gradio import parse_s3_path
from finetune_gradio import read_file_s3

path_project_ckpts = str(files("f5_tts").joinpath("../../ckpts"))

# -------------------------- Dataset Settings --------------------------- #
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'


# -------------------------- Argument Parsing --------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train CFM Model")

    parser.add_argument(
        "--exp_name",
        type=str,
        default="F5TTS_v1_Base",
        choices=["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"],
        help="Experiment name",
    )
    parser.add_argument("--dataset_name", type=str, default="Emilia_ZH_EN", help="Name of the dataset to use")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--batch_size_per_gpu", type=int, default=3200, help="Batch size per GPU")
    parser.add_argument(
        "--batch_size_type", type=str, default="frame", choices=["frame", "sample"], help="Batch size type"
    )
    parser.add_argument("--max_samples", type=int, default=64, help="Max sequences per batch")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_warmup_updates", type=int, default=20000, help="Warmup updates")
    parser.add_argument("--save_per_updates", type=int, default=50000, help="Save checkpoint every N updates")
    parser.add_argument(
        "--keep_last_n_checkpoints",
        type=int,
        default=-1,
        help="-1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints",
    )
    parser.add_argument("--last_per_updates", type=int, default=5000, help="Save last checkpoint every N updates")
    parser.add_argument("--finetune", action="store_true", help="Use Finetune")
    parser.add_argument("--pretrain", type=str, default=None, help="the path to the checkpoint")
    parser.add_argument(
        "--tokenizer", type=str, default="pinyin", choices=["pinyin", "char", "custom"], help="Tokenizer type"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to custom tokenizer vocab file (only used if tokenizer = 'custom')",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Log inferenced samples per ckpt save updates",
    )
    parser.add_argument("--logger", type=str, default=None, choices=[None, "wandb", "tensorboard"], help="logger")
    parser.add_argument(
        "--bnb_optimizer",
        action="store_true",
        help="Use 8-bit Adam optimizer from bitsandbytes",
    )

    return parser.parse_args()


# -------------------------- Training Settings -------------------------- #

import unicodedata

def main():
    args = parse_args()

    # checkpoint_path = str(files("f5_tts").joinpath(f"../../ckpts/{args.dataset_name}"))
    # 체크포인트를 저장하고자 하는 경로
    # checkpoint_path = "/mnt/e/home/gyopark/F5-TTS/ckpts/emilia_l40"
    # checkpoint_path = "/mnt/e/home/gyopark/F5-TTS/ckpts/kss"
    checkpoint_path = "/mnt/e/F5-TTS/ckpts/kss"

    # Model parameters based on experiment name

    if args.exp_name == "F5TTS_v1_Base": # 최신 모델 사용
        wandb_resume_id = None
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
        )
        if args.finetune:
            if args.pretrain is None: # 현재 None을 사용
                ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
                # ckpt_path = /mnt/e/home/gyopark/.cache/huggingface/hub/models--SWivid--F5-TTS \
                # /snapshots/84e5a410d9cead4de2f847e7c9369a6440bdfaca/F5TTS_v1_Base \
                # /model_1250000.safetensors
            else:
                ckpt_path = args.pretrain

    elif args.exp_name == "F5TTS_Base":
        wandb_resume_id = None
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=False,
            conv_layers=4,
            pe_attn_head=1,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"))
            else:
                ckpt_path = args.pretrain

    elif args.exp_name == "E2TTS_Base":
        wandb_resume_id = None
        model_cls = UNetT
        model_cfg = dict(
            dim=1024,
            depth=24,
            heads=16,
            ff_mult=4,
            text_mask_padding=False,
            pe_attn_head=1,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt"))
            else:
                ckpt_path = args.pretrain

    if args.finetune: # finetune = True
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        file_checkpoint = os.path.basename(ckpt_path)
        print("file_checkpoint : ", file_checkpoint)
        if not file_checkpoint.startswith("pretrained_"):  # Change: Add 'pretrained_' prefix to copied model
            file_checkpoint = "pretrained_" + file_checkpoint
        file_checkpoint = os.path.join(checkpoint_path, file_checkpoint)
        if not os.path.isfile(file_checkpoint):
            shutil.copy2(ckpt_path, file_checkpoint)
            print("copy checkpoint for finetune")
        print("checkpoint_path : ", checkpoint_path)
        print("file_checkpoint : ", file_checkpoint)

    # Use the tokenizer and tokenizer_path provided in the command line arguments
    tokenizer = args.tokenizer
    if tokenizer == "custom":
        if not args.tokenizer_path:
            raise ValueError("Custom tokenizer selected, but no tokenizer_path provided.")

        tokenizer_path = args.tokenizer_path
        print("📌 Using vocab from:", tokenizer_path)

        # 🔢 Get vocab size from tokenizer_path
        if tokenizer_path.startswith("s3://"):
            vocab_bucket, vocab_key = parse_s3_path(tokenizer_path)
            vocab_content = read_file_s3(vocab_bucket, vocab_key)
            vocab_lines_raw = vocab_content.splitlines()
        else:
            with open(tokenizer_path, "r", encoding="utf-8") as f:
                vocab_lines_raw = f.read().splitlines()

        # 맨 앞 줄이 빈 줄이라면 그대로 유지, 나머지 줄은 strip 적용
        if vocab_lines_raw and vocab_lines_raw[0] == "":
            print("🔍 Detected leading empty line in vocab — preserving it.")
            vocab_lines = [""] + [line.rstrip("\n") for line in vocab_lines_raw[1:]]
        else:
            vocab_lines = [line.rstrip("\n") for line in vocab_lines_raw]
        print(vocab_lines[:10])  # Print first 10 lines for debugging
        print(vocab_lines[-10:])  # Print last 10 lines for debugging
        target_vocab_size = len(vocab_lines)
        
        if vocab_lines and vocab_lines[0] == "":
            target_vocab_size += 1
        print(f"✅ Target vocab size from tokenizer_path: {target_vocab_size}")
        # 🧠 Expand pretrained checkpoint to match vocab size
        old_ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))

        # emilia_l40 -> dataset_name
        # new_ckpt_path = "/mnt/e/home/gyopark/F5-TTS/ckpts/emilia_l40/pretrained_F5TTS_v1_Base_extended_direct.safetensors"
        new_ckpt_path = checkpoint_path + "/pretrained_F5TTS_v1_Base_extended_direct.safetensors"
        print("new_ckpt_path : ", new_ckpt_path)

        # Check current vocab size from checkpoint
        from safetensors.torch import load_file
        sd = load_file(old_ckpt_path, device="cpu")
        current_vocab_size = sd["ema_model.transformer.text_embed.text_embed.weight"].size(0)
        
        # vocab 오류 발생시 체크해볼부분
        num_new_tokens = target_vocab_size - current_vocab_size
        print(f"🔧 Expanding checkpoint from {current_vocab_size} to {target_vocab_size} tokens")

        if num_new_tokens > 0:
            expand_model_embeddings(
                ckpt_path=old_ckpt_path,
                new_ckpt_path=new_ckpt_path,
                num_new_tokens=num_new_tokens
            )
            print(f"✅ Extended checkpoint saved to: {new_ckpt_path}")

            # 🔥 확장 후 원본 삭제
            if os.path.exists(file_checkpoint):
                os.remove(file_checkpoint)
                print(f"🧹 Deleted original checkpoint: {file_checkpoint}")

            ckpt_path = new_ckpt_path
            # checkpoint_path = os.path.dirname(ckpt_path)
        else:
            # 🔒 여기를 바꿔야 함: ckpt_path는 원래 pretrained 경로를 사용하도록
            print("📭 No extension needed, vocab size already matches or is smaller.")
            ckpt_path = old_ckpt_path  # 그대로 사용
            # checkpoint_path = os.path.dirname(ckpt_path)
    else:
        tokenizer_path = args.dataset_name

    # /mnt/e/home/gyopark/F5-TTS/ckpts/pretrained_F5TTS_v1_Base_extended_direct.safetensors
    print("ckpt_path : ", ckpt_path) 
    print("tokenizer_path : ", tokenizer_path)
    print("tokenizer : ", tokenizer)

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    print("\nvocab : ", vocab_size)
    print("vocoder : ", mel_spec_type)
    
    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    model = CFM( # text_num_embeds=vocab_size
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    
    print("CHECKPOINT PATH : ", checkpoint_path)
    trainer = Trainer(
        model,
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=args.batch_size_per_gpu,
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logger=args.logger,
        wandb_project=args.dataset_name,
        wandb_run_name=args.exp_name,
        wandb_resume_id=wandb_resume_id,
        log_samples=args.log_samples,
        last_per_updates=args.last_per_updates,
        bnb_optimizer=args.bnb_optimizer,
    )

    train_dataset = load_dataset(args.dataset_name, tokenizer, mel_spec_kwargs=mel_spec_kwargs)

    trainer.train(
        train_dataset,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()