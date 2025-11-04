import argparse, json, os, re, heapq
from typing import List, Tuple, Optional

import torch
from torch.serialization import add_safe_globals
from omegaconf import DictConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
from omegaconf.listconfig import ListConfig
from typing import Any
from collections import defaultdict

# Разрешаем torch.load видеть OmegaConf DictConfig и т.д. 
add_safe_globals([DictConfig, ContainerMetadata, Any, dict, defaultdict, AnyNode, Metadata, ListConfig, list, int])


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import datasets as hfds
import torchaudio
from jiwer import wer

import gigaam
from gigaam.preprocess import SAMPLE_RATE as MODEL_SR

from src import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--model_name", default="ctc")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--out", default="ckpts_ru")
    ap.add_argument("--top_k", type=int, default=3, help="сколько лучших чекпоинтов по WER хранить")
    ap.add_argument("--numbers_mode", choices=["keep","spell"], default="spell")
    ap.add_argument("--latin_mode", choices=["keep","translit"], default="translit")
    ap.add_argument("--eval_after_training", action="store_true",
                    help="посчитать WER после обучения (best.pt по умолчанию)")
    ap.add_argument("--eval_split", choices=["validation","test"], default="validation",
                    help="на каком сплите считать post-train WER")
    ap.add_argument("--eval_ckpt", type=str, default="", help="чекпоинт для оценки (если пусто — best.pt)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    _, _, _, vocab, normalizer = build_loaders(
        lang=args.lang, batch_size=args.batch_size, num_workers=args.num_workers,
        model_name=args.model_name, numbers_mode=args.numbers_mode, latin_mode=args.latin_mode
    )

    # trainer
    trainer = CTCTrainer(
        model_name=args.model_name,
        lr=args.lr,
        freeze_encoder=args.freeze_encoder,
        out_dir=args.out,
        top_k=args.top_k
    )


    ckpt_path = args.eval_ckpt or os.path.join(args.out, "best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Не найден чекпоинт для оценки: {ckpt_path}")
    # построим eval loader на нужном сплите, с теми же нормализациями и вокабом
    eval_loader, eval_refs = build_eval_loader(
        lang=args.lang, split=args.eval_split,
        batch_size=args.batch_size, num_workers=args.num_workers,
        vocab=vocab, normalizer=normalizer
    )
    trainer.load_from_ckpt(ckpt_path)
    eval_wer = trainer.evaluate_loader(eval_loader, eval_refs)
    print(f"[eval] split={args.eval_split}  ckpt={ckpt_path}  WER={eval_wer:.4f}")
    
    print("Checkpoint 5")

if __name__ == "__main__":
    main()
