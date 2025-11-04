import os, heapq
from typing import List, Tuple
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim

from jiwer import wer

import gigaam



class CTCTrainer:
    def __init__(self, model_name: str, lr: float = 2e-4, freeze_encoder: bool = False, out_dir: str = "ckpts", top_k:int=3, amp_dtype = "auto"):
        self.model: gigaam.model.GigaAMASR = gigaam.load_model(model_name, fp16_encoder=False)
        self.device = self.model._device
        print(f"Using device {self.device}")
        self.dtype = self.model._dtype
        self.vocab = list(self.model.cfg.decoding.vocabulary)
        from gigaam.decoding import CTCGreedyDecoding
        self.greedy = CTCGreedyDecoding(vocabulary=self.vocab)
        self.blank_id = len(self.vocab)
        if freeze_encoder:
            for p in self.model.encoder.parameters():
                p.requires_grad_(False)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = optim.AdamW(self.params, lr=lr, betas=(0.9,0.98), weight_decay=1e-4)
        self.crit = nn.CTCLoss(blank=self.blank_id, reduction="mean", zero_infinity=True)
        self.amp_dtype = amp_dtype  # "auto" | "fp16" | "bf16" | "off"
        use_fp16 = (self.device.type == "cuda") and (
            self.amp_dtype == "fp16" or (self.amp_dtype == "auto" and not torch.cuda.is_bf16_supported())
        )
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
        self.out_dir = out_dir
        self.top_k = max(1, top_k)
        os.makedirs(out_dir, exist_ok=True)
        self.best_heap: List[Tuple[float,str]] = []  # max-heap via (-wer, path)

        self.global_step = 0
        

    def _autocast_ctx(self):
        from contextlib import nullcontext
        dev = self.device.type
        if self.amp_dtype == "off":
            return nullcontext()
        if dev == "cuda":
            if self.amp_dtype == "bf16" or (self.amp_dtype == "auto" and torch.cuda.is_bf16_supported()):
                return torch.autocast("cuda", dtype=torch.bfloat16)
            elif self.amp_dtype in ("fp16", "auto"):
                return torch.autocast("cuda", dtype=torch.float16)
        elif dev == "mps":
            if self.amp_dtype in ("fp16", "auto"):
                return torch.autocast("mps", dtype=torch.float16)
        return nullcontext()

    def _save_ckpt(self, tag: str):
        path = os.path.join(self.out_dir, f"{tag}.pt")
        torch.save({"state_dict": self.model.state_dict(), "cfg": self.model.cfg}, path)
        return path

    def _track_topk(self, wer_value: float, path: str):
        heapq.heappush(self.best_heap, (-wer_value, path))
        if len(self.best_heap) > self.top_k:
            # drop worst (highest -wer => most negative) => actually keep smallest WER
            worst = heapq.nlargest(1, self.best_heap)[0]
            self.best_heap.remove(worst)
            try:
                os.remove(worst[1])
            except OSError:
                pass
            heapq.heapify(self.best_heap)

    def step_batch(self, batch) -> float:
        wavs, wav_lens, targets, target_lens = batch
        wavs = wavs.to(self.device)
        # на CPU лучше оставить float32
        if self.device.type != "cpu":
            wavs = wavs.to(self.dtype)

        wav_lens = wav_lens.to(self.device)
        targets = targets.to(self.device)
        target_lens = target_lens.to(self.device)

        self.opt.zero_grad(set_to_none=True)

        with self._autocast_ctx():
            enc, enc_len = self.model.forward(wavs, wav_lens)   # [B,C,T]
            log_probs = self.model.head(encoder_output=enc)      # [B,T,C]
            loss = self.crit(
                log_probs.transpose(0, 1),                       # -> [T,B,C]
                targets,
                enc_len.to(torch.int32),
                target_lens.to(torch.int32),
            )

        self.scaler.scale(loss).backward() if self.device.type == "cuda" else loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        if self.device.type == "cuda":
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()

        res = float(loss.item())
        del wavs, wav_lens, targets, target_lens, enc, enc_len, log_probs, loss
        if self.device.type == "cuda" and (self.global_step % 10 == 0):
            torch.cuda.empty_cache()


        return res

    @torch.no_grad()
    def validate(self, loader, refs: List[str]) -> float:
        self.model.eval()
        hyps: List[str] = []
        for (wavs, wav_lens, _, _) in loader:
            wavs = wavs.to(self.device).to(self.dtype)
            wav_lens = wav_lens.to(self.device)
            enc, enc_len = self.model.forward(wavs, wav_lens)
            preds = self.greedy.decode(self.model.head, enc, enc_len)
            hyps.extend(preds)
        refs = refs[:len(hyps)]
        return wer(refs, hyps)

    def fit(self, train_loader, valid_loader, refs_valid: List[str], epochs: int = 10) -> Tuple[str, float]:
        best_path = ""
        best_wer = float("inf")
        for ep in range(1, epochs+1):
            self.model.train()
            run_loss = 0.0
            print(f"start epoch {ep}")
            for step, batch in enumerate(train_loader, 1):
                print(f"Working with batch {step}")
                loss = self.step_batch(batch)
                run_loss += loss
                if step % 50 == 0:
                    print(f"[epoch {ep} step {step}] loss={run_loss/step:.4f}")

            val_wer = self.validate(valid_loader, refs_valid)
            print(f"[epoch {ep}] train_loss={run_loss/max(1,len(train_loader)):.4f}  val_WER={val_wer:.4f}")

            # save per-epoch
            ep_path = self._save_ckpt(f"epoch{ep:02d}_wer{val_wer:.4f}")
            # track top-k
            self._track_topk(val_wer, ep_path)

            # best & last
            last_path = self._save_ckpt("last")
            if val_wer < best_wer:
                best_wer = val_wer
                best_path = self._save_ckpt("best")

            print(f"[epoch {ep}] saved: {ep_path} (last -> {last_path})")
        print(f"[done] best_WER={best_wer:.4f}  best_ckpt={best_path}")
        return best_path, best_wer

    def load_from_ckpt(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)

    @torch.no_grad()
    def evaluate_loader(self, loader, refs: List[str]) -> float:
        return self.validate(loader, refs)