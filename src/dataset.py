import re, heapq
from typing import List

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import gigaam
from gigaam.preprocess import SAMPLE_RATE as MODEL_SR

import datasets as hfds


class TextNormalizer:
    def __init__(self, numbers_mode: str = "keep", latin_mode: str = "keep"):
        assert numbers_mode in ("keep", "spell")
        assert latin_mode in ("keep", "translit")
        self.numbers_mode = numbers_mode
        self.latin_mode = latin_mode
        self._num2words = None
        if self.numbers_mode == "spell":
            try:
                from num2words import num2words  # type: ignore
                self._num2words = num2words
            except Exception:
                print("[warn] num2words не найден; числа останутся как есть.")
                self.numbers_mode = "keep"

        self._latin_digraphs = [
            ("sch", "сч"), ("sh", "ш"), ("ch", "ч"), ("yo", "ё"), ("yu", "ю"),
            ("ya", "я"), ("ts", "ц"), ("zh", "ж"), ("kh", "х"), ("ph", "ф"),
        ]
        self._latin_single = {
            "a":"а","b":"б","c":"к","d":"д","e":"е","f":"ф","g":"г","h":"х","i":"и",
            "j":"дж","k":"к","l":"л","m":"м","n":"н","o":"о","p":"п","q":"к","r":"р",
            "s":"с","t":"т","u":"у","v":"в","w":"в","x":"кс","y":"й","z":"з",
        }
        self._allowed_re = re.compile(r"[a-zа-яё0-9 ]", re.IGNORECASE)

    def _spell_numbers_ru(self, text: str) -> str:
        def repl(m):
            num = m.group(0)
            if self._num2words:
                try:
                    return self._num2words(int(num), lang="ru")
                except Exception:
                    return num
            return num
        return re.sub(r"\d+", repl, text)

    def _translit_latin_to_ru(self, text: str) -> str:
        s = text
        for pat, sub in self._latin_digraphs:
            s = re.sub(pat, sub, s)
        out = []
        for ch in s:
            out.append(self._latin_single.get(ch, ch))
        return "".join(out)

    def normalize(self, s: str) -> str:
        s = (s or "").strip().lower()
        if self.numbers_mode == "spell":
            s = self._spell_numbers_ru(s)
        if self.latin_mode == "translit":
            def translit_match(m): return self._translit_latin_to_ru(m.group(0))
            s = re.sub(r"[a-z]+", translit_match, s)
        s = "".join(ch if self._allowed_re.match(ch) else " " for ch in s)
        s = re.sub(r"\s+", " ", s).strip()
        return s


class CTCTextEncoder:
    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.char2id = {c: i for i, c in enumerate(vocab)}
    def encode(self, text: str) -> List[int]:
        return [self.char2id[c] for c in text if c in self.char2id]
    def decode(self, ids: List[int]) -> str:
        return "".join(self.vocab[i] for i in ids)

class FleursDataset(Dataset):
    def __init__(self, lang: str, split: str, normalizer: TextNormalizer):
        super().__init__()
        self.ds = hfds.load_dataset("google/fleurs", lang, split=split)
        self.norm = normalizer

    def __len__(self) -> int: 
        return len(self.ds)

    def __getitem__(self, idx: int):
        ex = self.ds[idx]
        arr = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        wav = torch.tensor(arr, dtype=torch.float32)
        if wav.ndim > 1: wav = wav.mean(dim=0)
        txt = ex.get("transcription") or ex.get("raw_transcription") or ""
        txt = self.norm.normalize(txt)
        return wav, sr, txt
    
class FleursCollate:
    def __init__(self, text_encoder: CTCTextEncoder, model_sr: int = MODEL_SR):
        self.text_encoder = text_encoder
        self.model_sr = model_sr
        self._resamplers = {}
    def _resample(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.model_sr: return wav
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.model_sr)
        return self._resamplers[sr](wav)
    def __call__(self, batch):
        wavs, wav_lens, targets, target_lens = [], [], [], []
        for wav, sr, txt in batch:
            w = self._resample(wav, sr)
            wavs.append(w)
            wav_lens.append(torch.tensor(w.numel(), dtype=torch.long))
            ids = torch.tensor(self.text_encoder.encode(txt), dtype=torch.long)
            targets.append(ids)
            target_lens.append(torch.tensor(len(ids), dtype=torch.long))
        max_len = max(w.numel() for w in wavs)
        padded = torch.zeros(len(wavs), max_len, dtype=torch.float32)
        for i, w in enumerate(wavs): padded[i, : w.numel()] = w
        targets_cat = torch.cat(targets) if targets else torch.empty(0, dtype=torch.long)
        return padded, torch.stack(wav_lens), targets_cat, torch.stack(target_lens)


def build_loaders(lang: str, batch_size:int, num_workers:int, model_name:str,
                  numbers_mode:str, latin_mode:str):
    normalizer = TextNormalizer(numbers_mode=numbers_mode, latin_mode=latin_mode)
    ds_tr = FleursDataset(lang, split="train",      normalizer=normalizer)
    ds_va = FleursDataset(lang, split="validation", normalizer=normalizer)

    # возьмём вокаб из модели (ожин раз)
    tmp_model = gigaam.load_model(model_name, fp16_encoder=False)
    vocab = list(tmp_model.cfg.decoding.vocabulary)
    del tmp_model
    text_encoder = CTCTextEncoder(vocab)

    collate = FleursCollate(text_encoder, model_sr=MODEL_SR)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, collate_fn=collate, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, collate_fn=collate, drop_last=False)
    refs_va = [ds_va[i][2] for i in range(len(ds_va))]
    return dl_tr, dl_va, refs_va, vocab, normalizer

def build_eval_loader(lang:str, split:str, batch_size:int, num_workers:int, vocab:List[str], normalizer:TextNormalizer):
    ds = FleursDataset(lang, split=split, normalizer=normalizer)
    text_encoder = CTCTextEncoder(vocab)
    collate = FleursCollate(text_encoder, model_sr=MODEL_SR)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, collate_fn=collate, drop_last=False)
    refs = [ds[i][2] for i in range(len(ds))]
    return dl, refs