"""Lazy PhoMT checkpoint loading and the original decoding algorithms."""

import os
import re
from threading import Lock
from typing import List

import torch
from transformers import MarianTokenizer

from ..config import BASE_DIR, LEGACY_MODELS, LEGACY_TOKENIZER
from .transformer import Seq2SeqTransformerOptimized


def get_banned_tokens(tokens: List[int], ngram_size: int = 3) -> List[int]:
    if len(tokens) < ngram_size - 1:
        return []
    prefix = tuple(tokens[-(ngram_size - 1):])
    banned = []
    for i in range(len(tokens) - ngram_size + 1):
        if tuple(tokens[i : i + ngram_size - 1]) == prefix:
            banned.append(tokens[i + ngram_size - 1])
    return banned


def clean_translation(text: str) -> str:
    text = text.strip()
    # Strip leading dots, commas, hyphens, tildes produced by tokenizer
    text = re.sub(r'^[.\s,;\-–—~:]+', '', text).strip()
    if not text:
        return ""
    # Capitalize first character
    if text[0].islower():
        text = text[0].upper() + text[1:]
    return text


class LegacyBackend:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu':
            torch.set_num_threads(max(1, os.cpu_count() or 4))
        self.tokenizer = None
        self.pad_idx = None
        self.eos_idx = None
        self._models = {}
        self._load_lock = Lock()

    def get_model(self, direction):
        with self._load_lock:
            if self.tokenizer is None:
                try:
                    self.tokenizer = MarianTokenizer.from_pretrained(LEGACY_TOKENIZER, local_files_only=True)
                except OSError:
                    self.tokenizer = MarianTokenizer.from_pretrained(LEGACY_TOKENIZER)
                self.pad_idx = self.tokenizer.pad_token_id
                self.eos_idx = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
            if direction not in self._models:
                path = BASE_DIR / LEGACY_MODELS[direction]
                model = Seq2SeqTransformerOptimized(pad_idx=self.pad_idx).to(self.device)
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                model.eval()
                self._models[direction] = model
            return self._models[direction]

    def translate(self, text, direction, method='beam', beam_size=4, max_length=60):
        return self.translate_full_text(text, self.get_model(direction), method, beam_size, max_length)

    @torch.no_grad()
    def greedy_decode_sentence(self, text: str, model: Seq2SeqTransformerOptimized, max_len: int = 60) -> str:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=max_len)
        src = inputs['input_ids'].to(self.device)
        memory, src_padding_mask = model.encode(src)

        bos_id = self.pad_idx
        ys = torch.tensor([[bos_id]], dtype=torch.long, device=self.device)

        for step in range(max_len):
            dec_out = model.decode(ys, memory, src_padding_mask)
            logits = model.fc_out(dec_out[:, -1, :])
            # Prevent immediate termination on step 0
            if step == 0:
                logits[:, self.eos_idx] = -float('inf')
                logits[:, self.pad_idx] = -float('inf')
            next_tok = torch.argmax(logits, dim=-1).item()
            ys = torch.cat([ys, torch.tensor([[next_tok]], device=self.device)], dim=1)
            if next_tok == self.eos_idx:
                break

        decoded = self.tokenizer.decode(ys[0, 1:], skip_special_tokens=True).strip()
        return clean_translation(decoded)

    @torch.no_grad()
    def beam_search_sentence(
        self,
        text: str,
        model: Seq2SeqTransformerOptimized,
        beam_size: int = 4,
        max_len: int = 60,
        no_repeat_ngram_size: int = 3,
        length_penalty_alpha: float = 0.6
    ) -> str:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=max_len)
        src = inputs['input_ids'].to(self.device)
        memory, src_padding_mask = model.encode(src)

        bos_id = self.pad_idx
        candidates = [([bos_id], 0.0)]
        completed = []

        for step in range(max_len):
            tgt_tensor = torch.tensor([c[0] for c in candidates], device=self.device)
            cur_beam_size = tgt_tensor.size(0)
            mem_exp = memory.repeat(cur_beam_size, 1, 1)
            mask_exp = src_padding_mask.repeat(cur_beam_size, 1)

            dec_out = model.decode(tgt_tensor, mem_exp, mask_exp)
            logits = model.fc_out(dec_out[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=-1)

            # Prevent premature termination on early steps
            if step == 0:
                log_probs[:, self.eos_idx] = -float('inf')
                log_probs[:, self.pad_idx] = -float('inf')

            all_cands = []
            for beam_idx, (tokens, cum_score) in enumerate(candidates):
                banned = get_banned_tokens(tokens, no_repeat_ngram_size)
                if banned:
                    log_probs[beam_idx, banned] = -float('inf')
                topk_lp, topk_tok = torch.topk(log_probs[beam_idx], k=beam_size)
                for lp, tok in zip(topk_lp.tolist(), topk_tok.tolist()):
                    all_cands.append((tokens + [tok], cum_score + lp))

            all_cands = sorted(all_cands, key=lambda x: x[1], reverse=True)
            candidates = []
            for tokens, score in all_cands:
                if tokens[-1] == self.eos_idx:
                    l = len(tokens) - 1
                    lp = ((5.0 + l) / 6.0) ** length_penalty_alpha
                    completed.append((tokens, score / lp))
                else:
                    candidates.append((tokens, score))
                if len(candidates) == beam_size:
                    break
            if len(completed) >= beam_size or len(candidates) == 0:
                break

        if completed:
            best_tokens, _ = max(completed, key=lambda x: x[1])
        elif candidates:
            best_tokens, _ = max(candidates, key=lambda x: x[1] / (((5.0 + len(x[0]) - 1) / 6.0) ** length_penalty_alpha))
        else:
            best_tokens = [bos_id]

        decoded = self.tokenizer.decode(best_tokens[1:], skip_special_tokens=True).strip()
        return clean_translation(decoded)

    def translate_full_text(
        self,
        text: str,
        model: Seq2SeqTransformerOptimized,
        method: str = "beam",
        beam_size: int = 4,
        max_len: int = 60
    ) -> str:
        """Translates text chunk by chunk to preserve long text without degradation."""
        paragraphs = text.split('\n')
        translated_paragraphs = []

        for p in paragraphs:
            if not p.strip():
                translated_paragraphs.append("")
                continue

            sentences = re.split(r'(?<=[.!?])\s+', p.strip())
            trans_sentences = []
            for s in sentences:
                if not s.strip():
                    continue
                if method == "greedy":
                    res = self.greedy_decode_sentence(s.strip(), model, max_len=max_len)
                else:
                    res = self.beam_search_sentence(s.strip(), model, beam_size=beam_size, max_len=max_len)

                # Fallback to greedy if beam search returned empty
                if not res:
                    res = self.greedy_decode_sentence(s.strip(), model, max_len=max_len)

                if res:
                    trans_sentences.append(res)
            translated_paragraphs.append(" ".join(trans_sentences))

        return "\n".join(translated_paragraphs)
