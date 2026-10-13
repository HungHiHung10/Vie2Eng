import os
import sys
import time
import math
import re
from typing import Optional, List, Dict
from functools import lru_cache

sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from transformers import MarianTokenizer
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Tune PyTorch for CPU inference
if not torch.cuda.is_available():
    torch.set_num_threads(max(1, os.cpu_count() or 4))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Server] Using device: {device} (threads: {torch.get_num_threads()})")

# MarianTokenizer
TOKENIZER_NAME = 'Helsinki-NLP/opus-mt-en-vi'
print(f"[Server] Loading MarianTokenizer from {TOKENIZER_NAME}...")
tokenizer = MarianTokenizer.from_pretrained(TOKENIZER_NAME)
pad_idx = tokenizer.pad_token_id
eos_idx = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
print(f"[Server] Tokenizer loaded. Vocab size: {len(tokenizer)}, pad_idx: {pad_idx}, eos_idx: {eos_idx}")


# Model Architecture (matching trained checkpoints)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class Seq2SeqTransformerOptimized(nn.Module):
    def __init__(
        self,
        src_vocab_size: int = 53685,
        tgt_vocab_size: int = 53685,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_idx: int = 53684
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.fc_out.weight = self.tgt_embedding.weight

    def encode(self, src: torch.Tensor):
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        src_padding_mask = (src == self.pad_idx)
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask), src_padding_mask

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_padding_mask: torch.Tensor):
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt.device)
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)


def load_model_weights(path: str) -> Seq2SeqTransformerOptimized:
    print(f"[Server] Loading checkpoint: {path}...")
    t0 = time.time()
    model = Seq2SeqTransformerOptimized(pad_idx=pad_idx).to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck.get('model_state_dict', ck)
    model.load_state_dict(state)
    model.eval()
    print(f"[Server] Loaded {path} successfully in {time.time() - t0:.2f}s")
    return model


# Load both models
MODEL_VI_EN_PATH = "vienv2.pt"
MODEL_EN_VI_PATH = "enviv4.pt"

print(f"[Server] Initializing Vietnamese -> English model ({MODEL_VI_EN_PATH})...")
model_vien = load_model_weights(MODEL_VI_EN_PATH)

print(f"[Server] Initializing English -> Vietnamese model ({MODEL_EN_VI_PATH})...")
model_envi = load_model_weights(MODEL_EN_VI_PATH)


# Language Detection Logic
VI_CHARS_REGEX = re.compile(
    r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ'
    r'ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]',
    re.IGNORECASE
)

VI_COMMON_WORDS = {
    'và', 'của', 'là', 'có', 'được', 'trong', 'với', 'một', 'các', 'người',
    'không', 'đã', 'này', 'cho', 'những', 'khi', 'thì', 'mà', 'ở', 'từ',
    'tôi', 'bạn', 'chúng', 'anh', 'chị', 'em', 'họ', 'gì', 'thế', 'nào',
    'nhiều', 'rất', 'về', 'lại', 'như', 'đến', 'ra', 'phải', 'biết', 'sẽ',
    'cũng', 'làm', 'sau', 'trên', 'qua', 'năm', 'ngày', 'nói', 'theo', 'hơn',
    'việt', 'nam', 'bình', 'xin', 'chào', 'cảm', 'ơn', 'khỏe'
}

EN_COMMON_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'hello', 'how', 'are', 'is', 'love', 'fast', 'final', 'exam', 'very'
}

VI_UNACCENTED_WORDS = {
    'xin', 'chao', 'cam', 'on', 'tam', 'biet', 'toi', 'ban', 'chung',
    'anh', 'chi', 'em', 'ho', 'gi', 'the', 'nao', 'troi', 'dep', 'hoc',
    'sinh', 'vien', 'yeu', 'rat', 'nhieu', 'khong', 'duoc', 'nguoi',
    'viet', 'nam', 'khi', 'nao', 'o', 'dau', 'lam', 'sao'
}


def detect_language(text: str) -> str:
    """Detect if text is Vietnamese ('vi') or English ('en')."""
    if not text.strip():
        return 'en'
    
    # 1. Check for specific Vietnamese diacritics
    if VI_CHARS_REGEX.search(text):
        return 'vi'
    
    # 2. Check word tokens against dictionary
    words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
    if not words:
        return 'en'
    
    vi_matches = sum(1 for w in words if w in VI_COMMON_WORDS or w in VI_UNACCENTED_WORDS)
    en_matches = sum(1 for w in words if w in EN_COMMON_WORDS)
    
    if vi_matches > en_matches:
        return 'vi'
    return 'en'


# Decoding Algorithms
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


@torch.no_grad()
def greedy_decode_sentence(text: str, model: Seq2SeqTransformerOptimized, max_len: int = 60) -> str:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_len)
    src = inputs['input_ids'].to(device)
    memory, src_padding_mask = model.encode(src)

    bos_id = pad_idx
    ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    for step in range(max_len):
        dec_out = model.decode(ys, memory, src_padding_mask)
        logits = model.fc_out(dec_out[:, -1, :])
        # Prevent immediate termination on step 0
        if step == 0:
            logits[:, eos_idx] = -float('inf')
            logits[:, pad_idx] = -float('inf')
        next_tok = torch.argmax(logits, dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_tok]], device=device)], dim=1)
        if next_tok == eos_idx:
            break

    decoded = tokenizer.decode(ys[0, 1:], skip_special_tokens=True).strip()
    return clean_translation(decoded)


@torch.no_grad()
def beam_search_sentence(
    text: str,
    model: Seq2SeqTransformerOptimized,
    beam_size: int = 4,
    max_len: int = 60,
    no_repeat_ngram_size: int = 3,
    length_penalty_alpha: float = 0.6
) -> str:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_len)
    src = inputs['input_ids'].to(device)
    memory, src_padding_mask = model.encode(src)

    bos_id = pad_idx
    candidates = [([bos_id], 0.0)]
    completed = []

    for step in range(max_len):
        tgt_tensor = torch.tensor([c[0] for c in candidates], device=device)
        cur_beam_size = tgt_tensor.size(0)
        mem_exp = memory.repeat(cur_beam_size, 1, 1)
        mask_exp = src_padding_mask.repeat(cur_beam_size, 1)

        dec_out = model.decode(tgt_tensor, mem_exp, mask_exp)
        logits = model.fc_out(dec_out[:, -1, :])
        log_probs = torch.log_softmax(logits, dim=-1)

        # Prevent premature termination on early steps
        if step == 0:
            log_probs[:, eos_idx] = -float('inf')
            log_probs[:, pad_idx] = -float('inf')

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
            if tokens[-1] == eos_idx:
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

    decoded = tokenizer.decode(best_tokens[1:], skip_special_tokens=True).strip()
    return clean_translation(decoded)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences/paragraphs while keeping structure."""
    paragraphs = text.split('\n')
    units = []
    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            units.append("")
            continue
        parts = re.split(r'(?<=[.!?。])\s+', p_clean)
        for part in parts:
            if part.strip():
                units.append(part.strip())
    return units


def translate_full_text(
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
                res = greedy_decode_sentence(s.strip(), model, max_len=max_len)
            else:
                res = beam_search_sentence(s.strip(), model, beam_size=beam_size, max_len=max_len)
            
            # Fallback to greedy if beam search returned empty
            if not res:
                res = greedy_decode_sentence(s.strip(), model, max_len=max_len)
            
            if res:
                trans_sentences.append(res)
        translated_paragraphs.append(" ".join(trans_sentences))

    return "\n".join(translated_paragraphs)


# Simple LRU Cache
translation_cache: Dict[str, str] = {}


# FastAPI Application
app = FastAPI(
    title="Google Translate - PhoMT Marian NMT",
    description="Dual Transformer Translation System for Vietnamese and English",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "auto"  # "auto", "vi", "en"
    target_lang: str = "auto"  # "auto", "vi", "en"
    method: str = "beam"       # "beam" or "greedy"
    beam_size: int = 4
    max_length: int = 60


class TranslationResponse(BaseModel):
    translated_text: str
    detected_lang: str
    source_lang: str
    target_lang: str
    model_name: str
    latency_ms: float
    method: str
    from_cache: bool


@app.get("/", response_class=FileResponse)
def get_index():
    return FileResponse("templates/index.html")


@app.get("/logo.png", response_class=FileResponse)
def get_logo():
    return FileResponse("static/logo.png")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "models": {
            "vi_en": MODEL_VI_EN_PATH,
            "en_vi": MODEL_EN_VI_PATH
        },
        "cache_size": len(translation_cache)
    }


@app.post("/api/translate", response_model=TranslationResponse)
def api_translate(req: TranslationRequest):
    raw_text = req.text.strip()
    if not raw_text:
        return TranslationResponse(
            translated_text="",
            detected_lang="vi",
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            model_name="none",
            latency_ms=0.0,
            method=req.method,
            from_cache=False
        )

    t0 = time.time()
    
    # 1. Determine Source and Target Language
    detected = detect_language(raw_text)
    
    if req.source_lang == "auto":
        src_lang = detected
    else:
        src_lang = req.source_lang

    if req.target_lang == "auto" or req.target_lang == src_lang:
        tgt_lang = "en" if src_lang == "vi" else "vi"
    else:
        tgt_lang = req.target_lang

    # 2. Select Model
    if src_lang == "vi" and tgt_lang == "en":
        model = model_vien
        model_name = "vienv2.pt (PhoMT Vi -> En)"
    else:
        model = model_envi
        model_name = "enviv4.pt (PhoMT En -> Vi)"

    # 3. Check Cache
    cache_key = f"{src_lang}:{tgt_lang}:{req.method}:{req.beam_size}:{raw_text}"
    if cache_key in translation_cache:
        latency = (time.time() - t0) * 1000.0
        return TranslationResponse(
            translated_text=translation_cache[cache_key],
            detected_lang=detected,
            source_lang=src_lang,
            target_lang=tgt_lang,
            model_name=model_name,
            latency_ms=round(latency, 2),
            method=req.method,
            from_cache=True
        )

    # 4. Perform Translation
    beam_sz = max(1, min(6, req.beam_size))
    method = "greedy" if (req.method == "greedy" or beam_sz == 1) else "beam"
    
    translated = translate_full_text(
        text=raw_text,
        model=model,
        method=method,
        beam_size=beam_sz,
        max_len=req.max_length
    )

    # Cache result (limit cache to 500 entries)
    if len(translation_cache) > 500:
        translation_cache.clear()
    translation_cache[cache_key] = translated

    latency = (time.time() - t0) * 1000.0

    return TranslationResponse(
        translated_text=translated,
        detected_lang=detected,
        source_lang=src_lang,
        target_lang=tgt_lang,
        model_name=model_name,
        latency_ms=round(latency, 2),
        method=method,
        from_cache=False
    )


if __name__ == "__main__":
    import uvicorn
    print("\n[Server] Starting FastAPI server on http://localhost:8000 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
