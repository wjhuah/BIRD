import os, random, math, json, shutil
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification,
    AutoModel, Trainer, TrainingArguments, set_seed,
    DataCollatorForLanguageModeling, AddedToken
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import wandb, regex

UNK_PLACEHOLDERS = [
    "[UNK-00020-0]","[UNK-02836-0]","[UNK-04042-0]","[UNK-04667-0]",
    "[UNK-04684-0]","[UNK-04688-0]","[UNK-04694-0]","[UNK-04694-1]",
    "[UNK-05392-0]","[UNK-05396-0]","[UNK-05400-0]","[UNK-05416-0]",
    "[UNK-05423-0]","[UNK-05981-0]","[UNK-05989-0]","[UNK-05998-0]",
    "[UNK-06000-0]","[UNK-06004-0]","[UNK-06007-0]","[UNK-06515-0]",
    "[UNK-06515-1]","[UNK-09713-0]","[UNK-09733-0]","[UNK-09733-1]",
    "[UNK-10162-0]","[UNK-10167-0]","[UNK-10167-1]","[UNK-10175-0]"
]
SPECIAL_TOKENS = {
    "additional_special_tokens": [
        AddedToken(t, single_word=True, lstrip=False, rstrip=False, normalized=False)
        for t in UNK_PLACEHOLDERS
    ]
}
HANZI_PATTERN = regex.compile(r'^\p{IsHan}$')
def is_hanzi_or_unk(token: str) -> bool:
    if token in UNK_PLACEHOLDERS: return True
    return bool(HANZI_PATTERN.match(token))

def register_unk_placeholders(tokenizer, model):
    num_added = tokenizer.add_special_tokens(SPECIAL_TOKENS)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        try: model.tie_weights()
        except Exception as e: print(f"[warn] tie_weights failed: {e}")
    return tokenizer, model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try: torch.set_float32_matmul_precision("high")
except: pass

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

ERROR_DIR = os.path.join(BASE_DIR, "errors")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(ERROR_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def result_path(*parts):
    """Construct a path inside RESULTS_DIR."""
    return os.path.join(RESULTS_DIR, *parts)

def error_path(*parts):
    """Construct a path inside ERROR_DIR."""
    return os.path.join(ERROR_DIR, *parts)

GLYPH_CHAIN_FILES = [
    os.path.join(DATA_DIR, "Yin.edge"),
    os.path.join(DATA_DIR, "Western_Zhou.edge"),
    os.path.join(DATA_DIR, "SpringAutumn_WarringStates.edge"),
]

# ====== MODEL SELECTION ======
# Possible options:
#   "SIKU-BERT/sikuroberta"             → SikuRoBERTa Chinese historical corpora
#   "bert-base-multilingual-cased"  → mBERT
#   "xlm-roberta-base"     → XLM-R base
#   "xlm-roberta-large"    → XLM-R large (default)
#   or any local checkpoint directory
MODEL_PATH       = "xlm-roberta-large"
DAPT_TRAIN_PATH  = os.path.join(DATA_DIR, "dapt.txt")
TAPT_TRAIN_PATH  = os.path.join(DATA_DIR, "tapt_train.txt")
TAPT_TEST_PATH   = os.path.join(DATA_DIR, "tapt_test.txt")

BLOCK_SIZE       = 128
BATCH_SIZE       = 32
WEIGHT_DECAY     = 0.01
SEED             = 42
MAX_CHARS        = 512
DROPOUT          = 0.10
MIN_GROUP_SIZE         = 2
GLYPH_MASK_BIAS        = 2.2
SAME_GROUP_RAND        = 0.6
GN_ALPHA_INIT          = 0.25
SMOOTH_ALPHA_MAX       = 0.10
BETA_WARMUP_RATIO      = 0.25
REP_SAMPLE_PER_GROUP   = 50
DAPT_EPOCHS            = 10
GLOBAL_RNG             = np.random.default_rng(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); set_seed(SEED)

def build_variant_groups():
    groups = defaultdict(set)
    char_to_bases = defaultdict(set)
    for path in GLYPH_CHAIN_FILES:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                var, base = parts
                groups[base].add(var)
                groups[base].add(base)
    for base, vs in groups.items():
        for ch in vs:
            char_to_bases[ch].add(base)
    all_chars = set(ch for vs in groups.values() for ch in vs)
    return groups, char_to_bases, all_chars

VARIANT_GROUPS, CHAR_TO_BASES, ALL_GLYPH_CHARS = build_variant_groups()

def build_cluster_index():
    base_list = sorted(list(VARIANT_GROUPS.keys()))
    base2cid = {b:i for i,b in enumerate(base_list)}
    char2cid = {}
    for ch, bases in CHAR_TO_BASES.items():
        if ch in base2cid: char2cid[ch] = base2cid[ch]
        else: char2cid[ch] = base2cid[sorted(list(bases))[0]]
    return base2cid, char2cid, len(base2cid)

BASE2CID, CHAR2CID, NUM_CLUSTERS = build_cluster_index()
CID2BASE = {cid: base for base, cid in BASE2CID.items()}

def read_lines(path, limit_len=MAX_CHARS):
    out=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if not s: continue
            if len(s)>limit_len: s=s[:limit_len]
            out.append(s)
    return out

def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for s in lines: f.write(s+"\n")

def build_hf_text_dataset(tokenizer, txt_path):
    ds = load_dataset("text", data_files={"train": txt_path})["train"]
    def tok(b):
        return tokenizer(b["text"], truncation=True, max_length=BLOCK_SIZE, return_special_tokens_mask=True)
    ds = ds.map(tok, batched=True, remove_columns=["text"])
    ds = ds.filter(lambda x: x is not None and "input_ids" in x and len(x["input_ids"])>0)
    return ds

def pick_new_glyphs_to_holdout(tokenizer, ratio):
    vocab = tokenizer.get_vocab()
    new_chars = sorted([ch for ch in ALL_GLYPH_CHARS if ch not in vocab])
    if len(new_chars) == 0:
        print("No new glyph characters found; the advantage of GN new-only alignment will be reduced.")
    rng = np.random.default_rng(SEED)
    by_group = defaultdict(list)
    for ch in new_chars:
        gid = CHAR2CID.get(ch, None)
        if gid is not None: by_group[gid].append(ch)
    holdout = set()
    for gid, arr in by_group.items():
        if len(arr) == 0: continue
        k = max(1, int(round(len(arr) * ratio)))
        picks = rng.choice(arr, size=min(k, len(arr)), replace=False)
        holdout.update(picks)
    seen_new = set(new_chars) - holdout
    return set(new_chars), holdout, seen_new

def filter_remove_holdout(lines, holdout_chars):
    holdout_chars = set(holdout_chars)
    return [s for s in lines if not any((ch in holdout_chars) for ch in s)]

def build_test_with_holdout(test_lines, holdout_chars, min_needed=300):
    sel = [s for s in test_lines if any((ch in holdout_chars) for ch in s)]
    real_count = len(sel)
    extra = []
    if real_count < min_needed:
        vocab_holdout = list(holdout_chars)
        rng = GLOBAL_RNG
        gid2hold = defaultdict(list)
        for ch in vocab_holdout:
            gid = CHAR2CID.get(ch, None)
            if gid is not None: gid2hold[gid].append(ch)
        for s in test_lines:
            for i, ch in enumerate(s):
                gid = CHAR2CID.get(ch, None)
                if gid is None: continue
                if gid in gid2hold:
                    rep = rng.choice(gid2hold[gid])
                    if rep != ch:
                        extra.append(s[:i] + rep + s[i+1:])
                        break
            if len(sel) + len(extra) >= min_needed: break
    print(f"[build_test] Original samples containing holdout: {real_count}, supplemented by replacements: {len(extra)}")
    return sel + extra

def freeze_layers(model, num_freeze=6):
    enc = None
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        enc = model.bert.encoder
    elif hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        enc = model.roberta.encoder
    else:
        print("[warn] Encoder not found; skipping layer freezing."); return model
    layers = getattr(enc, "layer", None)
    if layers is None:
        print("[warn] Encoder layers not found; skipping layer freezing."); return model
    for p in layers[:num_freeze].parameters(): p.requires_grad = False
    print(f"Froze {min(num_freeze, len(layers))} layers.")
    return model

def expand_tokenizer_with_glyphs(tokenizer, model, glyph_chars, groups, mean_resizing_disable=True):
    vocab = tokenizer.get_vocab()
    new_chars = [ch for ch in glyph_chars if ch not in vocab]
    if not new_chars: return 0, set()
    added = tokenizer.add_tokens(new_chars)
    new_set = set(new_chars)
    print(f"[expand_tokenizer] Added {added} new glyph tokens.")
    try:
        if mean_resizing_disable:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        else:
            model.resize_token_embeddings(len(tokenizer))
    except TypeError:
        model.resize_token_embeddings(len(tokenizer))
    try:
        if hasattr(model, "cls") and hasattr(model.cls, "predictions"):
            bias = getattr(model.cls.predictions, "bias", None)
            if bias is not None and bias.shape[0] != len(tokenizer):
                new_bias = torch.nn.Parameter(bias.new_zeros(len(tokenizer)))
                new_bias.data[:bias.shape[0]] = bias.data
                model.cls.predictions.bias = new_bias
        if hasattr(model, "lm_head"):
            bias = getattr(model.lm_head, "bias", None)
            if bias is not None and bias.shape[0] != len(tokenizer):
                new_bias = torch.nn.Parameter(bias.new_zeros(len(tokenizer)))
                new_bias.data[:bias.shape[0]] = bias.data
                model.lm_head.bias = new_bias
    except Exception as e:
        print(f"[warn] bias resize failed: {e}")
    try: model.tie_weights()
    except Exception as e: print(f"[warn] tie_weights failed after expand: {e}")
    return added, new_set

def gn_new_only_align(model, tokenizer, new_set, alpha=GN_ALPHA_INIT):
    emb = model.get_input_embeddings().weight.data
    vocab = tokenizer.get_vocab()
    moved=0
    for base, variants in VARIANT_GROUPS.items():
        anchor_ids = [vocab[ch] for ch in variants if ch in vocab and ch not in new_set]
        target_ids = [vocab[ch] for ch in variants if ch in vocab and ch in new_set]
        if not target_ids or not anchor_ids: continue
        centroid = emb[anchor_ids].mean(0, keepdim=True).to(emb.device)
        with torch.no_grad():
            emb[torch.tensor([vocab[c] for c in variants if c in new_set], device=emb.device)] = \
                alpha*centroid + (1-alpha)*emb[torch.tensor([vocab[c] for c in variants if c in new_set], device=emb.device)]
        moved += len(target_ids)
    print(f"[GN new-only] moved {moved} tokens.")

def build_tokenid2gid_and_gid2tok(tokenizer):
    vocab = tokenizer.get_vocab()
    V = len(vocab)
    tokenid2gid = torch.full((V,), -1, dtype=torch.long)
    gid2tok = defaultdict(list)
    for ch, gid in CHAR2CID.items():
        if ch in vocab:
            tid = vocab[ch]
            tokenid2gid[tid] = gid
            gid2tok[gid].append(tid)
    gid2tok = {g: torch.tensor(toks, dtype=torch.long) for g,toks in gid2tok.items() if len(toks)>=MIN_GROUP_SIZE}
    return tokenid2gid, gid2tok

class GroupBiasedMLMCollator:
    def __init__(self, tokenizer, mlm_probability, glyph_token_ids,
                 gid_of_token, gid2tok, same_group_rand=0.6,
                 glyph_bias=2.2, stride=None):
        self.tokenizer = tokenizer
        self.mlm_probability = float(mlm_probability)
        self.glyph_token_ids = set(int(t) for t in glyph_token_ids)
        self.gid_of_token = gid_of_token
        self.gid2tok = gid2tok
        self.same_group_rand = float(same_group_rand)
        self.glyph_bias = float(glyph_bias)
        self.vocab_size = len(tokenizer)
        self.stride = stride

    def __call__(self, features):
        batch = self.tokenizer.pad(features, return_tensors="pt")
        input_ids = batch["input_ids"]
        attn = batch.get("attention_mask", torch.ones_like(input_ids))
        spec = batch.get("special_tokens_mask", torch.zeros_like(input_ids))
        labels = input_ids.clone()
        B, L = input_ids.shape
        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)

        for b in range(B):
            cand = (attn[b] == 1) & (spec[b] == 0)
            tokens = [self.tokenizer.convert_ids_to_tokens(int(t)) for t in input_ids[b]]
            hanzi_mask = torch.tensor([is_hanzi_or_unk(tok) for tok in tokens],
                                      dtype=torch.bool, device=input_ids.device)
            cand = cand & hanzi_mask
            idx = torch.nonzero(cand, as_tuple=False).flatten()
            if idx.numel() == 0: continue

            if self.stride and self.stride > 1:
                new_idx = []
                for pos in idx.tolist():
                    tok = self.tokenizer.convert_ids_to_tokens(int(input_ids[b, pos].item()))
                    if not is_hanzi_or_unk(tok): continue
                    if pos == L - 1:
                        if pos - 1 in idx: new_idx.append(pos - 1)
                        continue
                    new_idx.append(pos)
                idx = torch.tensor(new_idx, device=input_ids.device)

            num_cand = idx.numel()
            if num_cand == 0: continue

            k = max(1, int(round(self.mlm_probability * num_cand)))
            w = torch.ones(num_cand, dtype=torch.float)
            toks = input_ids[b, idx]
            is_glyph = torch.tensor([int(t.item()) in self.glyph_token_ids for t in toks],
                                    dtype=torch.bool)
            w[is_glyph] *= self.glyph_bias
            probs = w / (w.sum() + 1e-9)
            sel_ind = torch.multinomial(probs, num_samples=min(k, num_cand), replacement=False)
            sel_pos = idx[sel_ind]
            mask_positions[b, sel_pos] = True

        labels[~mask_positions] = -100
        mask80 = mask_positions & (torch.rand_like(input_ids, dtype=torch.float) < 0.8)
        input_ids[mask80] = self.tokenizer.mask_token_id
        remain = mask_positions & ~mask80
        rand10 = remain & (torch.rand_like(input_ids, dtype=torch.float) < 0.5)
        if rand10.any():
            idxs = torch.nonzero(rand10, as_tuple=True)
            for (b, j) in zip(idxs[0].tolist(), idxs[1].tolist()):
                orig = int(input_ids[b, j].item())
                gid = int(self.gid_of_token[orig].item()) if orig < len(self.gid_of_token) else -1
                choose_same = (GLOBAL_RNG.random() < self.same_group_rand) and (gid in self.gid2tok) and (self.gid2tok[gid].numel() > 1)
                if choose_same:
                    cand = self.gid2tok[gid].tolist()
                    cand = [c for c in cand if c != orig] or cand
                    new_tok = GLOBAL_RNG.choice(cand)
                else:
                    new_tok = orig
                    while new_tok == orig:
                        new_tok = int(GLOBAL_RNG.integers(self.vocab_size))
                input_ids[b, j] = new_tok

        batch.pop("special_tokens_mask", None)
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch

class SmoothGNTrainer(Trainer):
    def __init__(self, *args, tokenid2gid=None, gid2tok=None,
                 smooth_alpha_max=0.10, beta_warmup_ratio=0.25,
                 smooth_glyph_only=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenid2gid = tokenid2gid
        self._gid2tok = gid2tok
        self.smooth_alpha_max = float(smooth_alpha_max)
        self.beta_warmup_ratio = float(beta_warmup_ratio)
        self.smooth_glyph_only = bool(smooth_glyph_only)

    def _sched_alpha(self):
        gs = max(self.state.global_step, 0); ts = max(self.state.max_steps, 1)
        cos = 0.5 * (1 + math.cos(math.pi * gs / ts))
        warm = int(self.beta_warmup_ratio * ts)
        if gs < warm: return self.smooth_alpha_max * (gs/max(1,warm))
        return self.smooth_alpha_max * cos

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**{k:v for k,v in inputs.items() if k in ["input_ids","attention_mask","token_type_ids","labels"]})
        logits = outputs.logits
        labels = inputs.get("labels", None)
        loss = outputs.loss
        if labels is not None:
            mask_pos = (labels != -100)
            if mask_pos.any():
                log_probs = torch.log_softmax(logits[mask_pos], dim=-1)
                gold = labels[mask_pos]
                gids = self._tokenid2gid.to(log_probs.device)[gold]
                alpha = self._sched_alpha()
                if alpha > 0:
                    terms=[]
                    for i in range(gold.size(0)):
                        gid = int(gids[i].item())
                        if self.smooth_glyph_only and gid < 0: continue
                        tok_ids = self._gid2tok.get(gid, None)
                        if tok_ids is None or tok_ids.numel() < MIN_GROUP_SIZE: continue
                        terms.append(log_probs[i, tok_ids.to(log_probs.device)].mean())
                    if len(terms)>0:
                        smooth_mean = torch.stack(terms).mean()
                        loss = (1.0 - alpha) * loss + alpha * (-smooth_mean)
        return (loss, outputs) if return_outputs else loss

@torch.no_grad()
def evaluate_model(model, tokenizer, file_path, k=(1,5,10), stride=8, batch_size=32, log_path=None):
    model.eval(); model.to(device)
    vocab = tokenizer.get_vocab()
    inv_vocab = {i:t for t,i in vocab.items()}
    total  = {kk:0 for kk in k}; correct= {kk:0 for kk in k}
    g_total= {kk:0 for kk in k}; g_corr = {kk:0 for kk in k}
    errors, corrects = [], []
    if not os.path.exists(file_path): raise FileNotFoundError(file_path)
    lines = [l.strip() for l in open(file_path,'r',encoding='utf-8') if l.strip()]
    valid_token_ids = torch.tensor(
        [tid for tok, tid in vocab.items() if is_hanzi_or_unk(tok)],
        dtype=torch.long, device=device
    )
    valid_mask = torch.full((len(vocab),), float("-inf"), device=device)
    valid_mask[valid_token_ids] = 0.0

    for start in range(0, len(lines), batch_size):
        enc = tokenizer(lines[start:start+batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=BLOCK_SIZE)
        input_ids = enc["input_ids"].to(device)
        attention = enc["attention_mask"].to(device)
        B, L = input_ids.shape
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        for r in range(stride):
            to_mask = (pos % stride == r) & (pos > 0) & (pos < L-1) & attention.bool()
            if not to_mask.any(): continue
            hanzi_masks=[]
            for b in range(B):
                toks = [inv_vocab.get(int(t.item()), "") for t in input_ids[b]]
                m = torch.tensor([is_hanzi_or_unk(t) for t in toks], dtype=torch.bool, device=device)
                hanzi_masks.append(m.unsqueeze(0))
            to_mask = to_mask & torch.cat(hanzi_masks,0)
            if not to_mask.any(): continue

            masked = input_ids.clone()
            gold   = input_ids[to_mask]
            masked[to_mask] = tokenizer.mask_token_id
            logits = model(input_ids=masked, attention_mask=attention).logits
            logits = logits + valid_mask
            probs  = torch.softmax(logits[to_mask], dim=-1)
            batch_idx, token_idx = torch.nonzero(to_mask, as_tuple=True)

            for kk in k:
                topk_idx = torch.topk(probs, kk, dim=-1).indices
                total[kk]  += gold.numel()
                correct[kk]+= (topk_idx == gold.unsqueeze(-1)).any(dim=-1).sum().item()
                for i in range(gold.size(0)):
                    true_id = int(gold[i].item())
                    ch = inv_vocab.get(true_id, None)
                    gid = CHAR2CID.get(ch, -1) if ch is not None else -1
                    if gid == -1: continue
                    g_total[kk] += 1
                    base = CID2BASE.get(gid, None)
                    if base is None: continue
                    group_tok = [vocab[c] for c in VARIANT_GROUPS[base] if c in vocab]
                    preds = topk_idx[i].tolist()
                    if any(p in group_tok for p in preds):
                        g_corr[kk] += 1
                if kk == 5:
                    for i in range(gold.size(0)):
                        true_id = int(gold[i].item())
                        preds = topk_idx[i].tolist()
                        topk_txt = [inv_vocab.get(int(x), "") for x in preds]
                        row = {
                            "true_char": inv_vocab.get(true_id, ""),
                            "pred_topk": "".join(topk_txt),
                            "pred1": topk_txt[0] if topk_txt else "",
                            "context": tokenizer.decode(masked[batch_idx[i]], skip_special_tokens=True)
                        }
                        if true_id not in preds: errors.append(row)
                        else: corrects.append(row)

    results = {}
    for kk in k:
        results[f"Exact@{kk}"] = round(100*correct[kk]/max(1,total[kk]), 2)
        results[f"Group@{kk}"] = round(100*g_corr[kk]/max(1,g_total[kk]), 2)
    if log_path:
        if errors:
            pd.DataFrame(errors).to_csv(log_path, index=False, encoding="utf-8")
            print(f"[eval] Error log: {len(errors)} entries → {log_path}")
        if corrects:
            correct_path = log_path.replace("errors.csv", "correct.csv")
            pd.DataFrame(corrects).to_csv(correct_path, index=False, encoding="utf-8")
            print(f"[eval] Correct log: {len(corrects)} entries → {correct_path}")
    return results

def embedding_cluster_report(model, tokenizer):
    W = model.get_input_embeddings().weight.data.detach().cpu()
    vocab = tokenizer.get_vocab()
    per_group = {}
    for base, vs in VARIANT_GROUPS.items():
        toks = [vocab[ch] for ch in vs if ch in vocab][:REP_SAMPLE_PER_GROUP]
        if len(toks) < MIN_GROUP_SIZE: continue
        vecs = F.normalize(W[toks].to(torch.float32), dim=-1)
        center = vecs.mean(0, keepdim=True)
        intra = (vecs @ center.T).mean().item()
        per_group[base] = dict(intra=intra, size=len(toks))
    intra_avg = np.mean([v["intra"] for v in per_group.values()]) if per_group else 0.0
    centers=[]
    for base, vs in VARIANT_GROUPS.items():
        toks = [vocab[ch] for ch in vs if ch in vocab][:REP_SAMPLE_PER_GROUP]
        if len(toks) < MIN_GROUP_SIZE: continue
        vecs = F.normalize(W[toks].to(torch.float32), dim=-1)
        centers.append(vecs.mean(0, keepdim=True))
    inter = 0.0
    if len(centers)>=2:
        C = torch.cat(centers,0).to(torch.float32)
        sim = (C @ C.T).cpu().numpy()
        for i in range(sim.shape[0]): sim[i,i]=1.0
        second = np.partition(sim, -2, axis=1)[:,-2]
        inter = float(np.mean(second))
    return dict(intra_avg=intra_avg, inter_nearest_avg=inter, n_groups=len(centers))

def make_trainer(model, tokenizer, train_dataset, val_dataset,
                 use_gn, use_bias, out_dir,
                 num_train_epochs, lr, mlm_prob, stride):
    tokenid2gid, gid2tok = build_tokenid2gid_and_gid2tok(tokenizer)
    glyph_token_ids = set(t.item() for toks in gid2tok.values() for t in toks)
    collator = (GroupBiasedMLMCollator(tokenizer, mlm_prob, glyph_token_ids,
                                       tokenid2gid, gid2tok,
                                       same_group_rand=SAME_GROUP_RAND,
                                       glyph_bias=GLYPH_MASK_BIAS,
                                       stride=stride)
                if use_bias else
                DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob))

    args = TrainingArguments(
        output_dir=out_dir, overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=8,
        warmup_ratio=0.0, warmup_steps=0,
        learning_rate=lr, weight_decay=WEIGHT_DECAY,
        seed=SEED, logging_steps=50,
        report_to=["wandb"], bf16=True,
        eval_strategy="no",
        save_strategy="no"
    )

    if use_gn:
        return SmoothGNTrainer(model=model, args=args,
                               train_dataset=train_dataset,
                               eval_dataset=None, 
                               data_collator=collator, tokenizer=tokenizer,
                               tokenid2gid=tokenid2gid, gid2tok=gid2tok,
                               smooth_alpha_max=SMOOTH_ALPHA_MAX,
                               beta_warmup_ratio=BETA_WARMUP_RATIO,
                               smooth_glyph_only=True)
    else:
        return Trainer(model=model, args=args,
                       train_dataset=train_dataset,
                       eval_dataset=None, 
                       data_collator=collator, tokenizer=tokenizer)

def run_two_stage(scn, files, holdout_chars, seen_new, lr, mlm_prob):
    print(f"\n=== Running {scn['name']} ===")
    is_local = os.path.isdir(scn["base_ckpt"])
    tokenizer = AutoTokenizer.from_pretrained(scn["base_ckpt"], local_files_only=is_local)
    model = AutoModelForMaskedLM.from_pretrained(scn["base_ckpt"], local_files_only=is_local).to(device)
    model.config.attention_probs_dropout_prob = DROPOUT
    model.config.hidden_dropout_prob = DROPOUT
    model.config.dropout = getattr(model.config, "dropout", DROPOUT)
    tokenizer, model = register_unk_placeholders(tokenizer, model)

    added_seen, seen_set = expand_tokenizer_with_glyphs(tokenizer, model, seen_new, VARIANT_GROUPS)
    if scn["use_gn"] and added_seen > 0:
        gn_new_only_align(model, tokenizer, seen_set, alpha=GN_ALPHA_INIT)

    if scn["run_dapt"]:
        print("[stage-1] Running DAPT...")
        model = freeze_layers(model, num_freeze=6)
        raw_lines = [l.strip() for l in open(files["dapt_train_path"], encoding="utf-8") if l.strip()]
        train_lines, val_lines = train_test_split(raw_lines, test_size=0.1, random_state=SEED)
        train_path = files["dapt_train_path"].replace(".txt", "_train_split.txt")
        val_path   = files["dapt_train_path"].replace(".txt", "_val_split.txt")
        write_lines(train_path, train_lines); write_lines(val_path, val_lines)
        ds_train = build_hf_text_dataset(tokenizer, train_path)
        ds_val   = build_hf_text_dataset(tokenizer, val_path)
        dapt_trainer = make_trainer(
            model, tokenizer, ds_train, ds_val,
            use_gn=scn["use_gn"], use_bias=scn["use_bias"],
            out_dir= result_path("dapt", scn["name"]),
            num_train_epochs=DAPT_EPOCHS, lr=lr, mlm_prob=mlm_prob,
            stride=int(wandb.config["stride"])
        )
        dapt_trainer.train()
        dapt_dir = result_path("DAPT_base")
        os.makedirs(dapt_dir, exist_ok=True)
        model.save_pretrained(dapt_dir); tokenizer.save_pretrained(dapt_dir)

    if scn["run_tapt"]:
        print("[stage-2] Running TAPT...")
        num_epochs = int(wandb.config["epochs"])
        raw_lines = [l.strip() for l in open(files["tapt_train_path"], encoding="utf-8") if l.strip()]
        train_lines, val_lines = train_test_split(raw_lines, test_size=0.1, random_state=SEED)
        train_path = files["tapt_train_path"].replace(".txt", "_train_split.txt")
        val_path   = files["tapt_train_path"].replace(".txt", "_val_split.txt")
        write_lines(train_path, train_lines); write_lines(val_path, val_lines)
        ds_train = build_hf_text_dataset(tokenizer, train_path)
        ds_val   = build_hf_text_dataset(tokenizer, val_path)
        tapt_trainer = make_trainer(
            model, tokenizer, ds_train, ds_val,
            use_gn=scn["use_gn"], use_bias=scn["use_bias"],
            out_dir = result_path("tapt", scn["name"]),
            num_train_epochs=num_epochs, lr=lr, mlm_prob=mlm_prob,
            stride=int(wandb.config["stride"])
        )
        tapt_trainer.train()

    added_ho, ho_set = expand_tokenizer_with_glyphs(tokenizer, model, holdout_chars, VARIANT_GROUPS)
    if scn["use_gn"] and added_ho > 0:
        gn_new_only_align(model, tokenizer, ho_set, alpha=GN_ALPHA_INIT)

    if scn["name"] == "DAPT_only":
        final_dir = result_path("DAPT_base")
    else:
        final_dir = result_path("final", scn["name"])
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir); tokenizer.save_pretrained(final_dir)

    final_model = AutoModelForMaskedLM.from_pretrained(final_dir).to(device)
    final_tokenizer = AutoTokenizer.from_pretrained(final_dir)
    drive_dir = error_path(scn["name"])
    os.makedirs(drive_dir, exist_ok=True)
    log_path = os.path.join(drive_dir, "errors.csv")

    results = evaluate_model(
        final_model, final_tokenizer,
        files["tapt_test_holdout_path"],
        k=(1, 5, 10),
        stride=int(wandb.config["stride"]),
        log_path=log_path
    )
    rep = embedding_cluster_report(final_model, final_tokenizer)

    wandb.log({
        f"{scn['name']}/Exact@1": results.get("Exact@1", 0.0),
        f"{scn['name']}/Exact@5": results.get("Exact@5", 0.0),
        f"{scn['name']}/Exact@10": results.get("Exact@10", 0.0),
        f"{scn['name']}/Group@1": results.get("Group@1", 0.0),
        f"{scn['name']}/Group@5": results.get("Group@5", 0.0),
        f"{scn['name']}/Group@10": results.get("Group@10", 0.0),
        f"{scn['name']}/IntraCos": rep["intra_avg"],
        f"{scn['name']}/InterCos": rep["inter_nearest_avg"],
        f"{scn['name']}/NumGroups": rep["n_groups"],
    })

    return dict(
        name=scn["name"],
        exact1=results.get("Exact@1", 0.0),
        exact5=results.get("Exact@5", 0.0),
        exact10=results.get("Exact@10", 0.0),
        group1=results.get("Group@1", 0.0),
        group5=results.get("Group@5", 0.0),
        group10=results.get("Group@10", 0.0),
        intra=rep["intra_avg"],
        inter=rep["inter_nearest_avg"],
        ng=rep["n_groups"],
    )

DATA_PATH  = os.path.join(DATA_DIR, "dating.csv")
MAX_LEN    = 128
CLS_BATCH  = 16
CLS_EPOCHS = 5
CLS_LR     = 2e-5
os.makedirs(ERROR_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
expected = ["id","dynasty","period","text"]
df = df.rename(columns={df.columns[i]: expected[i] for i in range(min(4,len(df.columns)))})

df_dynasty = df.dropna(subset=["dynasty","text"]).copy()
df_multi   = df.dropna(subset=["dynasty","period","text"]).copy()

def build_label_map(values):
    labels = sorted(values.unique())
    m = {l:i for i,l in enumerate(labels)}
    return m, {i:l for l,i in m.items()}

dyn_label2id, dyn_id2label = build_label_map(df_dynasty["dynasty"])
per_label2id, per_id2label = build_label_map(df_multi["period"])

df_dynasty["label"] = df_dynasty["dynasty"].map(dyn_label2id)
df_multi["dynasty_label"] = df_multi["dynasty"].map(dyn_label2id)
df_multi["period_label"]  = df_multi["period"].map(per_label2id)

def make_dataset(df_, label_name):
    tr, te = train_test_split(df_, test_size=0.2, random_state=42, stratify=df_[label_name])
    return Dataset.from_pandas(tr), Dataset.from_pandas(te), te.reset_index(drop=True)

train_dyn, test_dyn, test_df_dyn = make_dataset(df_dynasty, "label")
train_multi, test_multi, test_df_multi = make_dataset(df_multi, "dynasty_label")

def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

def run_dynasty_cls(checkpoint, tokenizer, scenario):
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=len(dyn_label2id),
        id2label=dyn_id2label, label2id=dyn_label2id
    ).to(device)

    tr = train_dyn.map(lambda b: tokenize(b,tokenizer), batched=True)
    te = test_dyn.map(lambda b: tokenize(b,tokenizer), batched=True)

    args = TrainingArguments(
        output_dir=os.path.join(checkpoint,"dynasty_cls"),
        save_strategy="no", learning_rate=CLS_LR,
        per_device_train_batch_size=CLS_BATCH,
        per_device_eval_batch_size=CLS_BATCH,
        num_train_epochs=CLS_EPOCHS, report_to="none"
    )

    def compute_metrics(pred):
        logits, labels = pred
        preds = logits.argmax(-1)
        return {"eval_accuracy": accuracy_score(labels,preds),
                "eval_f1": f1_score(labels,preds,average="macro")}

    trainer = Trainer(model=model,args=args,
                      train_dataset=tr,eval_dataset=te,
                      tokenizer=tokenizer,compute_metrics=compute_metrics)

    trainer.train()
    metrics = trainer.evaluate()

    preds = trainer.predict(te).predictions.argmax(-1)
    errors = []
    for t, p, txt in zip(test_df_dyn["label"], preds, test_df_dyn["text"]):
        if t != p:
            errors.append({
                "true": dyn_id2label[int(t)],
                "pred": dyn_id2label[int(p)],
                "text": txt
            })
    if errors:
        out_path = os.path.join(ERROR_DIR, f"errors_{scenario}_dynasty.csv")
        pd.DataFrame(errors).to_csv(out_path, index=False, encoding="utf-8")
        print(f"[dynasty error] {len(errors)} entries → {out_path}")

    wandb.log({
        f"{scenario}/Acc_Dyn": metrics["eval_accuracy"],
        f"{scenario}/F1_Dyn":  metrics["eval_f1"],
    })

    return metrics

class HierarchicalModel(nn.Module):
    def __init__(self, checkpoint,num_dyn,num_per):
        super().__init__()
        self.bert = AutoModel.from_pretrained(checkpoint)
        hidden = self.bert.config.hidden_size
        self.classifier_dyn = nn.Linear(hidden,num_dyn)
        self.classifier_per = nn.Linear(hidden+num_dyn,num_per)
    def forward(self,input_ids=None,attention_mask=None,dynasty_label=None,period_label=None):
        out = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        pooled = out.last_hidden_state[:,0]
        logits_dyn = self.classifier_dyn(pooled)
        pooled_with_dyn = torch.cat([pooled,logits_dyn],dim=-1)
        logits_per = self.classifier_per(pooled_with_dyn)
        loss=None
        if dynasty_label is not None and period_label is not None:
            loss = nn.CrossEntropyLoss()(logits_dyn,dynasty_label)+nn.CrossEntropyLoss()(logits_per,period_label)
        return {"loss":loss,"logits_dyn":logits_dyn,"logits_per":logits_per}

class HierTrainer(Trainer):
    def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
        labels_dyn = inputs.get("dynasty_label"); labels_per = inputs.get("period_label")
        out = model(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],
                    dynasty_label=labels_dyn,period_label=labels_per)
        return (out["loss"], out) if return_outputs else out["loss"]

def run_hierarchical(checkpoint,tokenizer,scenario):
    model = HierarchicalModel(checkpoint,len(dyn_label2id),len(per_label2id)).to(device)
    tr = train_multi.map(lambda b: tokenize(b,tokenizer),batched=True)
    te = test_multi.map(lambda b: tokenize(b,tokenizer),batched=True)
    args = TrainingArguments(output_dir=os.path.join(checkpoint,"hier_cls"),
                             save_strategy="no",learning_rate=CLS_LR,
                             per_device_train_batch_size=CLS_BATCH,
                             per_device_eval_batch_size=CLS_BATCH,
                             num_train_epochs=CLS_EPOCHS,report_to="none")
    def compute_metrics(pred):
        (logits_dyn,logits_per), labels = pred
        labels_dyn, labels_per = labels
        preds_dyn = logits_dyn.argmax(-1); preds_per = logits_per.argmax(-1)
        return {
            "eval_accuracy_dyn": accuracy_score(labels_dyn,preds_dyn),
            "eval_f1_dyn": f1_score(labels_dyn,preds_dyn,average="macro"),
            "eval_accuracy_per": accuracy_score(labels_per,preds_per),
            "eval_f1_per": f1_score(labels_per,preds_per,average="macro")
        }
    trainer = HierTrainer(model=model,args=args,
                          train_dataset=tr,eval_dataset=te,
                          tokenizer=tokenizer,compute_metrics=compute_metrics)
    trainer.train(); metrics = trainer.evaluate()
    wandb.log({
        f"{scenario}/Acc_Hier_Dyn": metrics["eval_accuracy_dyn"],
        f"{scenario}/F1_Hier_Dyn":  metrics["eval_f1_dyn"],
        f"{scenario}/Acc_Hier_Per": metrics["eval_accuracy_per"],
        f"{scenario}/F1_Hier_Per":  metrics["eval_f1_per"],
    })
    return metrics

if __name__ == "__main__":

    wandb.init(
        project="BronzeGlyph",
        entity=None,   # 或者 "your_wandb_account"
        config={
            "lr": 2e-5,
            "epochs": 3,
            "mlm_prob": 0.15,
            "stride": 8,
            "holdout_ratio": 0.3
        }
    )

    lr, mlm_prob = float(wandb.config["lr"]), float(wandb.config["mlm_prob"])

    base_tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    all_new, holdout_chars, seen_new = pick_new_glyphs_to_holdout(base_tok, ratio=wandb.config["holdout_ratio"])

    raw_dapt = read_lines(DAPT_TRAIN_PATH) if os.path.exists(DAPT_TRAIN_PATH) else []
    raw_tapt_train = read_lines(TAPT_TRAIN_PATH)
    raw_tapt_test = read_lines(TAPT_TEST_PATH)

    dapt_filtered = filter_remove_holdout(raw_dapt, holdout_chars)
    tapt_filtered = filter_remove_holdout(raw_tapt_train, holdout_chars)
    tapt_test_holdout = build_test_with_holdout(raw_tapt_test, holdout_chars, min_needed=300)

    dapt_train_path = os.path.join(RESULTS_DIR, "dapt_train_holdout.txt")
    tapt_train_path = os.path.join(RESULTS_DIR, "tapt_train_holdout.txt")
    tapt_test_path = os.path.join(RESULTS_DIR, "tapt_test_holdout.txt")

    write_lines(dapt_train_path, dapt_filtered)
    write_lines(tapt_train_path, tapt_filtered)
    write_lines(tapt_test_path, tapt_test_holdout)

    files = dict(
        dapt_train_path=dapt_train_path,
        tapt_train_path=tapt_train_path,
        tapt_test_holdout_path=tapt_test_path
    )

    scenarios = [
        dict(name="DAPT_only", base_ckpt=MODEL_PATH, run_dapt=True, run_tapt=False, use_gn=False, use_bias=False),
        dict(name="Baseline", base_ckpt=MODEL_PATH, run_dapt=False, run_tapt=False, use_gn=False, use_bias=False),
        dict(name="TAPT_only", base_ckpt=MODEL_PATH, run_dapt=False, run_tapt=True, use_gn=False, use_bias=False),
        dict(name="TAPT_from_DAPT", base_ckpt=result_path("DAPT_base"), run_dapt=False, run_tapt=True, use_gn=False, use_bias=False),
        dict(name="TAPT_GN", base_ckpt=result_path("DAPT_base"), run_dapt=False, run_tapt=True, use_gn=True, use_bias=False),
        dict(name="TAPT_Bias", base_ckpt=result_path("DAPT_base"), run_dapt=False, run_tapt=True, use_gn=False, use_bias=True),
        dict(name="TAPT_GN_Bias", base_ckpt=result_path("DAPT_base"), run_dapt=False, run_tapt=True, use_gn=True, use_bias=True),
    ]

    rows = []
    for scn in scenarios:
        out = run_two_stage(scn, files, holdout_chars, seen_new, lr=lr, mlm_prob=mlm_prob)
        ckpt = result_path("DAPT_base") if scn["name"] == "DAPT_only" else result_path("final", scn["name"])
        if not os.path.isdir(ckpt):
            print(f"[warn] checkpoint not found: {ckpt}")
            continue
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        met_dyn = run_dynasty_cls(ckpt, tokenizer, scn["name"])
        met_hier = run_hierarchical(ckpt, tokenizer, scn["name"])
        rows.append([
            scn["name"],
            out["exact1"], out["exact5"], out["exact10"],
            out["group1"], out["group5"], out["group10"],
            met_dyn["eval_accuracy"], met_dyn["eval_f1"],
            met_hier["eval_accuracy_dyn"], met_hier["eval_f1_dyn"],
            met_hier["eval_accuracy_per"], met_hier["eval_f1_per"],
            out["intra"], out["inter"], out["ng"]
        ])

    df_out = pd.DataFrame(rows, columns=[
        "Scenario","Exact@1","Exact@5","Exact@10",
        "Group@1","Group@5","Group@10",
        "Acc_Dyn","F1_Dyn","Acc_Hier_Dyn","F1_Hier_Dyn",
        "Acc_Hier_Per","F1_Hier_Per",
        "IntraCos","NearestInterCos","#Groups"
    ])

    save_path = os.path.join(RESULTS_DIR, "final_results.csv")
    df_out.to_csv(save_path, index=False)
    print("\n==== ALL ====")
    print(df_out)
    print(f"\nSaved → {save_path}")