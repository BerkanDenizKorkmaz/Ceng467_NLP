"""
Question 2 – Named Entity Recognition
Dataset : CoNLL-2003
Models  : (1) BiLSTM-CRF   (2) BERT Token Classifier (bert-base-cased)
Eval    : Precision, Recall, F1 (entity-level via seqeval)
"""

# ─────────────────────────────────────────────
# 0. DEPENDENCIES
# ─────────────────────────────────────────────
# pip install datasets transformers torch seqeval scikit-learn matplotlib seaborn

import os, re, random, warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED)
import torch; torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────
# 1. LOAD CoNLL-2003 & DESCRIBE ANNOTATION
# ─────────────────────────────────────────────
from datasets import load_dataset

print("\n" + "=" * 60)
print("STEP 1 – CoNLL-2003 Dataset")
print("=" * 60)

conll = load_dataset("lhoestq/conll2003")
train_ds, val_ds, test_ds = conll["train"], conll["validation"], conll["test"]

# Label list (BIO scheme)
LABEL_LIST = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']# ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG',
#  'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
NUM_LABELS = len(LABEL_LIST)
label2id   = {l: i for i, l in enumerate(LABEL_LIST)}
id2label   = {i: l for l, i in label2id.items()}

print(f"\nSplit sizes  – Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
print(f"Entity types – {set(l.split('-')[1] for l in LABEL_LIST if '-' in l)}")
print(f"BIO labels   – {LABEL_LIST}")

# Annotation statistics
from collections import Counter
entity_counts = Counter()
for ex in train_ds:
    for tag in ex["ner_tags"]:
        lbl = id2label[tag]
        if lbl.startswith("B-"):
            entity_counts[lbl[2:]] += 1

print("\nEntity counts in training set:")
for ent, cnt in sorted(entity_counts.items(), key=lambda x: -x[1]):
    print(f"  {ent:<6}: {cnt}")

# ── BIO tagging example ──
ex = train_ds[0]
print("\nBIO alignment example (sentence 0):")
print(f"  {'TOKEN':<15} {'BIO TAG'}")
print(f"  {'-'*25}")
for tok, tag in zip(ex["tokens"], ex["ner_tags"]):
    print(f"  {tok:<15} {id2label[tag]}")

# ─────────────────────────────────────────────
# 2. PREPROCESSING UTILITIES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 – Preprocessing & BIO Alignment")
print("=" * 60)

# ── 2a. Vocabulary for BiLSTM-CRF ──
from collections import Counter as Ctr

all_words = [tok for ex in train_ds for tok in ex["tokens"]]
word_freq  = Ctr(all_words)
VOCAB_SIZE = 20_000
vocab      = ["<PAD>", "<UNK>"] + [w for w, _ in word_freq.most_common(VOCAB_SIZE - 2)]
word2idx   = {w: i for i, w in enumerate(vocab)}

# character vocabulary
all_chars  = set(c for w in vocab for c in w)
char2idx   = {"<PAD>": 0, "<UNK>": 1}
char2idx.update({c: i+2 for i, c in enumerate(sorted(all_chars))})
CHAR_VOCAB  = len(char2idx)
MAX_WORD_LEN = 20

def encode_sentence_bilstm(tokens, tags, max_len=128):
    """Return word ids, char ids (padded), and tag ids (truncated/padded)."""
    tokens = tokens[:max_len]
    tags   = tags[:max_len]
    word_ids = [word2idx.get(t, 1) for t in tokens]
    char_ids = []
    for tok in tokens:
        cids = [char2idx.get(c, 1) for c in tok[:MAX_WORD_LEN]]
        cids += [0] * (MAX_WORD_LEN - len(cids))
        char_ids.append(cids)
    return word_ids, char_ids, list(tags)

print(f"Vocabulary size: {len(vocab)}")
print(f"Char vocab size: {CHAR_VOCAB}")

# ─────────────────────────────────────────────
# 3. MODEL 1 – BiLSTM-CRF
# ─────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

print("\n" + "=" * 60)
print("STEP 3 – BiLSTM-CRF")
print("=" * 60)

# ── CRF Layer ──
class CRF(nn.Module):
    """Linear-chain CRF."""
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        # Transition: transitions[i,j] = score of going from tag j to tag i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        # No transition TO start, no transition FROM end
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions   = nn.Parameter(torch.randn(num_tags))

    def _forward_alg(self, emissions, mask):
        batch, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]   # (B, T)
        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)            # (B, T, 1)
            broadcast_emit  = emissions[:, t].unsqueeze(1)  # (B, 1, T)
            next_score = broadcast_score + self.transitions + broadcast_emit
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)                # (B,)

    def _viterbi_decode(self, emissions, mask):
        batch, seq_len, num_tags = emissions.shape
        viterbi = self.start_transitions + emissions[:, 0]
        backpointers = []
        for t in range(1, seq_len):
            broadcast_viterbi = viterbi.unsqueeze(2)
            broadcast_emit    = emissions[:, t].unsqueeze(1)
            trans_score = broadcast_viterbi + self.transitions
            best_scores, best_tags = trans_score.max(dim=1)
            viterbi = best_scores + broadcast_emit.squeeze(1)
            backpointers.append(best_tags)
        viterbi += self.end_transitions
        best_last_tags = viterbi.argmax(dim=1)
        best_path = [best_last_tags]
        for bp in reversed(backpointers):
            best_last_tags = bp[torch.arange(batch), best_last_tags]
            best_path.append(best_last_tags)
        best_path.reverse()
        return torch.stack(best_path, dim=1)

    def neg_log_likelihood(self, emissions, tags, mask):
        forward_score = self._forward_alg(emissions, mask)
        # Gold score
        batch, seq_len = tags.shape
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(
            1, tags[:, 0:1]).squeeze(1)
        for t in range(1, seq_len):
            trans = self.transitions[tags[:, t], tags[:, t-1]]
            emit  = emissions[:, t].gather(1, tags[:, t:t+1]).squeeze(1)
            score += (trans + emit) * mask[:, t].float()
        last_tag_idx = mask.long().sum(1) - 1
        last_tags    = tags.gather(1, last_tag_idx.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        return (forward_score - score).mean()

    def forward(self, emissions, mask):
        return self._viterbi_decode(emissions, mask)


# ── BiLSTM-CRF Model ──
EMBED_DIM  = 100
CHAR_DIM   = 30
CHAR_HIDDEN= 50
HIDDEN_DIM = 256

class CharBiLSTMCRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embed = nn.Embedding(len(vocab), EMBED_DIM, padding_idx=0)
        self.char_embed = nn.Embedding(CHAR_VOCAB, CHAR_DIM, padding_idx=0)
        self.char_lstm  = nn.LSTM(CHAR_DIM, CHAR_HIDDEN, batch_first=True, bidirectional=True)
        total_input = EMBED_DIM + CHAR_HIDDEN * 2
        self.bilstm = nn.LSTM(total_input, HIDDEN_DIM // 2, num_layers=2,
                              bidirectional=True, batch_first=True, dropout=0.3)
        self.dropout   = nn.Dropout(0.3)
        self.hidden2tag = nn.Linear(HIDDEN_DIM, NUM_LABELS)
        self.crf        = CRF(NUM_LABELS)

    def _get_char_features(self, char_ids):
        # char_ids: (B, SeqLen, MaxWordLen)
        B, S, W = char_ids.shape
        char_ids_flat = char_ids.view(B * S, W)
        char_emb = self.char_embed(char_ids_flat)         # (B*S, W, C)
        _, (h, _) = self.char_lstm(char_emb)
        h = torch.cat([h[0], h[1]], dim=1)                # (B*S, 2*CHAR_HIDDEN)
        return h.view(B, S, -1)

    def forward(self, word_ids, char_ids, lengths, tags=None, mask=None):
        word_emb  = self.dropout(self.word_embed(word_ids))
        char_feat = self.dropout(self._get_char_features(char_ids))
        combined  = torch.cat([word_emb, char_feat], dim=-1)
        packed    = pack_padded_sequence(combined, lengths.cpu(),
                                         batch_first=True, enforce_sorted=False)
        out, _    = self.bilstm(packed)
        out, _    = pad_packed_sequence(out, batch_first=True)
        emissions = self.hidden2tag(self.dropout(out))
        if tags is not None:
            loss = self.crf.neg_log_likelihood(emissions, tags, mask)
            return loss
        return self.crf(emissions, mask)


# ── Dataset ──
class NERDataset(Dataset):
    def __init__(self, split, max_len=128):
        self.data = []
        for ex in split:
            wids, cids, tids = encode_sentence_bilstm(
                ex["tokens"], ex["ner_tags"], max_len)
            self.data.append((wids, cids, tids))

    def __len__(self):  return len(self.data)
    def __getitem__(self, i): return self.data[i]


def ner_collate(batch):
    word_ids_list, char_ids_list, tag_ids_list = zip(*batch)
    lengths = torch.tensor([len(w) for w in word_ids_list])
    max_len = lengths.max().item()

    word_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    char_pad = torch.zeros(len(batch), max_len, MAX_WORD_LEN, dtype=torch.long)
    tag_pad  = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask     = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, (w, c, t) in enumerate(zip(word_ids_list, char_ids_list, tag_ids_list)):
        L = len(w)
        word_pad[i, :L] = torch.tensor(w)
        char_pad[i, :L] = torch.tensor(c)
        tag_pad[i,  :L] = torch.tensor(t)
        mask[i, :L]     = True
    return word_pad, char_pad, tag_pad, mask, lengths


BATCH_SIZE  = 32
EPOCHS_CRF  = 10
LR_CRF      = 1e-3

print("Building datasets…")
train_ner = NERDataset(train_ds)
val_ner   = NERDataset(val_ds)
test_ner  = NERDataset(test_ds)

train_dl_ner = DataLoader(train_ner, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=ner_collate)
val_dl_ner   = DataLoader(val_ner,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=ner_collate)
test_dl_ner  = DataLoader(test_ner,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=ner_collate)

bilstm_crf   = CharBiLSTMCRF().to(DEVICE)
opt_crf      = torch.optim.Adam(bilstm_crf.parameters(), lr=LR_CRF, weight_decay=1e-4)
scheduler_crf= torch.optim.lr_scheduler.ReduceLROnPlateau(opt_crf, patience=2, factor=0.5)

def train_crf_epoch(model, loader):
    model.train(); total_loss = 0; n = 0
    for wids, cids, tids, mask, lengths in loader:
        wids, cids, tids, mask = (x.to(DEVICE) for x in (wids, cids, tids, mask))
        opt_crf.zero_grad()
        loss = model(wids, cids, lengths, tags=tids, mask=mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt_crf.step()
        total_loss += loss.item(); n += 1
    return total_loss / n

def predict_crf(model, loader):
    model.eval(); all_preds = []; all_true = []
    with torch.no_grad():
        for wids, cids, tids, mask, lengths in loader:
            wids, cids, mask = wids.to(DEVICE), cids.to(DEVICE), mask.to(DEVICE)
            preds = model(wids, cids, lengths, mask=mask)  # (B, S)
            for i, length in enumerate(lengths):
                L = length.item()
                all_preds.append([id2label[p] for p in preds[i, :L].cpu().tolist()])
                all_true.append( [id2label[t] for t in tids[i, :L].tolist()])
    return all_preds, all_true

from seqeval.metrics import classification_report as seq_report, f1_score as seq_f1

print("\nTraining BiLSTM-CRF…")
best_val_f1_crf = 0
for epoch in range(1, EPOCHS_CRF + 1):
    loss = train_crf_epoch(bilstm_crf, train_dl_ner)
    val_preds, val_true = predict_crf(bilstm_crf, val_dl_ner)
    val_f1 = seq_f1(val_true, val_preds)
    scheduler_crf.step(1 - val_f1)
    print(f"  Epoch {epoch:2d}/{EPOCHS_CRF}  loss={loss:.4f}  val_F1={val_f1:.4f}")
    if val_f1 > best_val_f1_crf:
        best_val_f1_crf = val_f1
        torch.save(bilstm_crf.state_dict(), "best_bilstm_crf.pt")

bilstm_crf.load_state_dict(torch.load("best_bilstm_crf.pt"))
crf_preds, crf_true = predict_crf(bilstm_crf, test_dl_ner)
print("\n--- BiLSTM-CRF Test Report ---")
print(seq_report(crf_true, crf_preds, digits=4))

# ─────────────────────────────────────────────
# 4. MODEL 2 – BERT Token Classifier
# ─────────────────────────────────────────────
from transformers import (AutoTokenizer,
                          AutoModelForTokenClassification,
                          get_linear_schedule_with_warmup)

print("\n" + "=" * 60)
print("STEP 4 – BERT Token Classification")
print("=" * 60)

BERT_MODEL   = "bert-base-cased"          # cased is better for NER
BATCH_BERT   = 16
EPOCHS_BERT  = 4
LR_BERT      = 3e-5
MAX_LEN_BERT = 128

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

def tokenize_and_align(examples, tokenizer, max_len=MAX_LEN_BERT):
    """
    Tokenize with WordPiece and propagate BIO labels.
    Sub-word tokens after the first get label -100 (ignored in loss).
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    all_labels = []
    for i, label_seq in enumerate(examples["ner_tags"]):
        word_ids    = tokenized.word_ids(batch_index=i)
        prev_word   = None
        label_ids   = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev_word:
                label_ids.append(label_seq[wid])
            else:
                label_ids.append(-100)   # sub-word continuation → ignore
            prev_word = wid
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


class BERTNERDataset(Dataset):
    def __init__(self, split):
        self.items = []
        batch_size = 256
        for start in range(0, len(split), batch_size):
            batch = split[start:start+batch_size]
            enc   = tokenize_and_align(batch, bert_tokenizer)
            for i in range(len(batch["tokens"])):
                self.items.append({
                    "input_ids":      enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i],
                    "token_type_ids": enc.get("token_type_ids",
                                              [torch.zeros_like(enc["input_ids"][i])])[i],
                    "labels":         torch.tensor(enc["labels"][i], dtype=torch.long),
                })

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


print("Tokenising for BERT…")
bert_train_ds_ner = BERTNERDataset(train_ds)
bert_val_ds_ner   = BERTNERDataset(val_ds)
bert_test_ds_ner  = BERTNERDataset(test_ds)

bert_train_dl = DataLoader(bert_train_ds_ner, batch_size=BATCH_BERT, shuffle=True)
bert_val_dl   = DataLoader(bert_val_ds_ner,   batch_size=BATCH_BERT, shuffle=False)
bert_test_dl  = DataLoader(bert_test_ds_ner,  batch_size=BATCH_BERT, shuffle=False)

bert_ner = AutoModelForTokenClassification.from_pretrained(
    BERT_MODEL,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
).to(DEVICE)

opt_bert     = torch.optim.AdamW(bert_ner.parameters(), lr=LR_BERT)
total_steps  = len(bert_train_dl) * EPOCHS_BERT
sched_bert   = get_linear_schedule_with_warmup(
    opt_bert, num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps)

def bert_train_epoch_ner(model, loader):
    model.train(); total_loss = 0; n = 0
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbl  = batch["labels"].to(DEVICE)
        opt_bert.zero_grad()
        out  = model(input_ids=ids, attention_mask=mask, labels=lbl)
        out.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_bert.step(); sched_bert.step()
        total_loss += out.loss.item(); n += 1
    return total_loss / n

def bert_predict_ner(model, loader):
    model.eval(); all_preds = []; all_true = []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbl  = batch["labels"]
            logits = model(input_ids=ids, attention_mask=mask).logits
            pred_ids = logits.argmax(-1).cpu()
            for i in range(len(ids)):
                p_seq = []; t_seq = []
                for j in range(lbl.shape[1]):
                    if lbl[i, j] == -100: continue
                    p_seq.append(id2label[pred_ids[i, j].item()])
                    t_seq.append(id2label[lbl[i, j].item()])
                all_preds.append(p_seq)
                all_true.append(t_seq)
    return all_preds, all_true

print("\nTraining BERT NER…")
best_val_f1_bert = 0
for epoch in range(1, EPOCHS_BERT + 1):
    loss = bert_train_epoch_ner(bert_ner, bert_train_dl)
    val_preds_bert, val_true_bert = bert_predict_ner(bert_ner, bert_val_dl)
    val_f1_bert = seq_f1(val_true_bert, val_preds_bert)
    print(f"  Epoch {epoch}/{EPOCHS_BERT}  loss={loss:.4f}  val_F1={val_f1_bert:.4f}")
    if val_f1_bert > best_val_f1_bert:
        best_val_f1_bert = val_f1_bert
        bert_ner.save_pretrained("best_bert_ner")

bert_ner = AutoModelForTokenClassification.from_pretrained("best_bert_ner").to(DEVICE)
bert_preds_test, bert_true_test = bert_predict_ner(bert_ner, bert_test_dl)
print("\n--- BERT NER Test Report ---")
print(seq_report(bert_true_test, bert_preds_test, digits=4))

# ─────────────────────────────────────────────
# 5. EVALUATION & ERROR ANALYSIS
# ─────────────────────────────────────────────
from seqeval.metrics import precision_score, recall_score, f1_score as sf1

print("\n" + "=" * 60)
print("STEP 5 – Evaluation Summary")
print("=" * 60)

def metrics(true, pred, name):
    p = precision_score(true, pred)
    r = recall_score(true, pred)
    f = sf1(true, pred)
    print(f"  {name:<20} Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")
    return p, r, f

print(f"\n{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>8}")
print("-" * 52)
crf_p, crf_r, crf_f    = metrics(crf_true,        crf_preds,       "BiLSTM-CRF")
bert_p, bert_r, bert_f = metrics(bert_true_test,   bert_preds_test, "BERT")

# ── Per-entity-type bar chart ──
from seqeval.metrics import classification_report as seq_cr_str
import io, re as _re

def parse_seqeval_report(report_str):
    """Parse entity-level rows from seqeval report string."""
    rows = {}
    for line in report_str.strip().split("\n"):
        line = line.strip()
        parts = line.split()
        if len(parts) == 5 and parts[0] not in ("micro", "macro", "weighted"):
            try:
                rows[parts[0]] = {
                    "precision": float(parts[1]),
                    "recall":    float(parts[2]),
                    "f1":        float(parts[3]),
                }
            except ValueError:
                pass
    return rows

crf_report_str  = seq_cr_str(crf_true, crf_preds, digits=4, output_dict=False)
bert_report_str = seq_cr_str(bert_true_test, bert_preds_test, digits=4, output_dict=False)

crf_entity  = parse_seqeval_report(crf_report_str)
bert_entity = parse_seqeval_report(bert_report_str)

entity_types = sorted(set(crf_entity) | set(bert_entity))
crf_f1s  = [crf_entity.get(e, {}).get("f1", 0)  for e in entity_types]
bert_f1s = [bert_entity.get(e, {}).get("f1", 0) for e in entity_types]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(entity_types)); w = 0.35
ax.bar(x - w/2, crf_f1s,  w, label="BiLSTM-CRF", color="#4C72B0")
ax.bar(x + w/2, bert_f1s, w, label="BERT",        color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels(entity_types)
ax.set_ylim(0, 1.05); ax.set_ylabel("F1 Score")
ax.set_title("Per-Entity-Type F1: BiLSTM-CRF vs BERT")
ax.legend(); plt.tight_layout()
plt.savefig("ner_entity_f1.png", dpi=150, bbox_inches="tight")
print("\n[Saved] ner_entity_f1.png")

# ── Overall comparison bar chart ──
fig2, ax2 = plt.subplots(figsize=(7, 4))
categories = ["Precision", "Recall", "F1"]
crf_vals   = [crf_p, crf_r, crf_f]
bert_vals  = [bert_p, bert_r, bert_f]
x2 = np.arange(3); w2 = 0.3
ax2.bar(x2 - w2/2, crf_vals,  w2, label="BiLSTM-CRF", color="#4C72B0")
ax2.bar(x2 + w2/2, bert_vals, w2, label="BERT",        color="#DD8452")
ax2.set_xticks(x2); ax2.set_xticklabels(categories)
ax2.set_ylim(0.7, 1.0); ax2.set_ylabel("Score")
ax2.set_title("NER Model Comparison")
ax2.legend(); plt.tight_layout()
plt.savefig("ner_model_comparison.png", dpi=150, bbox_inches="tight")
print("[Saved] ner_model_comparison.png")

# ─────────────────────────────────────────────
# 6. ERROR ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 – Error Analysis")
print("=" * 60)

def categorise_errors(true_seqs, pred_seqs, model_name):
    """
    Classify NER errors into:
      - Boundary errors  : entity detected but span is wrong
      - Type/confusion   : span correct but wrong entity type
      - Missed entities  : gold entity not predicted at all
      - Spurious         : predicted entity not in gold
    """
    boundary = type_conf = missed = spurious = 0

    def extract_spans(seq):
        spans = {}
        start = None; cur_type = None
        for i, tag in enumerate(seq):
            if tag.startswith("B-"):
                if start is not None:
                    spans[(start, i-1)] = cur_type
                start = i; cur_type = tag[2:]
            elif tag.startswith("I-"):
                pass  # continue span
            else:
                if start is not None:
                    spans[(start, i-1)] = cur_type
                    start = None; cur_type = None
        if start is not None:
            spans[(start, len(seq)-1)] = cur_type
        return spans

    for true_seq, pred_seq in zip(true_seqs, pred_seqs):
        gold_spans = extract_spans(true_seq)
        pred_spans = extract_spans(pred_seq)
        gold_set   = set(gold_spans.keys())
        pred_set   = set(pred_spans.keys())

        for span in gold_set & pred_set:           # exact span match
            if gold_spans[span] != pred_spans[span]:
                type_conf += 1                      # type confusion
        for span in gold_set - pred_set:            # gold not predicted
            # check partial overlap
            overlaps = any(
                not (span[1] < ps[0] or ps[1] < span[0])
                for ps in pred_set)
            if overlaps:
                boundary += 1
            else:
                missed += 1
        for span in pred_set - gold_set:            # spurious prediction
            overlaps = any(
                not (span[1] < gs[0] or gs[1] < span[0])
                for gs in gold_set)
            if not overlaps:
                spurious += 1

    total = boundary + type_conf + missed + spurious
    print(f"\n[{model_name}] Error breakdown (total={total}):")
    print(f"  Boundary errors    : {boundary:4d} ({100*boundary/max(total,1):.1f}%)")
    print(f"  Type confusion     : {type_conf:4d} ({100*type_conf/max(total,1):.1f}%)")
    print(f"  Missed entities    : {missed:4d} ({100*missed/max(total,1):.1f}%)")
    print(f"  Spurious entities  : {spurious:4d} ({100*spurious/max(total,1):.1f}%)")
    return boundary, type_conf, missed, spurious

crf_errs  = categorise_errors(crf_true,       crf_preds,       "BiLSTM-CRF")
bert_errs = categorise_errors(bert_true_test, bert_preds_test, "BERT")

# ── Error breakdown stacked bar ──
error_cats  = ["Boundary", "Type Confusion", "Missed", "Spurious"]
fig3, ax3 = plt.subplots(figsize=(8, 4))
x3 = np.arange(4); colors = ["#e07b54", "#5c85d6", "#67b79e", "#c065c0"]
crf_totals  = sum(crf_errs)
bert_totals = sum(bert_errs)
crf_norm    = [v / max(crf_totals,1)  for v in crf_errs]
bert_norm   = [v / max(bert_totals,1) for v in bert_errs]
ax3.bar(x3 - 0.2, crf_norm,  0.4, label="BiLSTM-CRF", color="#4C72B0")
ax3.bar(x3 + 0.2, bert_norm, 0.4, label="BERT",        color="#DD8452")
ax3.set_xticks(x3); ax3.set_xticklabels(error_cats, rotation=10)
ax3.set_ylabel("Proportion of Errors"); ax3.set_title("Error Type Distribution")
ax3.legend(); plt.tight_layout()
plt.savefig("ner_error_breakdown.png", dpi=150, bbox_inches="tight")
print("\n[Saved] ner_error_breakdown.png")

# ── Print 5 example errors per model ──
def show_ner_errors(true_seqs, pred_seqs, raw_split, model_name, n=5):
    print(f"\n=== {model_name}: sample errors ===")
    shown = 0
    for i, (ts, ps) in enumerate(zip(true_seqs, pred_seqs)):
        if ts != ps and shown < n:
            tokens = raw_split[i]["tokens"][:len(ts)]
            print(f"  Sentence : {' '.join(tokens)}")
            mismatches = [(tok, t, p) for tok, t, p in zip(tokens, ts, ps) if t != p]
            for tok, t, p in mismatches:
                print(f"    TOKEN={tok:<15} TRUE={t:<10} PRED={p}")
            print()
            shown += 1

show_ner_errors(crf_true,       crf_preds,       list(test_ds), "BiLSTM-CRF")
show_ner_errors(bert_true_test, bert_preds_test, list(test_ds), "BERT")

# ─────────────────────────────────────────────
# 7. DISCUSSION
# ─────────────────────────────────────────────
DISCUSSION = """
╔══════════════════════════════════════════════════════════════╗
║  DISCUSSION – Contextual Embeddings & Model Comparison       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  BiLSTM-CRF                                                  ║
║  • Character embeddings handle OOV tokens and morphology.    ║
║  • CRF layer enforces valid BIO transitions (e.g., no I-PER  ║
║    after O without B-PER).                                   ║
║  • Struggles with long-range context and polysemous names.   ║
║  • Typical CoNLL-2003 F1: ~85-87%.                           ║
║                                                              ║
║  BERT (bert-base-cased)                                      ║
║  • WordPiece sub-word tokenisation + deep attention.         ║
║  • Contextual embeddings capture "Paris" differently in      ║
║    "Paris Hilton" vs "Paris, France".                        ║
║  • Fine-tuning only 3-4 epochs achieves ~90-92% F1.          ║
║  • Cased model is critical for NER (capitalisation matters). ║
║                                                              ║
║  COMMON ERROR PATTERNS                                       ║
║  • Boundary errors: model detects entity but over/under-     ║
║    segments ("New York City" vs "New York").                 ║
║  • Type confusion: ORG vs LOC for ambiguous names            ║
║    ("Washington" can be a person, city, or org).             ║
║  • Rare/unseen entities: both models struggle with           ║
║    low-frequency proper nouns.                               ║
║  • BERT reduces boundary & type confusion significantly      ║
║    due to full-sequence attention over context.              ║
║                                                              ║
║  ROLE OF CONTEXTUAL EMBEDDINGS                               ║
║  • Static embeddings (GloVe/word2vec) assign one vector per  ║
║    word regardless of context.                               ║
║  • BERT generates different token representations based on   ║
║    surrounding words, resolving ambiguity dynamically.       ║
║  • This is the key reason BERT outperforms BiLSTM-CRF on     ║
║    entities that are context-dependent.                      ║
╚══════════════════════════════════════════════════════════════╝
"""
print(DISCUSSION)
print("✓ NER pipeline complete.")
print("  Saved: ner_entity_f1.png  ner_model_comparison.png  ner_error_breakdown.png")