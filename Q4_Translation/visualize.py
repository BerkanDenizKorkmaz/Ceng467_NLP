"""
visualize.py – Generate all figures for the analysis report.
Saves PNGs to /home/claude/seq2seq_analysis/figures/
"""

import os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

from core import (
    MULTI30K_PAIRS, tokenize, build_vocabs, describe_preprocessing,
    generate_translations, evaluate_corpus, simulate_training_curves, bleu_score,
)
from models import (
    Seq2SeqAttention, TransformerModel, architecture_params,
    positional_encoding,
)

# Get the exact folder where visualize.py is currently sitting
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create a "figures" folder inside that directory
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C_S2S  = "#E07B54"   # warm orange  → Seq2Seq
C_TRF  = "#4C9BE8"   # cool blue    → Transformer
C_REF  = "#3DAA6B"   # green        → reference
C_DARK = "#1A1A2E"
C_GRID = "#E8E8F0"
CMAP_ATT = LinearSegmentedColormap.from_list("att", ["#F7F9FF", "#1A3A8F"])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#FAFBFF",
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.6,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Data
# ─────────────────────────────────────────────────────────────────────────────
src_vocab, tgt_vocab = build_vocabs(MULTI30K_PAIRS)
prep = describe_preprocessing(MULTI30K_PAIRS, src_vocab, tgt_vocab)
translations = generate_translations(MULTI30K_PAIRS)
scores = evaluate_corpus(translations)
curves = simulate_training_curves(epochs=30)
arch   = architecture_params(len(src_vocab), len(tgt_vocab))

print("Preprocessing summary:")
for k, v in prep.items():
    print(f"  {k}: {v}")
print()
print("Corpus metrics:")
for model in ("seq2seq", "transformer"):
    m = scores[model]
    print(f"  {model:12s}  BLEU={m['BLEU']:.2f}  METEOR={m['METEOR']:.2f}"
          f"  ChrF={m['ChrF']:.2f}  BERTScore={m['BERTScore_F1']:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1: Dataset Overview  (3 panels)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Multi30k Dataset — Preprocessing Overview", fontsize=15, fontweight="bold",
             color=C_DARK, y=1.01)

# (a) Sentence length distribution
src_lens = [len(tokenize(e)) for e, _ in MULTI30K_PAIRS]
tgt_lens = [len(tokenize(d)) for _, d in MULTI30K_PAIRS]
bins = np.arange(2, 25, 2)
axes[0].hist(src_lens, bins=bins, alpha=0.8, color=C_S2S, label="EN (source)", edgecolor="white", linewidth=0.7)
axes[0].hist(tgt_lens, bins=bins, alpha=0.7, color=C_TRF, label="DE (target)", edgecolor="white", linewidth=0.7)
axes[0].axvline(np.mean(src_lens), color=C_S2S, linestyle="--", linewidth=1.5)
axes[0].axvline(np.mean(tgt_lens), color=C_TRF, linestyle="--", linewidth=1.5)
axes[0].set_xlabel("Sentence Length (tokens)")
axes[0].set_ylabel("Count")
axes[0].set_title("(a) Token Length Distribution")
axes[0].legend()

# (b) Vocabulary sizes
vocab_data = {
    "EN Vocab": len(src_vocab),
    "DE Vocab": len(tgt_vocab),
    "Special\nTokens": 4,
}
bar_colors = [C_S2S, C_TRF, C_REF]
bars = axes[1].bar(vocab_data.keys(), vocab_data.values(), color=bar_colors,
                    edgecolor="white", linewidth=1.2, width=0.5)
for bar, val in zip(bars, vocab_data.values()):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[1].set_title("(b) Vocabulary Sizes")
axes[1].set_ylabel("Unique Tokens")
axes[1].set_ylim(0, max(vocab_data.values()) * 1.2)

# (c) Preprocessing pipeline
axes[2].axis("off")
steps = [
    "① Lowercase all text",
    "② Separate punctuation",
    "③ Whitespace tokenisation",
    "④ Build separate EN / DE vocabs",
    "⑤ Add <pad>, <sos>, <eos>, <unk>",
    "⑥ Filter: src & tgt len ≤ 40",
    "⑦ Integer-encode sentences",
    "⑧ Pad to batch max-length",
]
for i, step in enumerate(steps):
    color = C_S2S if i % 2 == 0 else C_TRF
    axes[2].text(0.05, 0.95 - i * 0.115, step, transform=axes[2].transAxes,
                 fontsize=10, color=color, fontweight="bold" if i < 2 else "normal",
                 va="top")
axes[2].set_title("(c) Preprocessing Pipeline", pad=6)
axes[2].add_patch(mpatches.FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
    boxstyle="round,pad=0.02", facecolor="#F0F4FF", edgecolor="#C0C8E8",
    transform=axes[2].transAxes, zorder=0))

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_dataset_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2: Architecture Diagrams (ASCII-style SVG rendered as matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Model Architectures", fontsize=15, fontweight="bold", color=C_DARK)

def draw_box(ax, x, y, w, h, label, color="#4C9BE8", fontsize=9, text_color="white"):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.015", facecolor=color, edgecolor="white",
        linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold", zorder=4)

def arrow(ax, x1, y1, x2, y2, color="#555"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5), zorder=2)

# ── Seq2Seq panel ──
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("Seq2Seq + Bahdanau Attention\n(Encoder–Decoder with GRU)", fontsize=12,
             color=C_S2S, pad=8)

# Source tokens
for i, tok in enumerate(["A", "man", "is", "walking", "."]):
    draw_box(ax, 1+i*1.6, 1.0, 1.3, 0.55, tok, color="#8B5E83", fontsize=8)

# Encoder
draw_box(ax, 3.8, 2.2, 5.8, 0.70, "Bi-GRU Encoder  (hidden=512 × 2)", C_S2S, fontsize=9)

# Encoder hidden states
for i in range(5):
    draw_box(ax, 1+i*1.6, 3.3, 1.2, 0.50, f"h₍{i+1}₎", "#D44D3E", fontsize=8)
    arrow(ax, 1+i*1.6, 2.55, 1+i*1.6, 3.05, C_S2S)

# Attention
draw_box(ax, 3.8, 4.5, 5.8, 0.70, "Bahdanau Attention  score = vᵀ·tanh(W₁h + W₂s)", "#E8A838", fontsize=8.5)
for i in range(5):
    arrow(ax, 1+i*1.6, 3.55, 3.8, 4.15, "#E8A838")

# Context vector
draw_box(ax, 3.8, 5.55, 3.0, 0.55, "Context Vector  c = Σ αᵢhᵢ", "#5A9E6F", fontsize=9)
arrow(ax, 3.8, 4.85, 3.8, 5.27, "#5A9E6F")

# Decoder GRU
draw_box(ax, 3.8, 6.7, 5.8, 0.70, "GRU Decoder  (hidden=512)", C_TRF, fontsize=9)
arrow(ax, 3.8, 5.83, 3.8, 6.35, C_TRF)

# Output projection + softmax
draw_box(ax, 3.8, 7.85, 4.0, 0.55, "Linear → Softmax → Target token", "#333", fontsize=8.5)
arrow(ax, 3.8, 7.05, 3.8, 7.57, "#333")

ax.text(5, 9.2, "Complexity: O(n) sequential steps\nLong-range: via attention + recurrence",
        ha="center", fontsize=9, style="italic", color="#555",
        bbox=dict(boxstyle="round", facecolor="#FFF5E6", edgecolor=C_S2S, alpha=0.8))

# ── Transformer panel ──
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("Transformer (Vaswani et al. 2017)\n(Self-Attention + FFN, N=6 layers)", fontsize=12,
             color=C_TRF, pad=8)

# Input embeddings + PE
draw_box(ax, 5, 1.0, 7.5, 0.60, "Token Embeddings + Positional Encoding  (d_model=512)", "#7B68EE", fontsize=8.5)

# Encoder stack
for layer in range(3):
    y = 2.4 + layer * 2.0
    draw_box(ax, 5, y,      7.5, 0.55, f"Multi-Head Self-Attention  (8 heads)", C_TRF, fontsize=8.5)
    draw_box(ax, 5, y+0.75, 7.5, 0.55, "Add & Norm  →  FFN  →  Add & Norm", "#888", fontsize=8.5, text_color="white")
    arrow(ax, 5, y-0.28, 5, y+0.20, C_TRF)
    arrow(ax, 5, y+0.47, 5, y+0.47, "#888")
    if layer > 0:
        arrow(ax, 5, y-0.82+0.05, 5, y-0.28, "#999")
    ax.text(9.0, y+0.10, f"Layer {layer+1}", fontsize=8, color="#999", va="center")

# Output
arrow(ax, 5, 8.53, 5, 8.85, "#333")
draw_box(ax, 5, 9.10, 5.0, 0.55, "Linear + Softmax (target vocab)", "#333", fontsize=8.5)

arrow(ax, 5, 1.30, 5, 2.11, "#7B68EE")

ax.text(5, 0.28, "Complexity: O(n²) attention per layer\nLong-range: direct all-to-all connections",
        ha="center", fontsize=9, style="italic", color="#555",
        bbox=dict(boxstyle="round", facecolor="#E6F0FF", edgecolor=C_TRF, alpha=0.8))

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_architectures.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3: Training Curves
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Simulated Training Dynamics (EN→DE, Multi30k)", fontsize=14,
             fontweight="bold", color=C_DARK)

ep = curves["epochs"]
for ax, split in zip(axes, ["train_loss", "val_loss"]):
    label = "Training Loss" if "train" in split else "Validation Loss"
    ax.plot(ep, curves["seq2seq"][split], color=C_S2S, lw=2.2, label="Seq2Seq + Attn")
    ax.plot(ep, curves["transformer"][split], color=C_TRF, lw=2.2, label="Transformer")
    ax.fill_between(ep, curves["seq2seq"][split], curves["transformer"][split],
                    alpha=0.12, color="#9999CC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(label)
    ax.legend()
    ax.set_xlim(1, 30)

# Annotate convergence
axes[0].annotate("Transformer\nconverges faster", xy=(10, curves["transformer"]["train_loss"][9]),
                 xytext=(14, 1.8), fontsize=8.5, color=C_TRF,
                 arrowprops=dict(arrowstyle="->", color=C_TRF))
axes[1].annotate("Seq2Seq plateaus\nat higher loss", xy=(25, curves["seq2seq"]["val_loss"][24]),
                 xytext=(15, 2.2), fontsize=8.5, color=C_S2S,
                 arrowprops=dict(arrowstyle="->", color=C_S2S))

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4: Metric Comparison (bar + box + radar)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5.5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
fig.suptitle("Translation Quality Metrics — Seq2Seq vs Transformer", fontsize=14,
             fontweight="bold", color=C_DARK)

metric_names = ["BLEU", "METEOR", "ChrF", "BERTScore F1"]
s2s_vals  = [scores["seq2seq"][k]  for k in ["BLEU", "METEOR", "ChrF", "BERTScore_F1"]]
trf_vals  = [scores["transformer"][k] for k in ["BLEU", "METEOR", "ChrF", "BERTScore_F1"]]

# (a) Grouped bar chart
ax1 = fig.add_subplot(gs[0])
x = np.arange(len(metric_names))
w = 0.35
b1 = ax1.bar(x - w/2, s2s_vals,  w, label="Seq2Seq + Attn", color=C_S2S, edgecolor="white", linewidth=1.2)
b2 = ax1.bar(x + w/2, trf_vals,  w, label="Transformer",    color=C_TRF, edgecolor="white", linewidth=1.2)
for bar in list(b1) + list(b2):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
             f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels(metric_names, rotation=15, ha="right")
ax1.set_ylabel("Score")
ax1.set_title("(a) Mean Corpus Scores")
ax1.legend(fontsize=8)
ax1.set_ylim(0, max(trf_vals) * 1.25)

# (b) Box plots per model (BLEU distribution)
ax2 = fig.add_subplot(gs[1])
s2s_bleu = scores["seq2seq"]["raw"]["bleu_scores"]
trf_bleu = scores["transformer"]["raw"]["bleu_scores"]
bp = ax2.boxplot([s2s_bleu, trf_bleu], patch_artist=True, notch=False,
                  medianprops=dict(color="white", linewidth=2.5),
                  whiskerprops=dict(linewidth=1.5),
                  capprops=dict(linewidth=1.5),
                  flierprops=dict(marker="o", markersize=4, alpha=0.6))
for patch, color in zip(bp["boxes"], [C_S2S, C_TRF]):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
ax2.set_xticklabels(["Seq2Seq\n+ Attn", "Transformer"])
ax2.set_ylabel("Sentence BLEU")
ax2.set_title("(b) Sentence-BLEU Distribution")

# (c) Radar / spider chart
ax3 = fig.add_subplot(gs[2], projection="polar")
categories = ["BLEU", "METEOR", "ChrF", "BERTScore"]
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Normalise to 0-100
norms = [100, 100, 100, 100]
s_vals = [v/n*100 for v, n in zip(s2s_vals, norms)] + [s2s_vals[0]/norms[0]*100]
t_vals = [v/n*100 for v, n in zip(trf_vals,  norms)] + [trf_vals[0]/norms[0]*100]

ax3.plot(angles, s_vals, color=C_S2S, linewidth=2, label="Seq2Seq")
ax3.fill(angles, s_vals, color=C_S2S, alpha=0.20)
ax3.plot(angles, t_vals, color=C_TRF, linewidth=2, label="Transformer")
ax3.fill(angles, t_vals, color=C_TRF, alpha=0.20)
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(categories, fontsize=9)
ax3.set_ylim(0, 100)
ax3.set_title("(c) Metric Profile\n(Radar Chart)", pad=20, fontsize=11)
ax3.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
ax3.grid(color=C_GRID, linewidth=0.8)

plt.savefig(f"{FIG_DIR}/fig4_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig4")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5: Attention Heatmaps (Seq2Seq Bahdanau)
# ─────────────────────────────────────────────────────────────────────────────
# Use sentence index 0: "A man in a blue shirt is standing on a ladder."
pair = MULTI30K_PAIRS[0]
src_tokens = tokenize(pair[0])
tgt_tokens = tokenize(pair[1])

src_ids = src_vocab.encode(src_tokens)
tgt_ids = tgt_vocab.encode(tgt_tokens)

seq2seq_model = Seq2SeqAttention(len(src_vocab), len(tgt_vocab), seed=42)
att_map = seq2seq_model.get_attention_map(src_ids, tgt_ids)  # (tgt_len, src_len)

# Trim to actual token lengths (skip SOS/EOS)
att_sub = att_map[:len(tgt_tokens), :len(src_tokens)]

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
fig.suptitle("Attention Visualisations", fontsize=14, fontweight="bold", color=C_DARK)

# Bahdanau (Seq2Seq)
im = axes[0].imshow(att_sub, aspect="auto", cmap=CMAP_ATT, vmin=0, vmax=att_sub.max())
axes[0].set_xticks(range(len(src_tokens))); axes[0].set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=8)
axes[0].set_yticks(range(len(tgt_tokens))); axes[0].set_yticklabels(tgt_tokens, fontsize=8)
axes[0].set_xlabel("Source (EN)"); axes[0].set_ylabel("Target (DE)")
axes[0].set_title("(a) Bahdanau Cross-Attention\nSeq2Seq Encoder→Decoder", color=C_S2S)
plt.colorbar(im, ax=axes[0], fraction=0.035, label="Attention weight")

# Transformer self-attention (encoder layer 0, head 0)
trf_model = TransformerModel(len(src_vocab), len(tgt_vocab), seed=1)
sa_map = trf_model.get_self_attention_map(src_ids, layer=0, head=0)
# Show only source tokens
sa_sub = sa_map[:len(src_tokens)+2, :len(src_tokens)+2]
xtlabels = ["<sos>"] + src_tokens + ["<eos>"]
im2 = axes[1].imshow(sa_sub, aspect="auto", cmap=CMAP_ATT)
axes[1].set_xticks(range(len(xtlabels))); axes[1].set_xticklabels(xtlabels, rotation=45, ha="right", fontsize=8)
axes[1].set_yticks(range(len(xtlabels))); axes[1].set_yticklabels(xtlabels, fontsize=8)
axes[1].set_xlabel("Source token (attending to)"); axes[1].set_ylabel("Source token (query)")
axes[1].set_title("(b) Transformer Self-Attention\nEncoder Layer 1, Head 1", color=C_TRF)
plt.colorbar(im2, ax=axes[1], fraction=0.035, label="Attention weight")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_attention.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig5")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6: Qualitative Examples (3 sentences)
# ─────────────────────────────────────────────────────────────────────────────
examples = [translations[0], translations[10], translations[45]]  # short, medium, long
fig, axes = plt.subplots(3, 1, figsize=(16, 9))
fig.suptitle("Qualitative Translation Examples", fontsize=14, fontweight="bold", color=C_DARK, y=1.01)

for ax, ex in zip(axes, examples):
    ax.axis("off")
    ref_b  = bleu_score(ex["ref_tokens"],         [ex["ref_tokens"]])["bleu"]
    s2s_b  = bleu_score(ex["seq2seq_tokens"],     [ex["ref_tokens"]])["bleu"]
    trf_b  = bleu_score(ex["transformer_tokens"], [ex["ref_tokens"]])["bleu"]

    info = (
        f"SRC:   {ex['source']}\n"
        f"REF:   {ex['reference']}\n"
        f"S2S:   {ex['seq2seq_text']}   [BLEU={s2s_b:.1f}]\n"
        f"TRF:   {ex['transformer_text']}   [BLEU={trf_b:.1f}]"
    )

    y_positions = [0.82, 0.58, 0.34, 0.10]
    line_colors = ["#1A1A2E", C_REF, C_S2S, C_TRF]
    labels = ["SRC", "REF", "S2S", "TRF"]
    lines = info.split("\n")

    ax.add_patch(mpatches.FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.02", facecolor="#F8F9FF",
        edgecolor="#D0D8F0", transform=ax.transAxes, linewidth=1.5))

    for y, color, label, line in zip(y_positions, line_colors, labels, lines):
        # Strip the "XXX:   " prefix and display with colored tag
        content = line.split(":", 1)[1].strip() if ":" in line else line
        ax.text(0.01, y, f"[{label}]", transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", color=color, va="top")
        ax.text(0.07, y, content, transform=ax.transAxes,
                fontsize=9, color="#222", va="top", wrap=True,
                bbox=dict(facecolor="white", alpha=0.0))

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig6_qualitative.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig6")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7: Positional Encoding Visualisation
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Positional Encoding — Transformer", fontsize=14, fontweight="bold", color=C_DARK)

PE = positional_encoding(50, 128)
im = axes[0].imshow(PE.T[:64], aspect="auto", cmap="RdYlBu_r")
axes[0].set_xlabel("Position"); axes[0].set_ylabel("Encoding Dimension")
axes[0].set_title("(a) Sinusoidal PE Matrix (d=128, first 64 dims)")
plt.colorbar(im, ax=axes[0], fraction=0.04)

# Show a few dimension waves
colors = [C_S2S, C_TRF, C_REF, "#9B59B6"]
for i, dim in enumerate([0, 2, 8, 32]):
    axes[1].plot(PE[:, dim], label=f"dim {dim}", color=colors[i], lw=1.8)
axes[1].set_xlabel("Position"); axes[1].set_ylabel("PE value")
axes[1].set_title("(b) Individual Dimension Waves")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig7_positional_encoding.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig7")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 8: Per-sentence metric deltas (Transformer − Seq2Seq)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Per-Sentence Metric Advantage  (Transformer − Seq2Seq)", fontsize=14,
             fontweight="bold", color=C_DARK)

raw_s = scores["seq2seq"]["raw"]
raw_t = scores["transformer"]["raw"]
metric_pairs = [
    ("BLEU",        "bleu_scores",   axes[0, 0]),
    ("METEOR",      "meteor_scores", axes[0, 1]),
    ("ChrF",        "chrf_scores",   axes[1, 0]),
    ("BERTScore F1","bert_f1",       axes[1, 1]),
]

for name, key, ax in metric_pairs:
    delta = np.array(raw_t[key]) - np.array(raw_s[key])
    colors_bar = [C_TRF if d >= 0 else C_S2S for d in delta]
    ax.bar(range(len(delta)), delta, color=colors_bar, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#333", linewidth=1)
    ax.axhline(delta.mean(), color="#FF6B6B", linewidth=1.5, linestyle="--",
               label=f"Mean Δ = {delta.mean():.2f}")
    ax.set_xlabel("Sentence index"); ax.set_ylabel(f"Δ {name}")
    ax.set_title(f"{name}  (blue=Transformer wins, orange=Seq2Seq wins)")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig8_delta_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig8")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL RESULTS SUMMARY")
print("="*65)
print(f"{'Metric':<18} {'Seq2Seq+Attn':>14} {'Transformer':>14} {'Δ (TRF-S2S)':>12}")
print("-"*65)
for k, label in [("BLEU","BLEU"),("METEOR","METEOR"),("ChrF","ChrF"),("BERTScore_F1","BERTScore F1")]:
    s = scores["seq2seq"][k]
    t = scores["transformer"][k]
    print(f"{label:<18} {s:>14.2f} {t:>14.2f} {t-s:>+12.2f}")
print("="*65)
print(f"\nSeq2Seq parameters : {arch['seq2seq_params']:,}")
print(f"Transformer params : {arch['transformer_params']:,}")
print(f"\nAll figures saved to {FIG_DIR}")