"""
=============================================================================
CENG 467 – NLU&G Take-Home Midterm  |  Question 3: Text Summarization
=============================================================================

Objective:
    Compare extractive (LexRank) and abstractive (BART) summarization methods
    on a subset of the CNN/DailyMail dataset, using ROUGE, BLEU, METEOR,
    and BERTScore metrics, with qualitative analysis.

Python version: 3.11.9
=============================================================================

Required installations (run these once before executing the script):
    pip install datasets transformers torch sentencepiece rouge-score nltk bert-score sumy # sumy provides LexRank

    Then inside Python:
        import nltk
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

import warnings
warnings.filterwarnings("ignore")          # keep the console tidy

import textwrap                            # pretty-printing long strings
import re                                  # light text cleaning

# ── Dataset ──────────────────────────────────────────────────────────────────
from datasets import load_dataset          # HuggingFace Datasets

# ── Extractive summarization (LexRank via sumy) ──────────────────────────────
from sumy.parsers.plaintext import PlaintextParser   # converts raw text → sumy doc
from sumy.nlp.tokenizers import Tokenizer            # sentence tokenizer wrapper
from sumy.summarizers.lex_rank import LexRankSummarizer  # graph-based extractive method
from sumy.nlp.stemmers import Stemmer                # optional stemming (EN)
from sumy.utils import get_stop_words                # EN stop-word list

# ── Abstractive summarization (BART via HuggingFace Transformers) ─────────────
from transformers import pipeline                    # high-level inference wrapper

# ── Evaluation metrics ────────────────────────────────────────────────────────
from rouge_score import rouge_scorer                 # ROUGE-1, ROUGE-2, ROUGE-L
import nltk                                          # BLEU + METEOR
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score           # contextual semantic similarity

# ── Misc ──────────────────────────────────────────────────────────────────────
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION  (tweak these constants to reproduce results)
# ─────────────────────────────────────────────────────────────────────────────

# Number of CNN/DailyMail articles to evaluate (keep small for fast runtimes)
NUM_SAMPLES = 10

# Number of sentences LexRank should return per article
LEXRANK_SENTENCES = 3

# BART summarization length bounds (tokens, not words)
BART_MIN_NEW_TOKENS = 56
BART_MAX_NEW_TOKENS = 142

# Number of qualitative examples to print in full
NUM_QUALITATIVE_EXAMPLES = 3

# BERTScore language (use 'en' for CNN/DailyMail)
BERTSCORE_LANG = "en"

# Random seed for reproducibility when shuffling the dataset
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DOWNLOAD NLTK DATA  (idempotent – safe to call multiple times)
# ─────────────────────────────────────────────────────────────────────────────

def download_nltk_resources() -> None:
    """Download NLTK corpora required by BLEU, METEOR, and sumy tokenizers."""
    resources = ["punkt", "punkt_tab", "wordnet", "omw-1.4"]
    for resource in resources:
        # quiet=True suppresses the "already downloaded" messages
        nltk.download(resource, quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATASET  –  CNN/DailyMail
# ─────────────────────────────────────────────────────────────────────────────

def load_cnn_dailymail(num_samples: int = NUM_SAMPLES) -> list[dict]:
    """
    Load a small subset of the CNN/DailyMail dataset (version 3.0.0).

    Dataset structure
    -----------------
    Each example is a Python dict with three string fields:
        • 'id'        – unique article identifier (SHA hash)
        • 'article'   – the full news article (source document)
        • 'highlights'– bullet-point human reference summary (target)

    The dataset was originally created by Nallapati et al. (2016) and
    contains ~300 k article–summary pairs from CNN and DailyMail websites.
    The 'highlights' field is the concatenation of the bullet points that
    accompany each online article, making it a fairly abstractive reference.

    We use the 'test' split (no labels needed) to avoid data-leakage concerns
    and select the first `num_samples` examples for tractability.

    Returns
    -------
    list[dict]  – list of raw HuggingFace dataset rows
    """
    print("=" * 70)
    print("STEP 1: Loading CNN/DailyMail dataset …")
    print("=" * 70)

    # streaming=False so we can index by position; 'test' split is standard
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

    # Shuffle with a fixed seed for reproducibility, then take the first N
    dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(num_samples))

    samples = list(dataset)          # convert Arrow table → plain Python list

    print(f"  ✓ Loaded {len(samples)} articles from the test split.\n")
    print("  Dataset structure (fields per example):")
    for key, val in samples[0].items():
        # Show field name + first 80 chars of the value so we understand it
        preview = str(val)[:80].replace("\n", " ")
        print(f"    '{key}': {preview!r} …")
    print()
    return samples

# ─────────────────────────────────────────────────────────────────────────────
# 4.  EXTRACTIVE SUMMARIZATION  –  LexRank
# ─────────────────────────────────────────────────────────────────────────────
#
# LexRank (Erkan & Radev, 2004) is a graph-based extractive method.
# Algorithm overview:
#   1. Represent each sentence as a TF-IDF vector.
#   2. Build a similarity graph: sentences = nodes,
#      cosine-similarity weights = edges.
#   3. Run the PageRank power-iteration on this graph.
#   4. Rank sentences by their stationary probability and pick the top-k.
#
# Strengths : No training needed; interpretable; computationally cheap.
# Weaknesses: Cannot paraphrase or merge information across sentences;
#             may select redundant sentences.
# ─────────────────────────────────────────────────────────────────────────────

def build_lexrank_summarizer() -> LexRankSummarizer:
    """
    Instantiate and return a pre-configured LexRank summarizer.

    sumy's LexRankSummarizer accepts an optional Stemmer to reduce words
    to their root form before computing TF-IDF, which can improve similarity
    estimates for morphologically rich text.
    """
    stemmer = Stemmer("english")
    summarizer = LexRankSummarizer(stemmer)
    # Inject English stop words so they are down-weighted in TF-IDF
    summarizer.stop_words = get_stop_words("english")
    return summarizer


def lexrank_summarize(text: str,
                      summarizer: LexRankSummarizer,
                      num_sentences: int = LEXRANK_SENTENCES) -> str:
    """
    Generate an extractive summary using LexRank.

    Parameters
    ----------
    text          : raw article string
    summarizer    : pre-built LexRankSummarizer instance
    num_sentences : how many sentences to extract

    Returns
    -------
    str – extracted sentences joined by a single space
    """
    # sumy requires a PlaintextParser that wraps the raw text with a Tokenizer
    parser = PlaintextParser.from_string(text, Tokenizer("english"))

    # summarizer() returns a list of sumy Sentence objects
    sentence_objects = summarizer(parser.document, num_sentences)

    # Convert sumy Sentence objects → plain Python strings, then join
    return " ".join(str(sentence) for sentence in sentence_objects)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  ABSTRACTIVE SUMMARIZATION  –  BART
# ─────────────────────────────────────────────────────────────────────────────
#
# BART (Lewis et al., 2020) is a denoising autoencoder pretrained by
# corrupting text and learning to reconstruct it.  The fine-tuned version
# 'facebook/bart-large-cnn' was trained specifically on CNN/DailyMail,
# making it a canonical baseline for this task.
#
# Architecture: Transformer encoder-decoder (similar to mT5 / T5).
# Generation strategy used here: beam search (default in the pipeline).
#
# Strengths : Fluent, coherent output; can merge & paraphrase information.
# Weaknesses: Higher GPU/CPU compute; prone to hallucination; slower.
# ─────────────────────────────────────────────────────────────────────────────

from transformers import BartForConditionalGeneration, BartTokenizer

def build_bart_pipeline():
    """
    Load BART-large-CNN model and tokenizer explicitly,
    bypassing the pipeline task registry.
    """
    print("STEP 2: Loading BART-large-CNN model (downloads on first run) …")

    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.eval()  # inference mode

    print("  ✓ Model loaded.\n")
    return (model, tokenizer)


import torch

def bart_summarize(text: str,
                   bart_pipeline,
                   min_new_tokens: int = BART_MIN_NEW_TOKENS,
                   max_new_tokens: int = BART_MAX_NEW_TOKENS) -> str:
    """
    Generate an abstractive summary using BART (no pipeline task registry needed).
    """
    model, tokenizer = bart_pipeline

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge(hypothesis: str, reference: str) -> dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    ROUGE (Lin, 2004) measures n-gram recall/precision between the generated
    summary (hypothesis) and the human reference.

    • ROUGE-1  : unigram overlap  → captures word-level coverage
    • ROUGE-2  : bigram overlap   → captures some fluency / phrase matching
    • ROUGE-L  : longest common subsequence → rewards sentence-level structure

    Returns
    -------
    dict with keys 'rouge1', 'rouge2', 'rougeL', each mapped to an F1 float.
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True      # stem before comparing to reduce surface variation
    )
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bleu(hypothesis: str, reference: str) -> float:
    """
    Compute sentence-level BLEU score with add-1 smoothing.

    BLEU (Papineni et al., 2002) counts n-gram precision (1-4 grams) and
    applies a brevity penalty for short outputs.  Originally designed for
    machine translation, it is included here for a translation-style
    precision signal.

    Smoothing (method1) avoids zero scores when higher-order n-gram counts
    are 0 – common for short summaries.

    Returns
    -------
    float in [0, 1]
    """
    ref_tokens  = reference.lower().split()
    hyp_tokens  = hypothesis.lower().split()
    smoother    = SmoothingFunction().method1
    score       = sentence_bleu([ref_tokens], hyp_tokens,
                                smoothing_function=smoother)
    return round(score, 4)


def compute_meteor(hypothesis: str, reference: str) -> float:
    """
    Compute METEOR score.

    METEOR (Banerjee & Lavie, 2005) aligns hypothesis and reference tokens
    using exact match, stem match, and WordNet synonymy, then computes an
    F-mean weighted toward recall.  It correlates better with human judgement
    than BLEU on short texts.

    Returns
    -------
    float in [0, 1]
    """
    ref_tokens  = reference.lower().split()
    hyp_tokens  = hypothesis.lower().split()
    score       = meteor_score([ref_tokens], hyp_tokens)
    return round(score, 4)


def compute_bertscore(hypotheses: list[str],
                      references: list[str],
                      lang: str = BERTSCORE_LANG) -> dict[str, float]:
    """
    Compute BERTScore (Zhang et al., 2020) for a batch of hypothesis-reference pairs.

    BERTScore uses contextual embeddings from a pre-trained BERT model to
    compute token-level cosine similarities and derives precision, recall,
    and F1.  It is robust to paraphrase and captures semantic similarity
    beyond surface n-gram overlap.

    Parameters
    ----------
    hypotheses : list of generated summary strings
    references : list of human reference strings (same order)
    lang       : language code used to select the BERT model

    Returns
    -------
    dict with 'precision', 'recall', 'f1' – each is the batch-mean float.
    """
    P, R, F1 = bert_score(
        hypotheses,
        references,
        lang=lang,
        verbose=False,        # suppress per-example progress bars
    )
    return {
        "precision": round(P.mean().item(), 4),
        "recall":    round(R.mean().item(), 4),
        "f1":        round(F1.mean().item(), 4),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FULL EVALUATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(samples: list[dict],
                 lexrank_summarizer: LexRankSummarizer,
                 bart_pipe) -> dict:
    """
    Run both summarizers over all samples and compute all metrics.

    Returns
    -------
    dict with keys:
        'lexrank_summaries'  : list[str]
        'bart_summaries'     : list[str]
        'references'         : list[str]
        'articles'           : list[str]
        'per_sample_metrics' : list[dict]   (one dict per article)
        'aggregate'          : dict          (mean over all samples)
    """
    print("=" * 70)
    print("STEP 3: Generating summaries and computing metrics …")
    print("=" * 70)

    lexrank_summaries = []
    bart_summaries    = []
    references        = []
    articles          = []
    per_sample        = []

    for idx, sample in enumerate(samples):
        article   = sample["article"]
        reference = sample["highlights"]       # human-written reference

        print(f"  Article {idx + 1}/{len(samples)} …", end=" ", flush=True)

        # ── Generate summaries ────────────────────────────────────────────────
        lr_summary   = lexrank_summarize(article, lexrank_summarizer)
        bart_summary = bart_summarize(article, bart_pipe)

        # ── Per-sample metrics ────────────────────────────────────────────────
        lr_rouge  = compute_rouge(lr_summary, reference)
        lr_bleu   = compute_bleu(lr_summary, reference)
        lr_meteor = compute_meteor(lr_summary, reference)

        ba_rouge  = compute_rouge(bart_summary, reference)
        ba_bleu   = compute_bleu(bart_summary, reference)
        ba_meteor = compute_meteor(bart_summary, reference)

        per_sample.append({
            "lexrank": {**lr_rouge, "bleu": lr_bleu, "meteor": lr_meteor},
            "bart":    {**ba_rouge, "bleu": ba_bleu, "meteor": ba_meteor},
        })

        lexrank_summaries.append(lr_summary)
        bart_summaries.append(bart_summary)
        references.append(reference)
        articles.append(article)

        print("done")

    # ── BERTScore is computed in one batch call (faster) ──────────────────────
    print("\n  Computing BERTScore (batch) for LexRank …", end=" ", flush=True)
    lr_bert = compute_bertscore(lexrank_summaries, references)
    print("done")

    print("  Computing BERTScore (batch) for BART …", end=" ", flush=True)
    ba_bert = compute_bertscore(bart_summaries, references)
    print("done\n")

    # ── Aggregate (mean) metrics across all samples ───────────────────────────
    metric_keys = ["rouge1", "rouge2", "rougeL", "bleu", "meteor"]

    def mean_metric(method_key: str, metric: str) -> float:
        return round(
            np.mean([row[method_key][metric] for row in per_sample]), 4
        )

    aggregate = {
        "lexrank": {
            **{k: mean_metric("lexrank", k) for k in metric_keys},
            "bertscore_precision": lr_bert["precision"],
            "bertscore_recall":    lr_bert["recall"],
            "bertscore_f1":        lr_bert["f1"],
        },
        "bart": {
            **{k: mean_metric("bart", k) for k in metric_keys},
            "bertscore_precision": ba_bert["precision"],
            "bertscore_recall":    ba_bert["recall"],
            "bertscore_f1":        ba_bert["f1"],
        },
    }

    return {
        "lexrank_summaries": lexrank_summaries,
        "bart_summaries":    bart_summaries,
        "references":        references,
        "articles":          articles,
        "per_sample_metrics": per_sample,
        "aggregate":          aggregate,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 8.  PRINTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def wrap(text: str, width: int = 90, indent: str = "    ") -> str:
    """Wrap a long string to `width` columns with a uniform left indent."""
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def print_aggregate_table(aggregate: dict) -> None:
    """Pretty-print the aggregate metric table for both methods."""
    print("=" * 70)
    print("AGGREGATE METRICS  (mean over all samples)")
    print("=" * 70)

    # Column headers
    header = f"{'Metric':<28} {'LexRank':>10} {'BART':>10}"
    print(header)
    print("-" * 50)

    metric_labels = {
        "rouge1":              "ROUGE-1 F1",
        "rouge2":              "ROUGE-2 F1",
        "rougeL":              "ROUGE-L F1",
        "bleu":                "BLEU",
        "meteor":              "METEOR",
        "bertscore_precision": "BERTScore Precision",
        "bertscore_recall":    "BERTScore Recall",
        "bertscore_f1":        "BERTScore F1",
    }

    for key, label in metric_labels.items():
        lr_val = aggregate["lexrank"].get(key, 0.0)
        ba_val = aggregate["bart"].get(key, 0.0)
        print(f"  {label:<26} {lr_val:>10.4f} {ba_val:>10.4f}")

    print()


def print_qualitative_examples(results: dict,
                                n: int = NUM_QUALITATIVE_EXAMPLES) -> None:
    """
    Print qualitative comparison for the first `n` samples.

    For each example we show:
        - A truncated version of the source article (first 400 chars)
        - The human reference summary
        - The LexRank extractive summary
        - The BART abstractive summary
        - Per-sample metrics for both methods
        - A brief textual analysis of the differences
    """
    print("=" * 70)
    print(f"QUALITATIVE ANALYSIS  (first {n} examples)")
    print("=" * 70)

    for i in range(n):
        article   = results["articles"][i]
        reference = results["references"][i]
        lr_sum    = results["lexrank_summaries"][i]
        ba_sum    = results["bart_summaries"][i]
        metrics   = results["per_sample_metrics"][i]

        print(f"\n{'─' * 70}")
        print(f"  EXAMPLE {i + 1}")
        print(f"{'─' * 70}")

        # ── Source article snippet ─────────────────────────────────────────────
        article_snippet = article[:400].replace("\n", " ") + " …"
        print("\n  SOURCE ARTICLE (first 400 chars):")
        print(wrap(article_snippet))

        # ── Human reference ────────────────────────────────────────────────────
        print("\n  HUMAN REFERENCE SUMMARY:")
        print(wrap(reference.replace("\n", " ")))

        # ── LexRank ───────────────────────────────────────────────────────────
        print("\n  LEXRANK (Extractive) SUMMARY:")
        print(wrap(lr_sum))

        # ── BART ──────────────────────────────────────────────────────────────
        print("\n  BART (Abstractive) SUMMARY:")
        print(wrap(ba_sum))

        # ── Per-sample metrics table ───────────────────────────────────────────
        print("\n  METRICS (this example):")
        mhdr = f"    {'Metric':<20} {'LexRank':>10} {'BART':>10}"
        print(mhdr)
        print("    " + "-" * 42)
        for mk in ["rouge1", "rouge2", "rougeL", "bleu", "meteor"]:
            lr_v = metrics["lexrank"][mk]
            ba_v = metrics["bart"][mk]
            print(f"    {mk:<20} {lr_v:>10.4f} {ba_v:>10.4f}")

        # ── Textual analysis ──────────────────────────────────────────────────
        print("\n  ANALYSIS:")
        _print_analysis(lr_sum, ba_sum, metrics)

    print()


def _print_analysis(lr_sum: str, ba_sum: str, metrics: dict) -> None:
    """
    Print a heuristic qualitative analysis for one example.

    The analysis compares:
        1. Fluency      – sentence-level coherence (heuristic: avg word length
                          and absence of abrupt cuts)
        2. Factual      – whether the summary stays close to the source
                          (extractive by definition does; abstractive may not)
        3. Coverage     – ROUGE-L as a proxy for information coverage
    """
    lr_r   = metrics["lexrank"]
    ba_r   = metrics["bart"]

    # Heuristic fluency proxy: average sentence length in words
    def avg_sent_len(text: str) -> float:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if not sentences:
            return 0.0
        return sum(len(s.split()) for s in sentences) / len(sentences)

    lr_asl = avg_sent_len(lr_sum)
    ba_asl = avg_sent_len(ba_sum)

    analysis_lines = []

    # Fluency
    if ba_asl > lr_asl * 1.1:
        analysis_lines.append(
            "FLUENCY: BART produces slightly smoother, more naturally-flowing "
            "sentences (avg %.1f words/sent vs LexRank %.1f), because it can "
            "restructure and connect ideas." % (ba_asl, lr_asl)
        )
    else:
        analysis_lines.append(
            "FLUENCY: Both methods produce similar sentence lengths "
            "(LexRank %.1f vs BART %.1f words/sent)." % (lr_asl, ba_asl)
        )

    # Factual consistency
    analysis_lines.append(
        "FACTUAL CONSISTENCY: LexRank is guaranteed faithful to the source "
        "because it copies verbatim sentences. BART may introduce minor "
        "hallucinations or paraphrase details, especially for numeric facts."
    )

    # Coverage (ROUGE-L)
    if lr_r["rougeL"] > ba_r["rougeL"] + 0.02:
        analysis_lines.append(
            "INFORMATION COVERAGE: LexRank achieves higher ROUGE-L (%.4f vs "
            "%.4f), suggesting its extracted sentences contain more reference "
            "n-grams—likely because both use the same source vocabulary." %
            (lr_r["rougeL"], ba_r["rougeL"])
        )
    elif ba_r["rougeL"] > lr_r["rougeL"] + 0.02:
        analysis_lines.append(
            "INFORMATION COVERAGE: BART achieves higher ROUGE-L (%.4f vs "
            "%.4f), indicating its paraphrasing aligns well with the human "
            "reference style." % (ba_r["rougeL"], lr_r["rougeL"])
        )
    else:
        analysis_lines.append(
            "INFORMATION COVERAGE: ROUGE-L scores are comparable "
            "(LexRank %.4f, BART %.4f)." % (lr_r["rougeL"], ba_r["rougeL"])
        )

    for line in analysis_lines:
        print(wrap(line, indent="    • "))

# ─────────────────────────────────────────────────────────────────────────────
# 9.  DISCUSSION / CONCLUSION
# ─────────────────────────────────────────────────────────────────────────────

def print_discussion() -> None:
    """
    Print a structured comparison of extractive vs abstractive summarization,
    covering computational cost, readability, and faithfulness.
    """
    print("=" * 70)
    print("DISCUSSION: EXTRACTIVE vs ABSTRACTIVE SUMMARIZATION")
    print("=" * 70)

    discussion = """
┌─────────────────────┬──────────────────────────┬──────────────────────────┐
│ Dimension           │ LexRank (Extractive)      │ BART (Abstractive)        │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Computational cost  │ Very low – O(n²) graph    │ High – 400M param model  │
│                     │ build; no GPU required    │ needs GPU for speed      │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Training data       │ None (unsupervised)       │ ~300k article–summary    │
│                     │                          │ pairs (CNN/DailyMail)    │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Readability         │ Moderate – verbatim       │ High – fluent, cohesive  │
│                     │ sentences may be long     │ paraphrasing             │
│                     │ or context-dependent      │                          │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Faithfulness        │ Perfect (no hallucination)│ Prone to minor errors,   │
│                     │ – copies source sentences │ especially numeric facts │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Information fusion  │ Cannot merge information  │ Can synthesize facts     │
│                     │ across sentences          │ across paragraphs        │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Domain transfer     │ Excellent – works on any  │ Limited – may degrade    │
│                     │ domain without retraining │ on out-of-domain text    │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Typical ROUGE-1     │ ~0.30 – 0.40              │ ~0.40 – 0.45 (fine-tuned)│
└─────────────────────┴──────────────────────────┴──────────────────────────┘

Key trade-offs
──────────────

1. Computational Cost
   LexRank runs in milliseconds on a CPU and requires no model weights,
   making it ideal for real-time or resource-constrained applications.
   BART-large requires several seconds per article even on a GPU and
   ~1.6 GB of VRAM, making it unsuitable for latency-sensitive pipelines
   without hardware investment.

2. Readability & Fluency
   BART consistently produces more readable output because the decoder
   generates natural language rather than copy-pasting sentences from
   potentially different paragraphs. LexRank summaries sometimes feel
   disjointed when extracted sentences rely on local coreferences
   (e.g., "He said …" where the referent is not extracted).

3. Faithfulness to the Source (Factual Consistency)
   Extractive methods are inherently faithful: they cannot introduce
   information not present in the original. Abstractive models can and
   do hallucinate, particularly with numbers, named entities, and
   temporal expressions. For fact-critical applications (medical, legal,
   financial), faithfulness is paramount and extractive methods may be
   preferred despite lower fluency.

4. Information Coverage
   BART can synthesize information from multiple paragraphs into a single
   coherent sentence, improving conceptual density. LexRank cannot fuse
   information across sentences; it selects the most "central" sentences,
   which may leave important but less connected details uncaptured.

5. Practical Recommendation
   • Use LexRank (or similar) when: speed, interpretability, or
     zero-shot domain transfer is critical.
   • Use BART (or similar) when: fluency, readability, and information
     synthesis matter more than strict factual faithfulness, and
     compute resources are available.
   • Hybrid approaches (extract then abstract, or guided decoding) are
     an active research direction that attempts to retain the faithfulness
     of extraction while improving fluency.
"""
    print(discussion)

# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate the full pipeline:
        1. Download NLTK data
        2. Load CNN/DailyMail subset
        3. Build LexRank summarizer
        4. Load BART pipeline
        5. Run evaluation over all samples
        6. Print aggregate metrics table
        7. Print qualitative examples with analysis
        8. Print discussion / conclusion
    """

    # ── Step 0: NLTK downloads ────────────────────────────────────────────────
    download_nltk_resources()

    # ── Step 1: Dataset ───────────────────────────────────────────────────────
    samples = load_cnn_dailymail(num_samples=NUM_SAMPLES)

    # ── Step 2: Build models ──────────────────────────────────────────────────
    lexrank_summarizer = build_lexrank_summarizer()
    bart_pipe          = build_bart_pipeline()

    # ── Step 3: Run evaluation ────────────────────────────────────────────────
    results = evaluate_all(samples, lexrank_summarizer, bart_pipe)

    # ── Step 4: Print aggregate metrics table ─────────────────────────────────
    print_aggregate_table(results["aggregate"])

    # ── Step 5: Print qualitative examples ───────────────────────────────────
    print_qualitative_examples(results, n=NUM_QUALITATIVE_EXAMPLES)

    # ── Step 6: Discussion ────────────────────────────────────────────────────
    print_discussion()

    print("=" * 70)
    print("Script completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()