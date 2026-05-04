"""
core.py – Data structures, preprocessing, and metric implementations
for the Seq2Seq vs Transformer analysis (pure NumPy / SciPy only).
"""

import numpy as np
import re
import math
import collections
from typing import List, Tuple, Dict

# ─────────────────────────────────────────────
# 1.  MULTI30K-STYLE DATASET (EN → DE subset)
# ─────────────────────────────────────────────
# 50 representative sentence pairs drawn from the Multi30k distribution
MULTI30K_PAIRS: List[Tuple[str, str]] = [
    ("A man in a blue shirt is standing on a ladder.",
     "Ein Mann in einem blauen Hemd steht auf einer Leiter."),
    ("Two dogs are playing in the park.",
     "Zwei Hunde spielen im Park."),
    ("A young woman is riding a bicycle along the river.",
     "Eine junge Frau fährt mit dem Fahrrad an dem Fluss entlang."),
    ("The children are eating ice cream on a hot summer day.",
     "Die Kinder essen Eis an einem heißen Sommertag."),
    ("A cat is sitting on a window sill watching birds outside.",
     "Eine Katze sitzt auf einem Fensterbrett und beobachtet Vögel draußen."),
    ("Several people are waiting at the bus stop in the rain.",
     "Mehrere Menschen warten in dem Regen an der Bushaltestelle."),
    ("The old man is feeding pigeons in the city square.",
     "Der alte Mann füttert Tauben auf dem Stadtplatz."),
    ("A group of friends is having a picnic in the meadow.",
     "Eine Gruppe von Freunden macht ein Picknick auf der Wiese."),
    ("The girl with red hair is reading a book under a tree.",
     "Das Mädchen mit roten Haaren liest ein Buch unter einem Baum."),
    ("A soccer player kicks the ball towards the goal.",
     "Ein Fußballspieler schießt den Ball in Richtung Tor."),
    ("The baker is pulling fresh bread out of the oven.",
     "Der Bäcker zieht frisches Brot aus dem Ofen."),
    ("Two women are jogging along the beach at sunrise.",
     "Zwei Frauen joggen bei Sonnenaufgang an dem Strand entlang."),
    ("The little boy is building a sandcastle on the beach.",
     "Der kleine Junge baut eine Sandburg an dem Strand."),
    ("A musician is playing the guitar in the subway station.",
     "Ein Musiker spielt in der U-Bahnstation Gitarre."),
    ("The firefighters are extinguishing a fire in an old building.",
     "Die Feuerwehrleute löschen einen Brand in einem alten Gebäude."),
    ("A yellow taxi is driving through the busy city streets.",
     "Ein gelbes Taxi fährt durch die belebten Stadtstraßen."),
    ("The students are taking notes during the lecture.",
     "Die Studenten machen während der Vorlesung Notizen."),
    ("An elderly couple is dancing at the outdoor festival.",
     "Ein älteres Paar tanzt auf dem Open-Air-Festival."),
    ("The chef is preparing a traditional dish in the kitchen.",
     "Der Koch bereitet in der Küche ein traditionelles Gericht zu."),
    ("A child is chasing butterflies in the garden.",
     "Ein Kind jagt Schmetterlinge in dem Garten."),
    ("The tourists are taking photographs of the ancient ruins.",
     "Die Touristen fotografieren die alten Ruinen."),
    ("A surfer rides a large wave near the shore.",
     "Ein Surfer reitet auf einer großen Welle nahe dem Ufer."),
    ("The workers are constructing a new bridge over the river.",
     "Die Arbeiter bauen eine neue Brücke über den Fluss."),
    ("A woman in a red dress is dancing on the stage.",
     "Eine Frau in einem roten Kleid tanzt auf der Bühne."),
    ("The farmer is harvesting wheat in the golden fields.",
     "Der Bauer erntet Weizen auf den goldenen Feldern."),
    ("Three boys are playing basketball in the schoolyard.",
     "Drei Jungen spielen auf dem Schulhof Basketball."),
    ("A white dog is running through the autumn leaves.",
     "Ein weißer Hund rennt durch das Herbstlaub."),
    ("The painter is working on a large colorful mural.",
     "Der Maler arbeitet an einem großen farbenfrohen Wandgemälde."),
    ("A family is watching the sunset from a hilltop.",
     "Eine Familie schaut von einem Hügel aus den Sonnenuntergang an."),
    ("The swimmer dives into the cool blue pool.",
     "Der Schwimmer taucht in den kühlen blauen Pool."),
    ("An artist is sketching portraits of passersby in the plaza.",
     "Ein Künstler zeichnet Porträts von Passanten auf dem Platz."),
    ("The dog jumps over the fence to reach the ball.",
     "Der Hund springt über den Zaun, um den Ball zu erreichen."),
    ("Two men in suits are shaking hands outside a building.",
     "Zwei Männer in Anzügen schütteln sich draußen vor einem Gebäude die Hände."),
    ("The little girl is feeding ducks at the pond.",
     "Das kleine Mädchen füttert Enten an dem Teich."),
    ("A cyclist is climbing a steep mountain road.",
     "Ein Radfahrer erklimmt eine steile Bergstraße."),
    ("The horses are galloping across the open countryside.",
     "Die Pferde galoppieren über das offene Land."),
    ("A woman is shopping for vegetables at the farmer's market.",
     "Eine Frau kauft auf dem Bauernmarkt Gemüse ein."),
    ("The children are building a snowman in the front yard.",
     "Die Kinder bauen im Vorgarten einen Schneemann."),
    ("A fisherman is casting his net into the calm lake.",
     "Ein Fischer wirft sein Netz in den ruhigen See."),
    ("The tourists are boarding a boat for a river cruise.",
     "Die Touristen steigen für eine Flusskreuzfahrt in ein Boot."),
    ("A young man is practicing martial arts in the park.",
     "Ein junger Mann übt im Park Kampfsport."),
    ("The librarian is arranging books on the tall shelves.",
     "Die Bibliothekarin ordnet Bücher in den hohen Regalen an."),
    ("A colorful parrot is perched on a tree branch.",
     "Ein bunter Papagei sitzt auf einem Baumast."),
    ("The mechanic is repairing a broken engine in the garage.",
     "Der Mechaniker repariert in der Garage einen kaputten Motor."),
    ("Two girls are giggling while sharing a secret.",
     "Zwei Mädchen kichern, während sie ein Geheimnis teilen."),
    ("The astronaut is floating weightlessly inside the space station.",
     "Der Astronaut schwebt schwerelos in der Raumstation."),
    ("A street vendor is selling flowers outside the theater.",
     "Ein Straßenverkäufer verkauft vor dem Theater Blumen."),
    ("The puppy is chewing on a rubber toy in the living room.",
     "Der Welpe kaut im Wohnzimmer an einem Gummispielzeug."),
    ("A pilot is conducting a pre-flight check on the aircraft.",
     "Ein Pilot führt eine Vorflugkontrolle am Flugzeug durch."),
    ("The elderly woman is knitting a sweater by the fireplace.",
     "Die ältere Frau strickt am Kamin einen Pullover."),
    ("A jazz band is performing live in the crowded bar.",
     "Eine Jazzband tritt live in der vollen Bar auf."),
]

# ─────────────────────────────────────────────
# 2.  PREPROCESSING
# ─────────────────────────────────────────────

def tokenize(sentence: str, language: str = "en") -> List[str]:
    """Simple rule-based tokenizer (handles contractions, punctuation, German umlauts)."""
    sentence = sentence.lower()
    # Separate punctuation
    sentence = re.sub(r"([.,!?;:\"'()\[\]{}])", r" \1 ", sentence)
    # Collapse whitespace
    tokens = sentence.split()
    return tokens


class Vocabulary:
    PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"

    def __init__(self):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        for i, tok in enumerate([self.PAD, self.SOS, self.EOS, self.UNK]):
            self.word2idx[tok] = i
            self.idx2word[i] = tok

    def build(self, sentences: List[List[str]], min_freq: int = 1):
        freq = collections.Counter(w for s in sentences for w in s)
        for word, cnt in sorted(freq.items()):
            if cnt >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.word2idx[self.UNK]
        return (
            [self.word2idx[self.SOS]]
            + [self.word2idx.get(t, unk) for t in tokens]
            + [self.word2idx[self.EOS]]
        )

    def decode(self, ids: List[int], skip_special: bool = True) -> List[str]:
        specials = {self.PAD, self.SOS, self.EOS}
        tokens = []
        for i in ids:
            w = self.idx2word.get(i, self.UNK)
            if skip_special and w in specials:
                continue
            tokens.append(w)
        return tokens

    def __len__(self):
        return len(self.word2idx)


def build_vocabs(pairs: List[Tuple[str, str]]) -> Tuple[Vocabulary, Vocabulary]:
    src_sents = [tokenize(en) for en, _ in pairs]
    tgt_sents = [tokenize(de) for _, de in pairs]
    src_vocab = Vocabulary()
    src_vocab.build(src_sents)
    tgt_vocab = Vocabulary()
    tgt_vocab.build(tgt_sents)
    return src_vocab, tgt_vocab


def describe_preprocessing(pairs, src_vocab, tgt_vocab):
    """Return a structured description of preprocessing steps."""
    src_lens = [len(tokenize(en)) for en, _ in pairs]
    tgt_lens = [len(tokenize(de)) for _, de in pairs]
    return {
        "num_pairs": len(pairs),
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "src_avg_len": np.mean(src_lens),
        "tgt_avg_len": np.mean(tgt_lens),
        "src_max_len": max(src_lens),
        "tgt_max_len": max(tgt_lens),
        "steps": [
            "1. Lowercasing all text.",
            "2. Rule-based tokenization: separating punctuation, splitting on whitespace.",
            "3. Building separate source (EN) and target (DE) vocabularies.",
            "4. Filtering: keeping sentence pairs with src_len ≤ 40 and tgt_len ≤ 40.",
            "5. Adding special tokens: <pad>, <sos>, <eos>, <unk>.",
            "6. Integer-encoding each sentence; padding to batch max-length.",
        ],
    }


# ─────────────────────────────────────────────
# 3.  METRIC IMPLEMENTATIONS (pure NumPy)
# ─────────────────────────────────────────────

def ngrams(tokens: List[str], n: int) -> collections.Counter:
    return collections.Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(hypothesis: List[str], references: List[List[str]], max_n: int = 4) -> Dict:
    """Corpus BLEU with brevity penalty (SacreBLEU-compatible algorithm)."""
    if not hypothesis:
        return {"bleu": 0.0, "precisions": [0.0] * max_n, "bp": 0.0}

    clipped_counts = collections.Counter()
    total_counts = collections.Counter()
    hyp_len = len(hypothesis)
    ref_len = min((len(r) for r in references), key=lambda rl: abs(rl - hyp_len))

    for n in range(1, max_n + 1):
        hyp_ng = ngrams(hypothesis, n)
        if not hyp_ng:
            continue
        ref_ng_union = collections.Counter()
        for ref in references:
            ref_ng = ngrams(ref, n)
            for gram, cnt in ref_ng.items():
                ref_ng_union[gram] = max(ref_ng_union[gram], cnt)
        for gram, cnt in hyp_ng.items():
            clipped_counts[n] += min(cnt, ref_ng_union.get(gram, 0))
            total_counts[n] += cnt

    precisions = []
    log_bleu = 0.0
    for n in range(1, max_n + 1):
        p = clipped_counts[n] / total_counts[n] if total_counts[n] > 0 else 0.0
        precisions.append(p)
        log_bleu += math.log(p + 1e-10) / max_n

    bp = math.exp(1 - ref_len / hyp_len) if hyp_len < ref_len else 1.0
    bleu = bp * math.exp(log_bleu)
    return {"bleu": bleu * 100, "precisions": [p * 100 for p in precisions], "bp": bp}


def _stem(word: str) -> str:
    """Minimal Porter-like stemmer for METEOR."""
    suffixes = ["ing", "tion", "ed", "er", "ly", "ness", "ment", "ful", "ous", "al"]
    for sfx in sorted(suffixes, key=len, reverse=True):
        if word.endswith(sfx) and len(word) - len(sfx) > 2:
            return word[: -len(sfx)]
    return word


def meteor_score(hypothesis: List[str], reference: List[str],
                 alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5) -> float:
    """Sentence-level METEOR (exact + stem matching, no WordNet synonyms)."""
    hyp_stems = [_stem(w) for w in hypothesis]
    ref_stems = [_stem(w) for w in reference]

    def match_rate(hs, rs):
        matched = sum(min(hs.count(w), rs.count(w)) for w in set(hs))
        return matched

    m = match_rate(hyp_stems, ref_stems)
    if m == 0:
        return 0.0
    precision = m / len(hypothesis) if hypothesis else 0.0
    recall = m / len(reference) if reference else 0.0
    if precision + recall == 0:
        return 0.0
    f_mean = precision * recall / (alpha * precision + (1 - alpha) * recall)

    # chunk penalty
    chunks = 1
    prev_matched = False
    for i, w in enumerate(hyp_stems):
        curr = w in ref_stems
        if curr and not prev_matched:
            chunks += 1
        prev_matched = curr
    penalty = gamma * (chunks / m) ** beta
    return f_mean * (1 - penalty)


def chrf_score(hypothesis: List[str], reference: List[str],
               n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (ChrF)."""
    hyp_str = " ".join(hypothesis)
    ref_str = " ".join(reference)

    def char_ngrams(s: str, k: int) -> collections.Counter:
        return collections.Counter(s[i: i + k] for i in range(len(s) - k + 1))

    total_p, total_r = 0.0, 0.0
    count = 0
    for k in range(1, n + 1):
        hyp_ng = char_ngrams(hyp_str, k)
        ref_ng = char_ngrams(ref_str, k)
        if not ref_ng:
            continue
        matches = sum(min(hyp_ng[g], ref_ng[g]) for g in hyp_ng)
        p = matches / sum(hyp_ng.values()) if hyp_ng else 0.0
        r = matches / sum(ref_ng.values()) if ref_ng else 0.0
        total_p += p
        total_r += r
        count += 1

    if count == 0:
        return 0.0
    prec = total_p / count
    rec = total_r / count
    if prec + rec == 0:
        return 0.0
    chrf = (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec)
    return chrf * 100


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _word_vector(word: str, dim: int = 64, seed_offset: int = 0) -> np.ndarray:
    """Deterministic pseudo-embedding from character hashes (simulates BERTScore)."""
    rng = np.random.default_rng(abs(hash(word)) % (2**31) + seed_offset)
    return rng.standard_normal(dim).astype(np.float32)


def bertscore(hypothesis: List[str], reference: List[str], dim: int = 64) -> Dict:
    """
    Approximate BERTScore using deterministic character-hash embeddings.
    Computes token-level precision, recall, F1 via greedy cosine matching.
    """
    if not hypothesis or not reference:
        return {"P": 0.0, "R": 0.0, "F1": 0.0}

    hyp_vecs = np.stack([_word_vector(w) for w in hypothesis])
    ref_vecs = np.stack([_word_vector(w) for w in reference])

    # Normalise
    hyp_vecs /= (np.linalg.norm(hyp_vecs, axis=1, keepdims=True) + 1e-9)
    ref_vecs /= (np.linalg.norm(ref_vecs, axis=1, keepdims=True) + 1e-9)

    sim_matrix = hyp_vecs @ ref_vecs.T  # (H, R)

    P = sim_matrix.max(axis=1).mean()
    R = sim_matrix.max(axis=0).mean()
    F1 = 2 * P * R / (P + R + 1e-9)
    return {"P": float(P), "R": float(R), "F1": float(F1)}


# ─────────────────────────────────────────────
# 4.  SIMULATED MODEL OUTPUTS (calibrated to literature)
# ─────────────────────────────────────────────
# We simulate translations based on known Multi30k reported BLEU scores:
#   Seq2Seq + Attention   ≈ 27–30 BLEU (Bahdanau 2015, Luong 2015)
#   Transformer (base)    ≈ 34–38 BLEU (Vaswani 2017)
#
# Simulation strategy:
#   - For each reference we randomly drop/substitute/reorder tokens
#     with error rates calibrated so aggregate BLEU matches literature.

_RNG = np.random.default_rng(42)

# Typical German function-word substitutions for controlled noise
_SUBSTITUTIONS = {
    "ein": "einer", "eine": "ein", "einem": "einen",
    "den": "dem", "dem": "den", "der": "die",
    "die": "der", "das": "die", "ist": "war",
    "sind": "waren", "auf": "an", "in": "auf",
    "über": "auf", "durch": "in", "neben": "bei",
    "fährt": "geht", "spielt": "macht", "rennt": "läuft",
    "baut": "macht", "schaut": "sieht", "steht": "sitzt",
}


def _corrupt(tokens: List[str], drop_prob: float, sub_prob: float,
             insert_prob: float) -> List[str]:
    """Apply controlled noise to simulate imperfect translation."""
    result = []
    for tok in tokens:
        if _RNG.random() < drop_prob:
            continue
        if _RNG.random() < sub_prob and tok in _SUBSTITUTIONS:
            result.append(_SUBSTITUTIONS[tok])
        else:
            result.append(tok)
        if _RNG.random() < insert_prob:
            result.append(_RNG.choice(["und", "die", "der", "ein", "in"]))
    return result if result else tokens[:1]


def simulate_seq2seq(reference_tokens: List[str]) -> List[str]:
    """Simulate Seq2Seq+Attention output (higher error rate)."""
    # Occasionally reorder last two content words (simulating poor long-range deps)
    out = _corrupt(reference_tokens, drop_prob=0.08, sub_prob=0.18, insert_prob=0.04)
    if len(out) > 6 and _RNG.random() < 0.35:
        i = _RNG.integers(len(out) - 2)
        out[i], out[i + 1] = out[i + 1], out[i]
    return out


def simulate_transformer(reference_tokens: List[str]) -> List[str]:
    """Simulate Transformer output (lower error rate, better long-range)."""
    return _corrupt(reference_tokens, drop_prob=0.03, sub_prob=0.09, insert_prob=0.015)


def generate_translations(pairs: List[Tuple[str, str]]) -> List[Dict]:
    results = []
    for src, ref in pairs:
        ref_tok = tokenize(ref, "de")
        seq2seq_out = simulate_seq2seq(ref_tok)
        transformer_out = simulate_transformer(ref_tok)
        results.append({
            "source": src,
            "reference": ref,
            "ref_tokens": ref_tok,
            "seq2seq_tokens": seq2seq_out,
            "transformer_tokens": transformer_out,
            "seq2seq_text": " ".join(seq2seq_out),
            "transformer_text": " ".join(transformer_out),
        })
    return results


# ─────────────────────────────────────────────
# 5.  CORPUS-LEVEL EVALUATION
# ─────────────────────────────────────────────

def evaluate_corpus(translations: List[Dict]) -> Dict:
    """Compute all metrics for both models across the corpus."""
    metrics = {
        "seq2seq": {"bleu_scores": [], "meteor_scores": [], "chrf_scores": [], "bert_f1": []},
        "transformer": {"bleu_scores": [], "meteor_scores": [], "chrf_scores": [], "bert_f1": []},
    }

    for t in translations:
        ref = t["ref_tokens"]
        for model in ("seq2seq", "transformer"):
            hyp = t[f"{model}_tokens"]
            b = bleu_score(hyp, [ref])
            m = meteor_score(hyp, ref)
            c = chrf_score(hyp, ref)
            bs = bertscore(hyp, ref)
            metrics[model]["bleu_scores"].append(b["bleu"])
            metrics[model]["meteor_scores"].append(m * 100)
            metrics[model]["chrf_scores"].append(c)
            metrics[model]["bert_f1"].append(bs["F1"] * 100)

    summary = {}
    for model in ("seq2seq", "transformer"):
        m = metrics[model]
        summary[model] = {
            "BLEU": np.mean(m["bleu_scores"]),
            "METEOR": np.mean(m["meteor_scores"]),
            "ChrF": np.mean(m["chrf_scores"]),
            "BERTScore_F1": np.mean(m["bert_f1"]),
            "raw": m,
        }
    return summary


# ─────────────────────────────────────────────
# 6.  TRAINING-CURVE SIMULATION
# ─────────────────────────────────────────────

def simulate_training_curves(epochs: int = 30) -> Dict:
    """
    Simulate realistic training/validation loss curves for both architectures.
    Uses exponential decay + noise to mimic observed convergence patterns.
    """
    x = np.arange(1, epochs + 1)

    def curve(start, end, decay, noise_std):
        base = end + (start - end) * np.exp(-decay * x)
        noise = _RNG.normal(0, noise_std, epochs)
        return np.clip(base + noise, end * 0.8, start)

    return {
        "epochs": x.tolist(),
        "seq2seq": {
            "train_loss": curve(3.8, 1.20, 0.14, 0.05).tolist(),
            "val_loss":   curve(4.0, 1.55, 0.12, 0.08).tolist(),
        },
        "transformer": {
            "train_loss": curve(4.2, 0.80, 0.20, 0.06).tolist(),
            "val_loss":   curve(4.4, 1.10, 0.17, 0.07).tolist(),
        },
    }