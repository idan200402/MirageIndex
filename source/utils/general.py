# standard library imports
import argparse
from pathlib import Path
import math
from typing import Any

# utility function imports
from source.utils.training_metrics import parse_bool
from source.utils.text import build_chunk_examples, POSITIVE_LABEL
# utility constant imports
from source.utils.data import DEFAULT_DATA_PATH, DEFAULT_ARTIFACTS_DIR, DEFAULT_SEED, DEFAULT_TEST_SIZE

def add_common_parsing(parser: argparse.ArgumentParser ) -> argparse.ArgumentParser:
    """Attach the command-line arguments shared by every model onto a parser.

    parser: an argparse.ArgumentParser to extend in place.\\
    Adds the dataset path, seed, test-size, metric-export flag, and artifacts
    directory options, then returns the same parser so calls can be chained.
    """
    # parsers relating to general model interactions
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the dataset. Defaults to {DEFAULT_DATA_PATH}",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed used to split the data.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of examples to use for testing.",
    )

    # parsers relating to artifact exports and test matrices
    parser.add_argument(
        "--export-metrics",
        type=parse_bool,
        default=False,
        help="True exports test metrics JSON to artifacts/model_name. False skips export.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Directory where exported metrics are written. Defaults to {DEFAULT_ARTIFACTS_DIR}.",
    )

    # span-mode arguments (all inert when --use-spans is False, and never serialized
    # into any artifact in that case, so the baseline output stays byte-identical)
    parser.add_argument(
        "--use-spans",
        type=parse_bool,
        default=False,
        help="True trains at the hallucination-span chunk level; False (default) is the document-level baseline.",
    )
    parser.add_argument(
        "--chunk-window",
        type=int,
        default=40,
        help="Word-window size for span-mode chunking.",
    )
    parser.add_argument(
        "--chunk-stride",
        type=int,
        default=20,
        help="Word-window stride for span-mode chunking.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.25,
        help="Min fraction of a chunk covered by a span for the chunk to be labeled hallucinated.",
    )
    parser.add_argument(
        "--aggregation",
        choices=("max", "mean_topk", "noisy_or"),
        default="max",
        help="How chunk scores are aggregated to a document score at inference.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="K for the mean_topk aggregation.")
    parser.add_argument(
        "--response-threshold",
        type=float,
        default=0.5,
        help="Decision threshold on the aggregated response score. Independent of --use-spans; always defined.",
    )

    return parser

def sigmoid(value: float) -> float:
    """Map a real-valued margin to a probability in the open interval (0, 1).

    value: the raw logit or margin to squash.\\
    Returns the logistic sigmoid of value, computed in a numerically stable way
    that avoids overflow for large-magnitude inputs.
    """
    # for non-negative inputs the standard form keeps exp arguments non-positive
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    # for negative inputs use the equivalent form so exp never overflows
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


# ---------------------------------------------------------------------------
# Span-mode aggregation and classical chunk-inference helpers
# ---------------------------------------------------------------------------

def spans_suffix(args: Any) -> str:
    """Return the artifact-name suffix that forks span-mode outputs from the baseline.

    args: the parsed argument namespace.\\
    Returns "_spans" when --use-spans is set, else "". Callers append this to
    args.model_name so span-aware runs write to a sibling artifact directory.
    """
    return "_spans" if getattr(args, "use_spans", False) else ""


def span_config_payload(args: Any) -> dict[str, Any]:
    """Collect the span-mode chunk/aggregation settings for the metrics.json payload.

    args: the parsed argument namespace.\\
    Returns the span_config block recorded alongside baseline metrics keys in span mode.
    """
    return {
        "chunk_window": args.chunk_window,
        "chunk_stride": args.chunk_stride,
        "overlap_threshold": args.overlap_threshold,
        "aggregation": args.aggregation,
        "top_k": args.top_k,
        "response_threshold": args.response_threshold,
    }


def compute_class_weight_ratio(labels: list[int], clamp: tuple[float, float] = (1.0, 20.0)) -> float:
    """Compute the positive-class weight for the negative-skewed chunk distribution.

    labels: the 0/1 chunk labels. clamp: (low, high) bounds on the returned ratio.\\
    Returns neg_count / max(pos_count, 1), clamped to [low, high]. Shared by the
    discriminative classical models (logistic regression, random forest, xgboost) to
    up-weight positive chunks; naive_bayes deliberately does not use it. This is the
    classical analogue of LLM_train.compute_pos_weight, returning a plain float.
    """
    neg_count = sum(1 for label in labels if label == 0)
    pos_count = sum(1 for label in labels if label == 1)
    ratio = neg_count / max(pos_count, 1)
    return min(clamp[1], max(clamp[0], ratio))


def aggregate_document_score(
    chunk_scores: list[float],
    method: str,
    top_k: int,
    chunk_spans: list[tuple[int, int]] | None = None,
) -> float:
    """Aggregate a document's chunk positive-scores into a single response score.

    chunk_scores: per-chunk positive probabilities. method: one of "max", "mean_topk",
    "noisy_or". top_k: K for mean_topk. chunk_spans: per-chunk character spans, required
    by noisy_or to decluster correlated overlapping chunks.\\
    Returns a score in [0, 1]; an empty document scores 0.0. Raises ValueError on an
    unknown method. For noisy_or, when chunk_spans is provided, overlapping chunks are
    first merged into clusters (keeping the max score per cluster) so the 50%-overlap
    sliding window does not inflate the product with duplicated evidence.
    """
    if not chunk_scores:
        return 0.0
    # clamp defensively so a stray out-of-range score cannot break the aggregation
    scores = [min(1.0, max(0.0, score)) for score in chunk_scores]

    if method == "max":
        return max(scores)

    if method == "mean_topk":
        k = min(top_k, len(scores))
        top_scores = sorted(scores, reverse=True)[:k]
        return sum(top_scores) / len(top_scores)

    if method == "noisy_or":
        if chunk_spans is not None:
            # decluster overlapping chunks so one hallucinated region is not counted
            # by several correlated near-duplicate windows
            paired = sorted(zip(chunk_spans, scores), key=lambda item: item[0][0])
            cluster_scores = []
            current_end = None
            current_max = 0.0
            for (start, end), score in paired:
                if current_end is not None and start < current_end:
                    # this chunk overlaps the running cluster, so extend and keep the max
                    current_end = max(current_end, end)
                    current_max = max(current_max, score)
                else:
                    # close the previous cluster (if any) and open a new one
                    if current_end is not None:
                        cluster_scores.append(current_max)
                    current_end = end
                    current_max = score
            if current_end is not None:
                cluster_scores.append(current_max)
        else:
            # legacy fallback: no spans supplied, so use the undeclustered scores
            cluster_scores = scores

        product = 1.0
        for score in cluster_scores:
            product *= 1 - score
        return 1 - product

    raise ValueError(f"Unknown aggregation method: {method!r}")


def document_label(response_score: float, threshold: float, positive_label: str) -> str:
    """Threshold an aggregated response score into a hard document label.

    response_score: the aggregated score. threshold: the decision threshold.
    positive_label: the label returned when the score clears the threshold.\\
    Returns positive_label when response_score >= threshold, else "no".
    """
    return positive_label if response_score >= threshold else "no"


def group_scores_by_doc(
    chunk_scores: list[float],
    chunk_spans: list[tuple[int, int]],
    doc_index: list[int],
    n_docs: int,
) -> tuple[list[list[float]], list[list[tuple[int, int]]]]:
    """Group flat chunk scores and spans back to their parent documents.

    chunk_scores / chunk_spans: per-chunk scores and character spans. doc_index: the
    positional document index of each chunk. n_docs: number of documents.\\
    Returns (grouped_scores, grouped_spans), each a length-n_docs list preserving chunk
    order, both empty for documents with no chunks. Asserts the DOC-TO-CHUNK ALIGNMENT
    INVARIANT (matching lengths and in-bounds indices) as a second line of defense.
    """
    assert len(chunk_scores) == len(chunk_spans) == len(doc_index)
    if doc_index:
        assert max(doc_index) < n_docs

    grouped_scores: list[list[float]] = [[] for _ in range(n_docs)]
    grouped_spans: list[list[tuple[int, int]]] = [[] for _ in range(n_docs)]
    for score, span, index in zip(chunk_scores, chunk_spans, doc_index):
        grouped_scores[index].append(score)
        grouped_spans[index].append(span)
    return grouped_scores, grouped_spans


def score_records_by_chunks(
    records: list[dict[str, Any]],
    model: Any,
    vectorizer: Any,
    args: Any,
) -> tuple[list[float], list[str]]:
    """Score records at the chunk level and aggregate to per-document predictions.

    records: the records to score, in order. model: a fitted classical model exposing
    predict_positive_scores. vectorizer: a fitted TfidfVectorizer, or None for models
    (Naive Bayes) that score raw text directly. args: supplies the chunk/aggregation config.\\
    Returns (response_scores, y_pred): the aggregated per-document scores and their
    thresholded labels. Shared by every classical model so aggregation is identical.
    """
    _queries, chunks, chunk_spans, doc_index, n_docs = build_chunk_examples(
        records, args.chunk_window, args.chunk_stride
    )
    # score each chunk on its own text: the query is constant across a document's chunks,
    # so prepending it dilutes the chunk-specific TF-IDF signal and attaches identical
    # query tokens to both positive and negative chunks of the same document
    chunk_documents = list(chunks)

    if vectorizer is not None:
        vectors = vectorizer.transform(chunk_documents)
        chunk_scores = model.predict_positive_scores(vectors)
    else:
        # Naive Bayes has no vectorizer and scores raw text against the positive label
        chunk_scores = model.predict_positive_scores(chunk_documents, POSITIVE_LABEL)

    grouped_scores, grouped_spans = group_scores_by_doc(chunk_scores, chunk_spans, doc_index, n_docs)
    response_scores = [
        aggregate_document_score(scores, args.aggregation, args.top_k, chunk_spans=spans)
        for scores, spans in zip(grouped_scores, grouped_spans)
    ]
    y_pred = [document_label(score, args.response_threshold, POSITIVE_LABEL) for score in response_scores]

    assert len(response_scores) == len(records) == n_docs
    return response_scores, y_pred