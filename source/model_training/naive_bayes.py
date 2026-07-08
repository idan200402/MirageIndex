import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# utility function imports
from source.utils.training_metrics import classification_metrics, maybe_export_metrics_json, project_relative_path
from source.utils.data import load_records, split_records, print_label_distribution
from source.utils.text import tokenize, record_to_text, build_train_chunk_examples
from source.utils.general import add_common_parsing, spans_suffix, score_records_by_chunks, span_config_payload

# utility constant imports
from source.utils.data import LABEL_FIELD
from source.utils.text import TEXT_FIELDS, POSITIVE_LABEL

# file specific constants
MODEL_NAME = "naive_bayes"

class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be greater than 0")
        self.alpha = alpha
        self.labels: list[str] = []
        self.vocabulary: set[str] = set()
        self.label_counts: Counter[str] = Counter()
        self.token_counts_by_label: dict[str, Counter[str]] = defaultdict(Counter)
        self.total_tokens_by_label: Counter[str] = Counter()

    def fit(self, texts: list[str], labels: list[str]) -> None:
        """Estimate per-label priors and token frequencies from the training data.

        texts: raw training documents. 
        labels: the class label for each document, aligned with texts.\\
        Returns nothing.\\
        The learned counts and vocabulary are stored on the instance.
        """
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if not texts:
            raise ValueError("Cannot train on an empty dataset")

        for text, label in zip(texts, labels):
            # count documents per label to form the class priors later
            self.label_counts[label] += 1
            tokens = tokenize(text)
            # the vocabulary is the union of every token seen in training
            self.vocabulary.update(tokens)
            # accumulate per-label token frequencies and their running totals
            self.token_counts_by_label[label].update(tokens)
            self.total_tokens_by_label[label] += len(tokens)

        # a stable, sorted label order keeps downstream iteration deterministic
        self.labels = sorted(self.label_counts)

    def predict(self, texts: list[str]) -> list[str]:
        """Return the most likely label for each input document.

        texts: raw documents to classify.\\
        Returns the argmax-log-probability label per document, in input order.
        """
        return [max(self._log_probabilities(text), key=self._log_probabilities(text).get) for text in texts]

    def predict_positive_scores(self, texts: list[str], positive_label: str) -> list[float]:
        """Return the normalized probability of positive_label for each document.

        texts: raw documents to score.
        positive_label: the class of interest.\\
        Returns one probability in [0, 1] per document, in input order.
        """
        return [self._probabilities(text).get(positive_label, 0.0) for text in texts]

    def _log_probabilities(self, text: str) -> dict[str, float]:
        """Compute the unnormalized log-posterior of each label for one document.

        text: the raw document to score.\\
        Returns a {label: log_probability} mapping over every known label.
        """
        total_examples = sum(self.label_counts.values())
        vocabulary_size = len(self.vocabulary)
        tokens = tokenize(text)
        log_probabilities = {}

        for label in self.labels:
            # start from the log prior: the share of training docs with this label
            log_probability = math.log(self.label_counts[label] / total_examples)
            # Laplace smoothing spreads alpha mass across the whole vocabulary
            denominator = self.total_tokens_by_label[label] + self.alpha * vocabulary_size

            for token in tokens:
                # summing log-likelihoods keeps the product numerically stable
                token_count = self.token_counts_by_label[label][token]
                log_probability += math.log((token_count + self.alpha) / denominator)

            log_probabilities[label] = log_probability

        return log_probabilities

    def _probabilities(self, text: str) -> dict[str, float]:
        """Convert the log-posteriors of one document into normalized probabilities.

        text: the raw document to score.\\
        Returns a {label: probability} mapping whose values sum to 1.
        """
        log_probabilities = self._log_probabilities(text)
        # subtracting the max before exponentiating avoids overflow (log-sum-exp trick)
        max_log_probability = max(log_probabilities.values())
        exp_values = {
            label: math.exp(log_probability - max_log_probability)
            for label, log_probability in log_probabilities.items()
        }
        # dividing by the total renormalizes the shifted exponentials back to probabilities
        normalizer = sum(exp_values.values())
        return {label: value / normalizer for label, value in exp_values.items()}

def parse_args() -> argparse.Namespace:
    """Build the command-line argument parser and return the parsed arguments.

    Combines the arguments shared by every model with the Naive Bayes specific smoothing option. 
    Returns the populated argparse.Namespace.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a Naive Bayes text baseline model.")
    # parsers that are general to all models
    parser = add_common_parsing(parser)
    # model specific parser
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help=f"Model name used for artifact export. Defaults to {MODEL_NAME}.",
    )
    # parsers relating specifically to naive bayes parameters
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing value.",
    )
    
    return parser.parse_args()

def main() -> None:
    """Run the end-to-end training and evaluation pipeline for this baseline.

    Loads the dataset, splits it, trains the Naive Bayes model on the raw text,
    evaluates it on the test split, optionally exports the metrics JSON, and prints
    a human-readable summary. Takes no arguments and returns nothing.
    """
    args = parse_args()
    # fork artifacts to naive_bayes_spans in span mode; inert (suffix "") otherwise
    args.model_name = args.model_name + spans_suffix(args)
    records = load_records(args.data)
    train_records, test_records = split_records(records, args.test_size, args.seed)

    y_true = [record[LABEL_FIELD] for record in test_records]
    model = MultinomialNaiveBayes(alpha=args.alpha)
    build_audit = None

    if args.use_spans:
        # cross-encoder fallback: each (query, chunk) pair is one document string, fed
        # to the SAME NB estimator. No class weighting (intentional -- see blueprint).
        queries, chunks, chunk_labels, build_audit = build_train_chunk_examples(
            train_records, args.chunk_window, args.chunk_stride, args.overlap_threshold
        )
        train_texts = [f"{query}\n{chunk}" for query, chunk in zip(queries, chunks)]
        train_str_labels = [POSITIVE_LABEL if label == 1 else "no" for label in chunk_labels]
        model.fit(train_texts, train_str_labels)
        # NB needs no vectorizer, so score raw chunk text directly
        y_score, y_pred = score_records_by_chunks(test_records, model, None, args)
    else:
        train_texts = [record_to_text(record) for record in train_records]
        train_labels = [record[LABEL_FIELD] for record in train_records]
        test_texts = [record_to_text(record) for record in test_records]
        model.fit(train_texts, train_labels)
        y_pred = model.predict(test_texts)
        y_score = model.predict_positive_scores(test_texts, POSITIVE_LABEL)

    metrics = classification_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score, positive_label=POSITIVE_LABEL)
    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))

    metrics_payload = {
        "model_name": args.model_name,
        "data_path": project_relative_path(args.data),
        "seed": args.seed,
        "test_size": args.test_size,
        "alpha": args.alpha,
        "train_examples": len(train_records),
        "test_examples": len(test_records),
        "vocabulary_size": len(model.vocabulary),
        "text_fields": list(TEXT_FIELDS),
        "trained_parameters": {
            "alpha": args.alpha,
            "labels": model.labels,
            "label_counts": dict(model.label_counts),
            "total_tokens_by_label": dict(model.total_tokens_by_label),
            "vocabulary_size": len(model.vocabulary),
        },
        "metrics": metrics,
    }
    if args.use_spans:
        # span-mode only additions; the baseline payload stays byte-identical
        metrics_payload["span_config"] = span_config_payload(args)
        metrics_payload["span_coverage"] = build_audit
    metrics_path = maybe_export_metrics_json(
        enabled=args.export_metrics,
        model_name=args.model_name,
        metrics=metrics_payload,
        artifacts_dir=args.artifacts_dir,
    )

    print("Naive Bayes Baseline")
    print("--------------------")
    print(f"data_path: {args.data}")
    print(f"seed: {args.seed}")
    print(f"test_size: {args.test_size}")
    print(f"alpha: {args.alpha}")
    print(f"export_metrics: {args.export_metrics}")
    print(f"train_examples: {len(train_records)}")
    print(f"test_examples: {len(test_records)}")
    print(f"vocabulary_size: {len(model.vocabulary)}")
    print(f"accuracy: {metrics['accuracy']:.4f} ({correct}/{len(test_records)})")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"pr_auc: {metrics['pr_auc']:.4f}" if metrics["pr_auc"] is not None else "pr_auc: undefined")
    print(f"roc_auc: {metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "roc_auc: undefined")
    if metrics_path is not None:
        print(f"metrics_path: {metrics_path}")

    print_label_distribution("Train Label Distribution", train_records)
    print_label_distribution("Test Label Distribution", test_records)


if __name__ == "__main__":
    main()
