# imports
import math
import re
from typing import Any
from collections import Counter

# constants
TEXT_FIELDS = ("user_query", "chatgpt_response")
POSITIVE_LABEL = "yes"
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_']+")

def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens.

    Lowercases the text, then extracts runs of letters, digits, underscores, and
    apostrophes (via ``TOKEN_PATTERN``), dropping punctuation and whitespace. For
    example, "Don't PANIC!" -> ["don't", "panic"]. This is the shared tokenizer
    the text-based models (Naive Bayes, TF-IDF) use so their vocabularies match.
    """
    return TOKEN_PATTERN.findall(text.lower())

def record_to_text(record: dict[str, Any]) -> str:
    """Flatten a record's text fields into a single string.

    Concatenates the ``TEXT_FIELDS`` ("user_query", "chatgpt_response") of a
    record into one newline-separated string, which is what the models actually
    tokenize and train on. Missing fields are treated as empty, and values are
    coerced to str so non-string fields don't break the join.
    """
    return "\n".join(str(record.get(field, "")) for field in TEXT_FIELDS)

class TfidfVectorizer:
    """Turn raw texts into TF-IDF feature vectors.

    Learns a vocabulary from a training corpus and converts each document into a
    sparse, L2-normalized TF-IDF vector. "Sparse" here means each vector is a
    ``dict[int, float]`` mapping a term's vocabulary index to its weight, terms
    not present in the document are simply absent (implicitly 0.0).

    Weighting scheme:
      - TF  = term count in the document / total terms in the document
      - IDF = log((1 + N) / (1 + df)) + 1, where N is the number of documents and
        df is the number of documents containing the term (smoothed so no term
        gets a zero or undefined weight)
      - each vector is then L2-normalized (divided by its Euclidean norm)

    Follows the fit / transform / fit_transform convention: call ``fit`` (or
    ``fit_transform``) on the training texts, then ``transform`` on any later
    texts using the same learned vocabulary.

    Args:
        max_features: keep at most this many terms, ranked by overall frequency.
        min_df: drop terms that appear in fewer than this many documents.

    Attributes:
        vocabulary: mapping of term -> column index (empty until fitted).
        idf: per-term IDF weights, aligned to the vocabulary indices.
    """

    def __init__(self, max_features: int = 20000, min_df: int = 1) -> None:
        """Configure the vectorizer, does not learn anything yet.

        Validates the hyperparameters and initializes an empty vocabulary/IDF
        table. Raises ValueError if ``max_features`` or ``min_df`` is not positive.
        """
        if max_features <= 0:
            raise ValueError("max_features must be greater than 0")
        if min_df <= 0:
            raise ValueError("min_df must be greater than 0")
        self.max_features = max_features
        self.min_df = min_df
        # vocabulary and idf stay empty until fit learns them from a corpus
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []

    def fit(self, texts: list[str]) -> None:
        """Learn the vocabulary and IDF weights from a training corpus.

        Tokenizes every document, counts overall term frequency and per-term
        document frequency, then selects the terms: those appearing in at least
        ``min_df`` documents, ranked by total frequency, capped at
        ``max_features``. The selected terms become ``vocabulary`` (term -> index)
        and their IDF weights are computed into ``idf``. Nothing is returned.
        state is stored on the instance.
        """
        document_frequency: Counter[str] = Counter()
        term_frequency: Counter[str] = Counter()

        # tally total occurrences and per-document presence for every token
        for text in texts:
            tokens = tokenize(text)
            term_frequency.update(tokens)
            # a set so each token contributes once per document
            document_frequency.update(set(tokens))

        # keep the most frequent terms that clear min_df, capped at max_features
        terms = [
            term
            for term, frequency in term_frequency.most_common()
            if document_frequency[term] >= self.min_df
        ][: self.max_features]

        # assign each surviving term a stable column index
        self.vocabulary = {term: index for index, term in enumerate(terms)}
        document_count = len(texts)
        # smoothed inverse document frequency so no term gets a zero or undefined weight
        self.idf = [
            math.log((1 + document_count) / (1 + document_frequency[term])) + 1
            for term in terms
        ]

    def transform(self, texts: list[str]) -> list[dict[int, float]]:
        """Convert texts into TF-IDF vectors using the learned vocabulary.

        For each document: count only the tokens that are in the vocabulary,
        weight them by TF * IDF, then L2-normalize the vector. Out-of-vocabulary
        tokens are ignored, and a document with no known tokens yields an empty
        vector ({}). Returns one ``dict[int, float]`` per input text, in order.

        Must be called after ``fit``, raises ValueError if the vocabulary is empty.
        """
        if not self.vocabulary:
            raise ValueError("Vectorizer must be fitted before transform")

        vectors = []
        for text in texts:
            # count only the tokens that made it into the vocabulary
            counts: Counter[int] = Counter()
            for token in tokenize(text):
                index = self.vocabulary.get(token)
                if index is not None:
                    counts[index] += 1

            total_terms = sum(counts.values())
            vector = {}
            if total_terms:
                # weight each term by its term frequency times its idf
                for index, count in counts.items():
                    vector[index] = (count / total_terms) * self.idf[index]

                # l2-normalize so document length does not dominate the vector
                norm = math.sqrt(sum(value * value for value in vector.values()))
                if norm:
                    vector = {index: value / norm for index, value in vector.items()}

            vectors.append(vector)

        return vectors

    def fit_transform(self, texts: list[str]) -> list[dict[int, float]]:
        """Fit on the texts and return their vectors in one step.

        Convenience for the training corpus: equivalent to calling ``fit`` then
        ``transform`` on the same texts. Use ``transform`` alone for later data
        (e.g. the test set) so it is encoded with the vocabulary learned here.
        """
        self.fit(texts)
        return self.transform(texts)