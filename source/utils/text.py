# imports
import math
import re
from typing import Any
from collections import Counter

# label field lives in data.py; imported here for the span-aware training builder
from source.utils.data import LABEL_FIELD

# constants
TEXT_FIELDS = ("user_query", "chatgpt_response")
POSITIVE_LABEL = "yes"
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_']+")

# schema field names used by the span-mode pipeline (document-level baseline uses
# TEXT_FIELDS / record_to_text instead and is unaffected by these)
QUERY_FIELD = "user_query"
RESPONSE_FIELD = "chatgpt_response"
SPANS_FIELD = "hallucination_spans"

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


# ---------------------------------------------------------------------------
# Span-mode chunking and labeling helpers
#
# These power the opt-in ``--use-spans`` mode, which reframes the unit of
# prediction from the whole (query, response) document to (query, chunk) pairs.
# The chunk boundaries are computed spans-BLIND (chunk_response), and only
# resolve_span_offsets / assign_chunk_labels are allowed to read the annotated
# hallucination_spans, and only AFTER boundaries are fixed. This ordering keeps
# chunk length/position/count from leaking span-annotation artifacts.
# ---------------------------------------------------------------------------

def chunk_response(response_text: str, window: int, stride: int) -> list[tuple[int, int]]:
    """Slide a fixed word window over a response and return chunk character spans.

    response_text: the raw response text (NOT lowercased, so match offsets index
    the original string). window: window size in word tokens. stride: step in word
    tokens between consecutive windows.\\
    Returns a list of (start_char, end_char) spans, one per window, where the span
    runs from the first word's start offset to the last word's end offset. Returns
    [] when the response has no word tokens, and a single all-covering chunk when
    there are fewer words than ``window``. This function is spans-BLIND: it never
    receives or consults hallucination_spans. Raises ValueError on invalid window
    or stride.
    """
    if window <= 0:
        raise ValueError("window must be greater than 0")
    if not 0 < stride <= window:
        raise ValueError("stride must satisfy 0 < stride <= window")

    # real character offsets of every word token, taken on the original casing
    word_spans = [match.span() for match in TOKEN_PATTERN.finditer(response_text)]
    if not word_spans:
        return []

    chunks = []
    total = len(word_spans)
    start = 0
    while start < total:
        end = min(start + window, total)
        window_spans = word_spans[start:end]
        # the chunk spans from the first word's start to the last word's end
        chunks.append((window_spans[0][0], window_spans[-1][1]))
        # stop once this window reached the end so the tail is not re-emitted
        if end == total:
            break
        start += stride
    return chunks


def chunk_text_of(response_text: str, char_span: tuple[int, int]) -> str:
    """Materialize the substring a chunk's character span refers to.

    response_text: the raw response text. char_span: the (start, end) character
    offsets produced by ``chunk_response``.\\
    Returns response_text[start:end], the candidate chunk string paired with the query.
    """
    start, end = char_span
    return response_text[start:end]


def resolve_span_offsets(response_text: str, spans: list[str]) -> list[tuple[int, int]]:
    """Locate every verbatim occurrence of each annotated span in the response.

    response_text: the raw response text. spans: the ``hallucination_spans`` list,
    whose entries are exact substrings of the response (some may not appear verbatim).\\
    Returns the list of (start, end) character intervals for every occurrence found.
    Empty spans and spans that never appear verbatim are silently skipped, so the
    result may be empty. This is the ONLY function permitted to read
    hallucination_spans, and it runs only AFTER chunk boundaries are fixed.
    """
    offsets = []
    for span in spans:
        if not span:
            continue
        search_from = 0
        # walk past each hit so every non-overlapping occurrence is captured
        while True:
            found = response_text.find(span, search_from)
            if found == -1:
                break
            offsets.append((found, found + len(span)))
            search_from = found + len(span)
    return offsets


def chunk_coverage_fractions(
    chunk_spans: list[tuple[int, int]],
    span_offsets: list[tuple[int, int]],
) -> list[float]:
    """Fraction of each chunk's characters covered by the union of resolved spans.

    chunk_spans: the (start, end) character spans of each chunk. span_offsets: the
    resolved hallucination-span intervals from ``resolve_span_offsets``.\\
    Returns one float in [0, 1] per chunk: the share of the chunk's characters covered
    by the UNION of the span intervals (overlapping spans merged so double-covered
    characters are counted once). A degenerate zero-length chunk yields 0.0. Consults
    only character overlap, never chunk index/position. This is the shared coverage
    primitive used both to threshold labels and to pick a fallback positive chunk.
    """
    fractions = []
    for chunk_start, chunk_end in chunk_spans:
        chunk_length = chunk_end - chunk_start
        # a degenerate zero-length chunk cannot be covered, so it stays 0.0
        if chunk_length <= 0:
            fractions.append(0.0)
            continue

        # intersect the chunk with each span, keeping only non-empty overlaps
        intersections = []
        for span_start, span_end in span_offsets:
            low = max(chunk_start, span_start)
            high = min(chunk_end, span_end)
            if low < high:
                intersections.append((low, high))

        # merge the intersection intervals so shared characters are counted once
        covered = 0
        if intersections:
            intersections.sort()
            current_low, current_high = intersections[0]
            for low, high in intersections[1:]:
                if low <= current_high:
                    current_high = max(current_high, high)
                else:
                    covered += current_high - current_low
                    current_low, current_high = low, high
            covered += current_high - current_low

        fractions.append(covered / chunk_length)
    return fractions


def assign_chunk_labels(
    chunk_spans: list[tuple[int, int]],
    span_offsets: list[tuple[int, int]],
    overlap_threshold: float,
) -> list[int]:
    """Label each chunk by how much of it the resolved hallucination spans cover.

    chunk_spans: the (start, end) character spans of each chunk. span_offsets: the
    resolved hallucination-span intervals from ``resolve_span_offsets``.
    overlap_threshold: minimum covered fraction for a chunk to count as hallucinated.\\
    Returns one 0/1 label per chunk: 1 when the chunk's covered fraction (from
    ``chunk_coverage_fractions``) is >= overlap_threshold, else 0.
    """
    return [
        1 if fraction >= overlap_threshold else 0
        for fraction in chunk_coverage_fractions(chunk_spans, span_offsets)
    ]


def build_chunk_examples(
    records: list[dict[str, Any]],
    window: int,
    stride: int,
) -> tuple[list[str], list[str], list[tuple[int, int]], list[int], int]:
    """Build spans-FREE (query, chunk) examples for validation/test/inference.

    records: the split's records, consumed in order. window / stride: chunking config.\\
    Returns (queries, chunks, chunk_spans, doc_index, n_docs). For each record, its
    response is chunked and one entry per chunk is appended: the record's query text,
    the chunk text, the chunk's (start, end) character span, and the record's positional
    index. Records that yield zero chunks still count toward ``n_docs`` and simply
    contribute no entries (their document later aggregates to 0.0). Carries NO labels.

    The chunk spans are always returned (not optional) because the noisy-OR aggregation
    declusters correlated overlapping chunks by their character position. The alignment
    assertions guard the DOC-TO-CHUNK ALIGNMENT INVARIANT: callers must pass records in
    the same order later used to build y_true, with no filtering/resorting in between.
    """
    queries: list[str] = []
    chunks: list[str] = []
    chunk_spans: list[tuple[int, int]] = []
    doc_index: list[int] = []

    for index, record in enumerate(records):
        response_text = str(record.get(RESPONSE_FIELD, ""))
        query = str(record.get(QUERY_FIELD, ""))
        for span in chunk_response(response_text, window, stride):
            queries.append(query)
            chunks.append(chunk_text_of(response_text, span))
            chunk_spans.append(span)
            doc_index.append(index)

    n_docs = len(records)
    assert len(queries) == len(chunks) == len(chunk_spans) == len(doc_index)
    if doc_index:
        assert max(doc_index) < n_docs
    return queries, chunks, chunk_spans, doc_index, n_docs


def build_train_chunk_examples(
    records: list[dict[str, Any]],
    window: int,
    stride: int,
    overlap_threshold: float,
) -> tuple[list[str], list[str], list[int], dict[str, int]]:
    """Build span-AWARE (query, chunk, label) examples for TRAINING ONLY.

    records: the training records. window / stride: chunking config. overlap_threshold:
    fraction of a chunk a span must cover for the chunk to be labeled hallucinated.\\
    Returns (queries, chunks, labels, audit). Chunk boundaries are computed spans-blind;
    then per record: a hallucination=="no" record contributes all-negative chunks; a
    hallucination=="yes" record has its spans resolved and, if at least one resolves,
    its chunks are labeled by overlap. When no chunk clears ``overlap_threshold`` its
    highest-overlap chunk is promoted to positive, so a resolved positive doc never
    trains as all-negative. A yes-record whose spans resolve to ZERO intervals is skipped
    entirely (we cannot know which chunks are positive). ``audit`` reports the
    span-coverage counts for the metrics payload.
    """
    queries: list[str] = []
    chunks: list[str] = []
    labels: list[int] = []

    yes_records_total = 0
    yes_records_resolved = 0
    yes_records_zero_spans = 0
    yes_records_fallback_positive = 0

    for record in records:
        response_text = str(record.get(RESPONSE_FIELD, ""))
        query = str(record.get(QUERY_FIELD, ""))
        chunk_spans = chunk_response(response_text, window, stride)

        if record.get(LABEL_FIELD) != POSITIVE_LABEL:
            # a non-hallucinated document contributes only negative chunks
            chunk_labels = [0] * len(chunk_spans)
        else:
            yes_records_total += 1
            span_offsets = resolve_span_offsets(response_text, record.get(SPANS_FIELD, []) or [])
            # a known-positive doc with no resolvable spans is excluded rather than
            # mislabeled all-negative
            if not span_offsets:
                yes_records_zero_spans += 1
                continue
            yes_records_resolved += 1
            chunk_labels = assign_chunk_labels(chunk_spans, span_offsets, overlap_threshold)
            # a resolved positive doc must never train as all-negative: when no chunk
            # clears the overlap threshold (a short span straddling window boundaries),
            # promote its highest-overlap chunk so the hallucination signal is not inverted
            if not any(chunk_labels):
                fractions = chunk_coverage_fractions(chunk_spans, span_offsets)
                if fractions and max(fractions) > 0:
                    chunk_labels[fractions.index(max(fractions))] = 1
                    yes_records_fallback_positive += 1

        for span, label in zip(chunk_spans, chunk_labels):
            queries.append(query)
            chunks.append(chunk_text_of(response_text, span))
            labels.append(label)

    positive_chunks = sum(labels)
    audit = {
        "yes_records_total": yes_records_total,
        "yes_records_resolved": yes_records_resolved,
        "yes_records_zero_spans": yes_records_zero_spans,
        "yes_records_fallback_positive": yes_records_fallback_positive,
        "positive_chunks": positive_chunks,
        "negative_chunks": len(labels) - positive_chunks,
    }
    return queries, chunks, labels, audit