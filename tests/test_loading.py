import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from rank_bm25 import BM25Okapi, BM25L, BM25Plus, BM25OkapiWithIDs
import re


corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?",
]
tokenized_corpus = [doc.split(" ") for doc in corpus]

algs = [
    BM25Okapi(tokenized_corpus),
    BM25L(tokenized_corpus),
    BM25Plus(tokenized_corpus)
]

corpus_id = [(text, f"id{i}") for i, text in enumerate(corpus)]
tokenized_corpus_id = [(t[0].split(" "), t[1]) for t in corpus_id]

algs_id = [BM25OkapiWithIDs(tokenized_corpus_id)]

query = "Hello"

def test_corpus_loading():
    for alg in algs:
        assert alg.corpus_size == 3
        assert alg.avgdl == 5
        assert alg.doc_len == [4, 6, 5]
    for alg in algs_id:
        assert alg.corpus_size == 3
        assert alg.avgdl == 5
        assert list(alg.doc_len.values()) == [4, 6, 5]

def tokenizer(doc):
    return doc.split(" ")

def test_tokenizer():
    bm25 = BM25Okapi(corpus, tokenizer=tokenizer)
    assert bm25.corpus_size == 3
    assert bm25.avgdl == 5
    assert bm25.doc_len == [4, 6, 5]
    bm25_ids = BM25OkapiWithIDs(corpus_id, tokenizer=tokenizer)
    assert bm25_ids.corpus_size == 3
    assert bm25_ids.avgdl == 5
    assert list(bm25_ids.doc_len.values()) == [4, 6, 5]

def test_search_equivalence():
    score = algs[0].get_scores(query)
    score_id = algs_id[0].get_scores(query)
    only_score = [t[1] for t in score_id]
    assert (score == only_score).all()
