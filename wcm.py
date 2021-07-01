#!/usr/bin/python

from pprint import pprint
from typing import Dict, List, Tuple

import MeCab
import numpy as np


def preprocess(words: List[str]) -> Tuple[
        np.array, Dict[str, int], Dict[str, int]]:
    """"単語の数値コード化"""
    word_to_id, id_to_word = {}, {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus: List[str], vocab_size: int,
                     window_size: int = 1) -> np.ndarray:
    """単語間共起行列の作成"""
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def cos_similarity(x, y, eps: float = 1e-8) -> float:
    """ベクトル間コサイン類似度の計算"""
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


def most_similar(query: str,
                 word_to_id: Dict[str, int], id_to_word: Dict[int, str],
                 word_matrix: np.ndarray, top: int) -> List[Tuple[str, float]]:
    """類似単語のランキング表示"""
    # クエリ取り出し
    if query not in word_to_id:
        raise ValueError('%s is not found' % query)
    else:
        query_id = word_to_id[query]
        query_vec = word_matrix[query_id]

    # コサイン類似度の算出
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # コサイン類似度の降順に出力
    res = []
    for i in (-similarity).argsort():
        if id_to_word[i] != query:
            res.append((id_to_word[i], similarity[i],))
            if len(res) >= top:
                break
    return res


def ppmi(c: np.ndarray, vec_size: int = 50, eps: float = 1e-8) -> np.ndarray:
    """単語間相互情報量の計算"""
    m = np.zeros_like(c, dtype=np.float32)
    n = np.sum(c)
    s = np.sum(c, axis=0)
    print("all=", c.shape)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            print("cal=", (i, j), end='\r')
            pmi = np.log2(c[i, j] * n / (s[j]*s[i]) + eps)
            m[i, j] = max(0, pmi)

    # 次元削減
    u, _, _ = np.linalg.svd(m)
    m = u[:, :vec_size]
    return m


def main() -> None:
    """メイン"""
    m = MeCab.Tagger('-Ochasen')
    words = []
    for text in open('gin_yoru.txt', 'r'):
        lines = m.parse(text).split('\n')
        matches = [line.split('\t')[2]
                   for line in lines if line.count('\t') == 5]
        words += matches

    corpus, word_to_id, id_to_word = preprocess(words)
    c = create_co_matrix(corpus, len(word_to_id))
    if input("type 'y' to use ppmi (too slow)> ") == 'y':
        c = ppmi(c)

    def simq(q: str) -> None:
        return pprint(
            most_similar(
                q, word_to_id, id_to_word, c, top=5))

    while True:
        q = input('q> ')
        if q in ['exit', 'close', 'quit', 'q']:
            break
        try:
            simq(q)
        except Exception as e:
            print("Failed!:", type(e))
            print(e)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        exit(0)
