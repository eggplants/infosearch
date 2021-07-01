# 4.x.y cannot work with models for Kyubyong/wordvectors
import json
from typing import List, Tuple, TypedDict

from gensim.models import Word2Vec  # gensim==3.8.3

# https://drive.google.com/open?id=0B0ZXk88koS2KVVNDS0lqdGNOSGM


class SimDict(TypedDict):
    term: str
    similarity: float


Sims = List[SimDict]


def get_synonyms(model: Word2Vec, input_word: str) -> Sims:
    """類似語の取得"""
    results = [{'term': word, 'similarity': sim}
               for word, sim in model.most_similar(input_word, topn=10)]
    return results


def calc_similarity(model: Word2Vec, t1: str, t2: str) -> float:
    """類似度計算"""
    return model.similarity(t1, t2)


def analogy(model: Word2Vec,
            relpair_x_y: Tuple[str, str], input_x: str) -> Sims:
    """アナロジー計算"""
    x, y = relpair_x_y
    results = [{"term": word, "similarity": sim}
               for word, sim in model.most_similar(positive=[y, input_x],
                                                   negative=[x], topn=10)]
    return results


def main() -> None:
    def p(d): return print(json.dumps(d, indent=4, ensure_ascii=False))

    model = Word2Vec.load('./data/ja.bin')

    syms = get_synonyms(model, 'アメリカ')
    res = analogy(model, ('フランス', 'パリ'), '日本')
    p(syms)
    p(res)


if __name__ == '__main__':
    main()
