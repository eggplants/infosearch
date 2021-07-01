import random
from typing import Any, Dict, List, Literal, Tuple, Union

import nltk

try:
    nltk.data.find('corpora/movie_reviews')
except LookupError:
    nltk.download('movie_reviews')

from nltk.corpus import movie_reviews as mr

MR_ALL_WORDS = nltk.FreqDist(w.lower()
                             for w in mr.words())
WORD_FEATURES = list(MR_ALL_WORDS.keys())[:2000]


def doc_features(doc) -> Dict[str, bool]:
    """素性抽出器"""
    doc_words = set(doc)
    features = {}
    for w in WORD_FEATURES:
        features['contains(%s)' % w] = w in doc_words
    return features


AccResults = Tuple[Union[Any, float, Literal[0]],
                   nltk.NaiveBayesClassifier]


def cal_accuracy(docs: List[Tuple[list, str]]) -> AccResults:
    """正解率計算"""
    feat_sets = [(doc_features(d), c) for d, c in docs]
    train_set, time_set = feat_sets[100:], feat_sets[:100]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    return (nltk.classify.accuracy(classifier, time_set), classifier,)


def main() -> None:
    docs = [(list(mr.words(id_)), category,)
            for category in mr.categories()
            for id_ in mr.fileids(category)]
    random.shuffle(docs)
    acc, clsfier = cal_accuracy(docs)
    print("Accuracy:", acc)
    clsfier.show_most_informative_features(5)


if __name__ == '__main__':
    main()
