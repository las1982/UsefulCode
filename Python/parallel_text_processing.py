import multiprocessing
from itertools import chain
from multiprocessing.pool import Pool

from nltk import WordNetLemmatizer
from tqdm import tqdm

import re
from nltk.corpus import stopwords
import time
from sklearn.datasets import fetch_20newsgroups

global_stop_words = stopwords.words('english')

train = fetch_20newsgroups(subset="all")
text = " ".join(train.data)
lemmatizer = WordNetLemmatizer()
num_cores = multiprocessing.cpu_count()


def get_batches(tokens, n):
    batches = []
    idx = len(tokens) // 8
    idxs = []
    for i in range(n):
        start = i * idx
        end = (i + 1) * idx
        if i + 1 == n:
            end = len(tokens)
        idxs.append((start, end))
        batches.append(tokens[start: end])
    assert len(set(tokens)) == len(set(chain.from_iterable(batches)))
    return batches


def get_words(tokens):
    words = []
    for word in tqdm(tokens):
        word = lemmatizer.lemmatize(word)
        if word not in global_stop_words:
            words.append(word)
    return words


def get_words_par(tokens, cores):
    with Pool(cores) as pool:
        df = pool.map(get_words, tokens)
        pool.close()
        pool.join()
        return df


tokens = re.split("\W+", text.lower())
start = time.time()
print(len(get_words(tokens)))
print("taken", time.time() - start)

batches = get_batches(tokens, n=num_cores)
start = time.time()
print(len(list(chain.from_iterable(get_words_par(batches, num_cores)))))
print("taken", time.time() - start)
