import gzip
from gensim.models import Word2Vec
import gensim
import logging
import os

model= Word2Vec.load("w2v.model")



w1 = "dirty"
print("Most similar to {0}".format(w1), model.wv.most_similar(positive=w1))

# look up top 6 words similar to 'polite'
w1 = ["polite"]
print(
    "Most similar to {0}".format(w1),
    model.wv.most_similar(
        positive=w1,
        topn=6))

# look up top 6 words similar to 'france'
w1 = ["france"]
print(
    "Most similar to {0}".format(w1),
    model.wv.most_similar(
        positive=w1,
        topn=6))

# look up top 6 words similar to 'shocked'
w1 = ["shocked"]
print(
    "Most similar to {0}".format(w1),
    model.wv.most_similar(
        positive=w1,
        topn=6))

# look up top 6 words similar to 'shocked'
w1 = ["beautiful"]
print(
    "Most similar to {0}".format(w1),
    model.wv.most_similar(
        positive=w1,
        topn=6))

# get everything related to stuff on the bed
w1 = ["bed", 'sheet', 'pillow']
w2 = ['couch']
print(
    "Most similar to {0}".format(w1),
    model.wv.most_similar(
        positive=w1,
        negative=w2,
        topn=10))

# similarity between two different words
print("Similarity between 'dirty' and 'smelly'",
      model.wv.similarity(w1="dirty", w2="smelly"))

# similarity between two identical words
print("Similarity between 'dirty' and 'dirty'",
      model.wv.similarity(w1="dirty", w2="dirty"))

# similarity between two unrelated words
print("Similarity between 'dirty' and 'clean'",
      model.wv.similarity(w1="dirty", w2="clean"))
