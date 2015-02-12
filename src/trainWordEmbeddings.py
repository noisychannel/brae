#!/usr/bin/env python

import sys
from gensim.models import Word2Vec

class MySentences(object):
  def __init__(self, fname):
    self.fname = fname

  def __iter__(self):
    for line in open(self.fname):
      yield line.strip().split()


if len(sys.argv) < 4:
  print "Usage : ./trainWordEmbeddings <input> <output> <dim>"

print sys.argv

inputFile = sys.argv[1]
outputFile = sys.argv[2]
n = int(sys.argv[3])

sentences = MySentences(inputFile)
model = Word2Vec(sentences, size=n, window=8, iter=15, sample=1e-5, workers=1)

model.save(outputFile)
