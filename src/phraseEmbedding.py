#!/usr/bin/env python

import numpy as np
import codecs
import HTMLParser
from multiprocessing.dummy import Pool as ThreadPool
import autoencoder as autoencoder
import wordvectors as wordvectors

n = 50
a = autoencoder.AutoEncoder(n)
w = wordvectors.WordVectors("/export/a04/gkumar/code/custom/brae/tools/word2vec/afp_eng.vectors.50.bin")
h = HTMLParser.HTMLParser()
backPropTrainingData = []

ppFile = codecs.open("/export/a04/gkumar/code/custom/brae/data/en.es.phrasepairs.tsv", encoding="utf8")
phrases = []
for phrasePair in ppFile:
  phrases.append(phrasePair)
ppFile.close()

# Given a source string 
def computeBestBinaryTree(phraseVectors, autoEncoder):
  # This is a greedy construction (Socher et al, 2010)
  minError = float("inf")
  minErrorIndex = None
  minErrorParent = None
  errors = []
  for i in range(len(phraseVectors) - 1):
    # Check reconstruction error for (i, i+1)
    p = autoEncoder.computeParentVector(phraseVectors[i], phraseVectors[i+1])
    reconError = autoEncoder.computeSingleReconstructionError(p, phraseVectors[i], phraseVectors[i+1])
    if reconError < minError:
      minError = reconError
      minErrorIndex = i
      minErrorParent = p

  #print "Chose indices (" + str(i) + ", " + str(i+1) + ") for combination"
  #print "The minimum error was " + str(minError)

  return minErrorIndex, minError, minErrorParent


def processPhrasePair(phrasePair):
  phrasePair = phrasePair.strip().split("\t")

  tgtPhrase = h.unescape(phrasePair[1])
  words = tgtPhrase.split()

  words = [word for word in words if w[word] != None]

  phraseVectors = []
  for word in words:
    phraseVectors.append(np.reshape(w[word], (n,1)))

  # Used to track combination pattern
  combinationPattern = list(words)

  while len(phraseVectors) > 1:
    combinedIndex, combinedError, combinedParent = computeBestBinaryTree(phraseVectors, a)
    backPropTrainingData.append((phraseVectors[combinedIndex], phraseVectors[combinedIndex+1], combinedParent, combinedError))
    phraseVectors[combinedIndex] = combinedParent
    del phraseVectors[combinedIndex+1]
    combinationPattern[combinedIndex] = (combinationPattern[combinedIndex], combinationPattern[combinedIndex+1])
    del combinationPattern[combinedIndex+1]

  print combinationPattern
  print len(backPropTrainingData)

for phrasePair in phrases:
  processPhrasePair(phrasePair)

print len(backPropTrainingData)
