#!/usr/bin/env python

import numpy as np
import autoencoder as autoencoder
import wordvectors as wordvectors

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
    reconError = autoEncoder.computeReconstructionError(p, phraseVectors[i], phraseVectors[i+1])
    if reconError < minError:
      minError = reconError
      minErrorIndex = i
      minErrorParent = p

  print "Chose indices (" + str(i) + ", " + str(i+1) + ") for combination"
  print "The minimum error was " + str(minError)

  return minErrorIndex, minError, minErrorParent

a = autoencoder.AutoEncoder()
phraseVectors = [np.random.rand(100,1),
                  np.random.rand(100,1),
                  np.random.rand(100,1),
                  np.random.rand(100,1),
                  np.random.rand(100,1)]

# Used to track combination pattern
phraseIndices = [i for i in range(len(phraseVectors))]

while len(phraseVectors) > 1:
  combinedIndex, combinedError, combinedParent = computeBestBinaryTree(phraseVectors, a)
  phraseVectors[combinedIndex] = combinedParent
  del phraseVectors[combinedIndex+1]
  phraseIndices[combinedIndex] = (phraseIndices[combinedIndex], phraseIndices[combinedIndex+1])
  del phraseIndices[combinedIndex+1]

print phraseIndices

w = wordvectors.WordVectors("/export/a04/gkumar/code/custom/brae/tools/word2vec/vectors.bin")
print w.vocab
print w.vectors
