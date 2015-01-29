#!/usr/bin/env python

import numpy as np
import codecs
import sys
import HTMLParser
from multiprocessing.dummy import Pool as ThreadPool
import autoencoder as autoencoder
import wordvectors as wordvectors

if len(sys.argv) < 3:
  print "./phraseEmbedding <n> <lambda>"
  sys.exit(1)

# Dimensionality of the input vector
n = int(sys.argv[1])
regularLambda = float(sys.argv[2])
learningRate = 0.01
numIterations = 100
a = autoencoder.AutoEncoder(n)
w = wordvectors.WordVectors("/export/a04/gkumar/code/custom/brae/tools/word2vec/afp_eng.vectors." + str(n) + ".bin")
h = HTMLParser.HTMLParser()
backPropTrainingData = []
totalReconstructionError = 0

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
  minErrorParent_z = None
  minErrorC1C2Prime = None
  minErrorC1C2Prime_z = None
  errors = []
  for i in range(len(phraseVectors) - 1):
    # Check reconstruction error for (i, i+1)
    p_z, p = autoEncoder.computeParentVector(phraseVectors[i], phraseVectors[i+1])
    c1c2Prime_z, c1c2Prime, reconError = autoEncoder.computeSingleReconstructionError(p, phraseVectors[i], phraseVectors[i+1])
    if reconError < minError:
      minError = reconError
      minErrorIndex = i
      minErrorParent = p
      minErrorParent_z = p_z
      minErrorC1C2Prime = c1c2Prime
      minErrorC1C2Prime_z = c1c2Prime_z

  #print "Chose indices (" + str(i) + ", " + str(i+1) + ") for combination"
  #print "The minimum error was " + str(minError)

  return minErrorIndex, minError, minErrorParent, minErrorParent_z, minErrorC1C2Prime, minErrorC1C2Prime_z


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
    # A few values while creation of the best tree are cached for use in backprop
    combinedIndex, combinedError, combinedParent, combinedParent_z, \
        reconstructedC1C2, reconstructedC1C2_z = computeBestBinaryTree(phraseVectors, a)
    # Cache values for backprop training and calculating gradients
    backPropTrainingData.append((np.vstack((phraseVectors[combinedIndex], phraseVectors[combinedIndex+1])), \
        combinedParent, combinedParent_z, reconstructedC1C2, reconstructedC1C2_z))
    # Record total reconstruction error
    global totalReconstructionError
    totalReconstructionError = totalReconstructionError + combinedError**2
    # Change the phrase vector to include the parent
    phraseVectors[combinedIndex] = combinedParent
    del phraseVectors[combinedIndex+1]
    combinationPattern[combinedIndex] = (combinationPattern[combinedIndex], combinationPattern[combinedIndex+1])
    del combinationPattern[combinedIndex+1]



for i in range(numIterations):
  for phrasePair in phrases:
    processPhrasePair(phrasePair)
  cost = (1./len(backPropTrainingData)) * totalReconstructionError + \
      regularLambda * (np.sum(a.paramMatrix * a.paramMatrix) + \
      np.sum(a.reconParamMatrix * a.reconParamMatrix))
  print "Cost (" + str(i) + ") = " + str(cost)
  #print "================================="
  #print "Iteration : " + str(i+1)
  dJ_dW1, dJ_dW2, dJ_db2, dJ_db3 =  a.getGradients(backPropTrainingData, True, 0.1)
  a.paramMatrix = a.paramMatrix - learningRate * dJ_dW1
  a.reconParamMatrix = a.reconParamMatrix - learningRate * dJ_dW2
  a.bias = a.bias - learningRate * dJ_db2
  a.reconBias = a.reconBias - learningRate * dJ_db3
  totalReconstructionError = 0.
