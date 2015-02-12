#!/usr/bin/env python

import sys
import codecs
import cPickle as pickle
import os
import HTMLParser
import phraseEmbedding as phraseEmbeddingModel
import wordvectors as wordvectors
import semanticTuning as semanticTuningModel

if len(sys.argv) < 5:
  print "./brae <n> <nu> <lambda> <alpha>"
  sys.exit(1)

# Command line arguments 
###########################################
# Dimensionality of input
n = int(sys.argv[1])
# The learning rate for SGD
learningRate = float(sys.argv[2])
# The regularization hyper-parameter
regularLambda = float(sys.argv[3])
# The interpolation parameter for reconstruction and
# semantic error
alpha = float(sys.argv[4])


# Hard-coded arguments (to be changed into CLI at some point) : TODO
###########################################
modelDir = "/export/a04/gkumar/code/custom/brae/model"
tgtWordVectorFile = "/export/a04/gkumar/code/custom/brae/tools/word2vec/afp_eng.vectors." + str(n) + ".sg.bin"
#tgtWordVectorFile = "/export/a04/gkumar/code/custom/brae/tools/word2vec/1.2.bin"
#tgtWordVectorFile = "/export/a04/gkumar/code/custom/brae/tools/word2vec/1.sg.new.bin"
srcWordVectorFile = "/export/a04/gkumar/code/custom/brae/tools/word2vec/afp_es.vectors." + str(n) + ".hs.bin"
phrasePairFile = codecs.open("/export/a04/gkumar/code/custom/brae/data/en.es.phrasepairs.tsv", encoding="utf8")
# The learning rate for stochastic gradient descent
# The number of iterations for which backprop should be performed
numIterations = 50


# Secondary static variables (initialized once, never changed)
###########################################
tgtWordVectors = wordvectors.WordVectors(tgtWordVectorFile)
print "--- Done reading target word embeddings"
#srcWordVectors = wordvectors.WordVectors(srcWordVectorFile)
#print "--- Done reading source word embeddings"
tgtPhrases = []
srcPhrases = []


# Helper stuff
###########################################
h = HTMLParser.HTMLParser()


# Read all phrase pairs and store
for phrasePair in phrasePairFile:
  # Phrase pair assumed to be in the order, source target
  phrases = phrasePair.strip().split("\t")
  srcPhrases.append(h.unescape(phrases[0]))
  tgtPhrases.append(h.unescape(phrases[1]))

phrasePairs = []
for src, tgt in zip(srcPhrases, tgtPhrases):
  if src != None and tgt != None:
    phrasePairs.append((src, tgt))


# For each language, build a phrase embedding (RAE) model and train it, first
tgtPhraseModel = phraseEmbeddingModel.PhraseEmbeddingModel(n, tgtWordVectors)
tgtPhraseModel.train(tgtPhrases, numIterations, regularLambda, learningRate)
print "--- Done pre-training the target phase embeddings (RAE)"
sys.exit(1)
srcPhraseModel = phraseEmbeddingModel.PhraseEmbeddingModel(n, srcWordVectors)
srcPhraseModel.train(srcPhrases, numIterations, regularLambda, learningRate)

print "--- Done pre-training the source phase embeddings (RAE)"

# Now tune these with the semantic tuning model
semanticModel = semanticTuningModel.semanticTuningModel(n, alpha, srcPhraseModel, tgtPhraseModel)
semanticModel.train(phrasePairs, numIterations, learningRate, regularLambda)

# Close all open files
phrasePairFile.close()
