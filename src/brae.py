#!/usr/bin/env python

import sys
import codecs
import HTMLParser
import phraseEmbedding as phraseEmbeddingModel
import wordvectors as wordvectors

if len(sys.argv) < 3:
  print "./brae <n> <lambda>"
  sys.exit(1)

# Command line arguments 
###########################################
# Dimensionality of input
n = int(sys.argv[1])
# The regularization hyper-parameter
regularLambda = float(sys.argv[2])


# Hard-coded arguments (to be changed into CLI at some point) : TODO
###########################################
tgtWordVectorFile = "/export/a04/gkumar/code/custom/brae/tools/word2vec/afp_eng.vectors." + str(n) + ".bin"
srcWordVectorFile = None
phrasePairFile = codecs.open("/export/a04/gkumar/code/custom/brae/data/en.es.phrasepairs.tsv", encoding="utf8")
# The learning rate for stochastic gradient descent
learningRate = 0.01
# The number of iterations for which backprop should be performed
numIterations = 100


# Secondary static variables (initialized once, never changed)
###########################################
tgtWordVectors = wordvectors.WordVectors(tgtWordVectorFile)
srcWordVectors = None
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


# For each language, build a phrase embedding (RAE) model and train it, first
tgtPhraseModel = phraseEmbeddingModel.PhraseEmbeddingModel(n, tgtWordVectors)
tgtPhraseModel.train(tgtPhrases, numIterations, regularLambda, learningRate)

# Close all open files
phrasePairFile.close()
