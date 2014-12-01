#!/usr/bin/env python
# Takes a phrase and converts it into it's vector
# representation by using vector representations
# for each word in the phrase

# Author : Gaurav Kumar (Johns Hopkins University)

#TODO: Use a proper way of accepting arguments
#TODO: Commit on GIT
#TODO: Where do you get a vocabulary from ? : Maybe randomly generate?

import sys
import math
import numpy as np

# n needs to be sed empirically. Default to log_2(|V|)

vocabulary = []
# L is the embedding matrix
L = None
# We set n in a way where 2^n ~= |V|
n = math.ciel(math.log(len(vocabulary), 2))

# Create the embedding matrix here
for index, word in enumerate(vocabulary):
  # Create vector representation by getting the 
  # n-dimensional bit vector corresponding to the
  # index of this word in the vocabulary
  vec = pad(int2bin(index), n)
  vec = np.array(vec)
  if not L:
    L = vec
  else:
    np.hstack(L,vec)

# This can be made faster by using a dict for the vocab
phraseRaw = []
phrase = []
for word in phraseRaw:
  index = vocabulary.index(word)
  if not index:
    raise Exception("Word " + word + " does not exist in the vocabulary")
  e_i = np.zeroes(len(V), 1)
  e_i[index] = 1
  phrase.append(np.dot(L, e_i))


def autoencoder(c1, c2):
  global n
  W = np.zeroes(n, 2*n)
  b = np.zeroes(n, 1)
  c1_c2 = np.concat(c1, c2)
  # f is assumed to be tanh() here but any element wise activation function
  # can be used
  # TODO: Generalize the incorporation of the element wise activation function
  pass


