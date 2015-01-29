#!/usr/bin/env python

import numpy as np
import autoencoder as autoencoder

class PhraseEmbeddingModel:
  # Dimensionality of the input vector
  n = 50
  # A Word vector object that holds pre-trained word embeddings for the vocabulary
  w = None
  # Training data for backprop
  backPropTrainingData = []
  # Captures total reconstruction error per iteration of backprop
  totalReconstructionError = 0
  # An autoencoder that is shared by this phrase embedding model (RAE)
  a = None


  def __init__(self, n, w):
    """
    This initializes a phrase embedding model which is based on a
    recursive autoencoder (RAE). This model can be trained with 
    backpropagation and stochastic gradient descent

    Parameters: 
    n : The dimensionality of the input
    w : A word vector object that contains pre-trained word embeddings
    """
    self.n = n
    self.w = w
    self.a = autoencoder.AutoEncoder(self.n)


  # Given a source string 
  def findBestNodesToCombine(self, phraseVectors):
    """
    Given a phrase and the embeddings of each word in the phrase, this
    finds the best leaves to combine based on a greedy metric
    (Adjacent leaves that have the least reconstruction error
    with respect to an autoencoder)

    Parameters:
    phraseVectors : A list of word embeddings which make up the phrase

    Returns : 
    Details about which children were combined and in which order, what the 
    parent was and the reconstruction error. These are returned so that they 
    can be cached. This makes calculation of partial derivatives for 
    backpropagation faster.
    """
    minError = float("inf")
    minErrorIndex = None
    minErrorParent = None
    minErrorParent_z = None
    minErrorC1C2Prime = None
    minErrorC1C2Prime_z = None
    errors = []
    for i in range(len(phraseVectors) - 1):
      # Check reconstruction error for (i, i+1)
      p_z, p = self.a.computeParentVector(phraseVectors[i], phraseVectors[i+1])
      c1c2Prime_z, c1c2Prime, reconError = self.a.computeSingleReconstructionError(p, phraseVectors[i], phraseVectors[i+1])
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


  def computeOptimalBinaryTree(self, phrase):
    """
    Given a phrase and the embeddings of each word in the phrase, this constructs
    an 'optimal' binary tree for phrase embedding. This is based on the greedy
    construction of Socher et al., 2010. Pairs of leaves are combined until
    we are left with one node

    Parameters:
    phrase : A phrase for which a binary tree for phrase embedding needs to be created
    """

    words = phrase.split()
    words = [word for word in words if self.w[word] != None]
    phraseVectors = []

    for word in words:
      phraseVectors.append(np.reshape(self.w[word], (self.n,1)))

    # Used to track combination pattern
    combinationPattern = list(words)

    while len(phraseVectors) > 1:
      # A few values (while creation of the best tree) are cached for use in backprop
      combinedIndex, combinedError, combinedParent, combinedParent_z, \
          reconstructedC1C2, reconstructedC1C2_z = self.findBestNodesToCombine(phraseVectors)
      # Cache values for backprop training and calculating gradients
      self.backPropTrainingData.append((np.vstack((phraseVectors[combinedIndex], phraseVectors[combinedIndex+1])), \
          combinedParent, combinedParent_z, reconstructedC1C2, reconstructedC1C2_z))
      # Record total reconstruction error
      self.totalReconstructionError = self.totalReconstructionError + combinedError**2
      # Change the phrase vector to include the parent
      phraseVectors[combinedIndex] = combinedParent
      del phraseVectors[combinedIndex+1]
      combinationPattern[combinedIndex] = (combinationPattern[combinedIndex], combinationPattern[combinedIndex+1])
      del combinationPattern[combinedIndex+1]


  def train(self, phrases, numIterations, regularLambda, learningRate):
    """
    Trains this phrase embedding model with backpropagation

    Parameters:
    phrases: A list of phrases for training
    numIterations : The number of iterations backprop should be performed for
    regularLambda : The regularization hyper-parameter
    learningRate : The learning rate for SGD
    """

    for i in range(numIterations):
      for phrase in phrases:
        self.computeOptimalBinaryTree(phrase)

      cost = (1./len(self.backPropTrainingData)) * self.totalReconstructionError + \
          regularLambda * (np.sum(self.a.paramMatrix * self.a.paramMatrix) + \
          np.sum(self.a.reconParamMatrix * self.a.reconParamMatrix))
      print "Cost (" + str(i) + ") = " + str(cost)

      # Get partial derivatives
      dJ_dW1, dJ_dW2, dJ_db2, dJ_db3 =  self.a.getGradients(self.backPropTrainingData, True, regularLambda)

      # SGD : Update parameters
      self.a.paramMatrix = self.a.paramMatrix - learningRate * dJ_dW1
      self.a.reconParamMatrix = self.a.reconParamMatrix - learningRate * dJ_dW2
      self.a.bias = self.a.bias - learningRate * dJ_db2
      self.a.reconBias = self.a.reconBias - learningRate * dJ_db3

      # Reset reconstruction error for the next iteration
      self.totalReconstructionError = 0.
