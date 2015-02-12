#!/usr/bin/env python

import numpy as np
import autoencoder as autoencoder
from scipy.optimize import minimize
import sys
import copy

class PhraseEmbeddingModel:

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

    #self.a.paramMatrix = np.array([[-0.66901207, -0.18282628, -0.05600838, -0.93896959],
                                  #[-0.10363659, -0.53908909, 0.24603079, 0.50378757]])
    #self.a.reconParamMatrix = np.array([[-0.22829916, 0.92183385],
                                        #[-0.4777748, 0.47174915],
                                        #[ 0.22207423, -0.22005313],
                                        #[-0.53727245, -0.70092649]])
    self.totalReconstructionError = 0
    self.backPropTrainingData = []
    self.finalLayerBackPropData = []


  # Given a source string 
  def findBestNodesToCombine(self, phraseVectors, paramMatrix, bias, reconParamMatrix, reconBias):
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
      p_z, p = self.a.computeParentVector(phraseVectors[i], phraseVectors[i+1], paramMatrix, bias)
      c1c2Prime_z, c1c2Prime, reconError = self.a.computeSingleReconstructionError(p, phraseVectors[i], phraseVectors[i+1], reconParamMatrix, reconBias)
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


  def computeOptimalBinaryTree(self, phrase, paramMatrix, bias, reconParamMatrix, reconBias):
    """
    Given a phrase and the embeddings of each word in the phrase, this constructs
    an 'optimal' binary tree for phrase embedding. This is based on the greedy
    construction of Socher et al., 2010. Pairs of leaves are combined until
    we are left with one node

    Parameters:
    phrase : A phrase for which a binary tree for phrase embedding needs to be created

    Returns
      The result of the phrase embedding. This is the final parent with the same dimensionality
      as the word vectors
    """

    #TODO: Why is this done everytime? the wrapper should send back phrase vectors
    words = phrase.split()
    words = [word for word in words if self.w[word] != None]
    phraseVectors = []
    treeConstructed = False

    for word in words:
      phraseVectors.append(np.reshape(self.w[word], (self.n,1)))

    # Used to track combination pattern
    combinationPattern = list(words)

    while len(phraseVectors) > 1:
      # A few values (while creation of the best tree) are cached for use in backprop
      combinedIndex, combinedError, combinedParent, combinedParent_z, \
          reconstructedC1C2, reconstructedC1C2_z = self.findBestNodesToCombine(phraseVectors, paramMatrix, bias, reconParamMatrix, reconBias)
      # Cache values for backprop training and calculating gradients
      self.backPropTrainingData.append((np.vstack((phraseVectors[combinedIndex], phraseVectors[combinedIndex+1])), \
          combinedParent, combinedParent_z, reconstructedC1C2, reconstructedC1C2_z))
###########
      #if len(phraseVectors) == 2:
        ## Keep track of this for semantic tuning
        #self.finalLayerBackPropData.append((np.vstack((phraseVectors[combinedIndex], phraseVectors[combinedIndex+1])), \
          #combinedParent, combinedParent_z, reconstructedC1C2, reconstructedC1C2_z))
###########

      # Record total reconstruction error
      self.totalReconstructionError = self.totalReconstructionError + 0.5 * combinedError**2
      # Change the phrase vector to include the parent
      phraseVectors[combinedIndex] = combinedParent
      del phraseVectors[combinedIndex+1]
      combinationPattern[combinedIndex] = (combinationPattern[combinedIndex], combinationPattern[combinedIndex+1])
      del combinationPattern[combinedIndex+1]
    #print combinationPattern
##########
      #treeConstructed = True

    #if len(phraseVectors) > 1:
      #raise Exception("Phrase vector contains multiple elements. Tree building did not succeed")
    #else:
      #if treeConstructed:
        ## The opposite case is when we encouter a phrase that only contained OOVs wrt the 
        ## word embeddings or we encounter a single word phrase
        #return phraseVectors[0]
      #else:
        #self.finalLayerBackPropData.append((None, None, None, None, None))
        #return None
##########

  def train(self, phrases, numIterations, regularLambda, learningRate):
    self.phrases = phrases
    self.regularLambda = regularLambda
    self.SGD(numIterations, learningRate)
    #self.LBFGS(numIterations)


  def LBFGS(self, numIterations):
    initWeights = self.foldWeights(self.a.paramMatrix, self.a.reconParamMatrix, self.a.bias, self.a.reconBias)
    # Get gradients once to fill in the backprop data which is required to calculate cost
    self.getGradients(initWeights)
    res = minimize(self.computeCost, initWeights, method='L-BFGS-B', jac=self.getGradients, tol=0.00001, options={'disp': True, 'maxiter':numIterations})
    print res.message


  def SGD(self, numIterations, learningRate):
    for i in range(numIterations):
      # Minimize
      # SGD : Update parameters
      weights = self.foldWeights(self.a.paramMatrix, self.a.reconParamMatrix, self.a.bias, self.a.reconBias)
      dJ_dW1, dJ_dW2, dJ_db2, dJ_db3 = self.unfoldWeights(self.getGradients(weights))
      self.gradient_checking(weights, 1e-15)

      self.computeCost(weights)
      self.a.paramMatrix = self.a.paramMatrix - learningRate * dJ_dW1
      self.a.reconParamMatrix = self.a.reconParamMatrix - learningRate * dJ_dW2
      self.a.bias = self.a.bias - learningRate * dJ_db2
      self.a.reconBias = self.a.reconBias - learningRate * dJ_db3


  def getGradients(self, weights):
    """
    Trains this phrase embedding model with backpropagation

    Parameters:
    phrases: A list of phrases for training
    numIterations : The number of iterations backprop should be performed for
    regularLambda : The regularization hyper-parameter
    learningRate : The learning rate for SGD
    """

    # Reset reconstruction error for the next iteration
    self.totalReconstructionError = 0.
    self.backPropTrainingData = []

    paramMatrix, reconParamMatrix, bias, reconBias = self.unfoldWeights(weights)

    for phrase in self.phrases:
      #self.computeOptimalBinaryTree(phrase, self.a.paramMatrix, self.a.bias, self.a.reconParamMatrix, self.a.reconBias)
      self.computeOptimalBinaryTree(phrase, paramMatrix, bias, reconParamMatrix, reconBias)

    # Get partial derivatives
    # TODO: Enable regularization later
    # TODO: When regularization is enabled, send bias and reconBias as well
    dJ_dW1, dJ_dW2, dJ_db2, dJ_db3 =  self.a.getGradients(self.backPropTrainingData, paramMatrix, reconParamMatrix, False, self.regularLambda)

############
      #self.finalLayerBackPropData = []
    return self.foldWeights(dJ_dW1, dJ_dW2, dJ_db2, dJ_db3)


  def computeCost(self, weights):
    #TODO: Replace withj regularized cost
    #cost = (1./len(self.backPropTrainingData)) * self.totalReconstructionError + \
        #self.regularLambda * (np.sum(self.a.paramMatrix ** 2) + np.sum(self.a.reconParamMatrix ** 2 ))
    cost = (1./len(self.backPropTrainingData)) * self.totalReconstructionError
    print "Cost = " + str(cost)
    return cost


  def foldWeights(self, paramMatrix, reconParamMatrix, bias, reconBias):
      # SGD : Update parameters
    weights = np.reshape(paramMatrix, (1, np.shape(paramMatrix)[0] * np.shape(paramMatrix)[1]))[0]
    weights = np.hstack((weights, np.reshape(reconParamMatrix, (1, np.shape(reconParamMatrix)[0] * np.shape(reconParamMatrix)[1]))[0]))
    weights = np.hstack((weights, np.reshape(bias, (1, np.shape(bias)[0]))[0]))
    weights = np.hstack((weights, np.reshape(reconBias, (1, np.shape(reconBias)[0]))[0]))
    return weights


  def unfoldWeights(self, weights):
    # TODO: Too many redundant actions, re-write
    paramMatrixShape = np.shape(self.a.paramMatrix)
    paramMatrixSize = paramMatrixShape[0] * paramMatrixShape[1]
    tmp = weights[0:paramMatrixSize]
    tmpRemaining = weights[paramMatrixSize:]
    paramMatrix = np.reshape(tmp, paramMatrixShape)

    reconParamMatrixShape = np.shape(self.a.reconParamMatrix)
    reconParamMatrixSize = reconParamMatrixShape[0] * reconParamMatrixShape[1]
    tmp = tmpRemaining[0:reconParamMatrixSize]
    tmpRemaining = tmpRemaining[reconParamMatrixSize:]
    reconParamMatrix = np.reshape(tmp, reconParamMatrixShape)

    biasShape = np.shape(self.a.bias)
    biasSize = biasShape[0]
    tmp = tmpRemaining[0:biasSize]
    tmpRemaining = tmpRemaining[biasSize:]
    bias = np.reshape(tmp, biasShape)

    reconBias = np.reshape(tmpRemaining, np.shape(self.a.reconBias))

    return paramMatrix, reconParamMatrix, bias, reconBias

  def gradient_checking(self, theta, eps):
    f_approx = np.zeros(np.shape(theta))
    for i, t in enumerate(theta):
      theta_plus = copy.deepcopy(theta)
      theta_minus = copy.deepcopy(theta)
      theta_plus[i] = theta[i] + eps
      theta_minus[i] = theta[i] - eps

      self.backPropTrainingData = []
      self.totalReconstructionError = 0.
      for phrase in self.phrases:
        #self.computeOptimalBinaryTree(phrase, self.a.paramMatrix, self.a.bias, self.a.reconParamMatrix, self.a.reconBias)
        self.computeOptimalBinaryTree(phrase, paramMatrix, bias, reconParamMatrix, reconBias)
      cost_theta_plus = self.computeCost(theta_plus)

      self.backPropTrainingData = []
      self.totalReconstructionError = 0.
      for phrase in self.phrases:
        #self.computeOptimalBinaryTree(phrase, self.a.paramMatrix, self.a.bias, self.a.reconParamMatrix, self.a.reconBias)
        self.computeOptimalBinaryTree(phrase, paramMatrix, bias, reconParamMatrix, reconBias)
      cost_theta_minus = self.computeCost(theta_minus)

      f_approx[i] = (cost_function(theta_plus, data) - cost_function(theta_minus, data)) / (2 * eps)
    return f_approx
