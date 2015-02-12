import numpy as np
import sys
import activationfunction as activationfunction

class semanticTuningModel:

  # The dimensionality of the input
  n = 50
  # The weight matrix to transform the source embedding
  W_s = None
  # The bias vector to transform the source embedding
  b_s = None
  # The weight matrix to transform the target embedding
  W_t = None
  # The bias vector to transform the target embedding
  b_t = None
  # An activation function for the transforms
  a = None
  # Source phrase embedding model
  srcPhraseModel = None
  # Target phrase embedding model
  tgtPhraseModel = None
  # The interpolation parameter
  alpha = 0.1


  def __init__(self, n, alpha, srcPhraseModel, tgtPhraseModel):
    """
    Initializes a model to tune the phrase embeddings
    in the BRAE model using the principle that phrase
    embeddings for gold standard phrase pairs should be
    close together for all phrase pairs given some transformation
    The goal is to find that transform

    Parameters:
    n : The dimensionality of the input
    """

    self.n = n
    self.W_s = np.random.rand(self.n, self.n)
    self.W_t = np.random.rand(self.n, self.n)
    self.b_s = np.random.rand(self.n, 1)
    self.b_t = np.random.rand(self.n, 1)
    self.a = activationfunction.ActivationFunction("tanh")
    self.alpha = alpha
    self.srcPhraseModel = srcPhraseModel
    self.tgtPhraseModel = tgtPhraseModel


  def calculateSumOfGradients(self, phrasePairs, regularization = True, l=0.1):
    accumulator_W_s = np.zeros((self.n, self.n))
    accumulator_W_t = np.zeros((self.n, self.n))
    accumulator_b_s = np.zeros((self.n, 1))
    accumulator_b_t = np.zeros((self.n, 1))
    srcDeltas = []
    tgtDeltas = []
    nonNonePhrasePairs = 0
    srcTotalError = 0
    tgtTotalError = 0

    for phrasePair in phrasePairs:
      if phrasePair[0] is None or phrasePair[1] is None:
        srcDeltas.append(None)
        tgtDeltas.append(None)
        continue
      srcPhrase = np.asarray(phrasePair[0])
      tgtPhrase = np.asarray(phrasePair[1])
      srcDelta, srcError = self.calculateDeltas(srcPhrase, tgtPhrase, self.W_s, self.b_s)
      tgtDelta, tgtError = self.calculateDeltas(tgtPhrase, srcPhrase, self.W_t, self.b_t)
      srcTotalError = srcTotalError + srcError**2
      tgtTotalError = tgtTotalError + tgtError**2
      srcDeltas.append(srcDelta)
      tgtDeltas.append(tgtDelta)

      # Add these values to the accumulators
      accumulator_W_s = accumulator_W_s + np.dot(srcDelta, np.transpose(srcPhrase))
      accumulator_W_t = accumulator_W_t + np.dot(tgtDelta, np.transpose(tgtPhrase))
      accumulator_b_s = accumulator_b_s + srcDelta
      accumulator_b_t = accumulator_b_t + tgtDelta

      nonNonePhrasePairs = nonNonePhrasePairs + 1

    m = len(phrasePairs)
    dJ_dW_s = (1./m) * accumulator_W_s
    dJ_dW_t = (1./m) * accumulator_W_t
    if regularization:
      dJ_dW_s = dJ_dW_s + l * self.W_s
      dJ_dW_t = dJ_dW_t + l * self.W_t
    dJ_db_s = (1./m) * accumulator_b_s
    dJ_db_t = (1./m) * accumulator_b_t

    return dJ_dW_s, dJ_dW_t, dJ_db_s, dJ_db_t, srcDeltas, tgtDeltas, srcTotalError/nonNonePhrasePairs, tgtTotalError/nonNonePhrasePairs


  def calculateDeltas(self, x, y, W, b):
    z = np.dot(W, x) + b
    y_prime = self.a.apply(z)
    y_prime_prime = self.a.derivative(z)

    hypDiff = self.computeL2Norm(y, y_prime)
    return hypDiff * y_prime_prime, hypDiff


  def train(self, phrasePairs, numIterations, learningRate, regularLambda):
    srcRawPhrases = [phrasePair[0] for phrasePair in phrasePairs]
    tgtRawPhrases = [phrasePair[1] for phrasePair in phrasePairs]
    m = len(phrasePairs)

    for i in range(numIterations):
      print "Semantic tuning : Iteration (" + str(i) + ")"
      srcPhraseVectors, srcReconError = \
          self.getTrainedPhraseVectors(self.srcPhraseModel, srcRawPhrases, regularLambda)
      tgtPhraseVectors, tgtReconError = \
          self.getTrainedPhraseVectors(self.tgtPhraseModel, tgtRawPhrases, regularLambda)
      phrasePairVectors = [(x,y) for x,y in zip(srcPhraseVectors, tgtPhraseVectors)]

      # Calculate partial derivatives for the semantic transformation parameters
      dJ_dW_s, dJ_dW_t, dJ_db_s, dJ_db_t, srcDeltas, tgtDeltas, srcError, tgtError = \
        self.calculateSumOfGradients(phrasePairVectors, True, regularLambda)
      self.W_s = self.W_s = learningRate * (1 - self.alpha) * dJ_dW_s
      self.W_t = self.W_t = learningRate * (1 - self.alpha) * dJ_dW_t
      self.b_s = self.b_s = learningRate * (1 - self.alpha) * dJ_db_s
      self.b_t = self.b_t = learningRate * (1 - self.alpha) * dJ_db_t

      print "Tuning source"
      srcError = srcError + regularLambda * np.sum(self.W_s * self.W_s)
      print "Source semantic error = " + str(srcError)
      print "Source total error = " + str(self.alpha * srcReconError + (1 - self.alpha) * srcError)
      self.tunePhraseEmbeddingModel(self.srcPhraseModel, srcDeltas, regularLambda, learningRate, self.W_s)
      print "Tuning target"
      tgtError = tgtError + regularLambda + np.sum(self.W_t * self.W_t)
      print "Target semantic error = " + str(tgtError)
      print "Target total error = " + str(self.alpha * tgtReconError + (1 - self.alpha) * tgtError)
      self.tunePhraseEmbeddingModel(self.tgtPhraseModel, tgtDeltas, regularLambda, learningRate, self.W_t)
      print "--------------------------"


  def getTrainedPhraseVectors(self, phraseModel, rawPhrases, regularLambda):
    phraseVectors = []
    for phrase in rawPhrases:
      phraseVectors.append(phraseModel.computeOptimalBinaryTree(phrase))

    m = 0
    for item in phraseModel.finalLayerBackPropData:
      if item[0] is not None:
        m = m + 1

    reconCost = (1./m) * phraseModel.totalReconstructionError + \
        regularLambda * (np.sum(phraseModel.a.paramMatrix * phraseModel.a.paramMatrix) + \
        np.sum(phraseModel.a.reconParamMatrix * phraseModel.a.reconParamMatrix))

    return phraseVectors, reconCost



  def tunePhraseEmbeddingModel(self, phraseModel, deltas, regularLambda, learningRate, W):
    # Get partial derivatives
    # At this point, the training data for backprop has already been created
    # since the optimal binary tree for all phrases was pre-calculated.
    dJ_dW1, dJ_dW2, dJ_db2, dJ_db3 =  phraseModel.a.getGradients(phraseModel.finalLayerBackPropData, True, regularLambda, \
        self.alpha, deltas, W)

    # SGD : Update parameters
    phraseModel.a.paramMatrix = phraseModel.a.paramMatrix - learningRate * dJ_dW1
    phraseModel.a.reconParamMatrix = phraseModel.a.reconParamMatrix - learningRate * dJ_dW2
    phraseModel.a.bias = phraseModel.a.bias - learningRate * dJ_db2
    phraseModel.a.reconBias = phraseModel.a.reconBias - learningRate * dJ_db3

    # Reset reconstruction error for the next iteration
    phraseModel.totalReconstructionError = 0.
    phraseModel.backPropTrainingData = []
    phraseModel.finalLayerBackPropData = []


  def computeL2Norm(self, input1, input2):
    try:
      inputDiff = input1 - input2
    except:
      raise Exception("Invalid input type")
    return np.sqrt(np.sum(inputDiff**2))

