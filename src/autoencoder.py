import numpy as np
import activationfunction as activationfunction

class AutoEncoder:

  # Dimensionality of input
  n = 100

  # The parameter matrix W_1 (n X 2n)
  paramMatrix = None

  # The bias vector b_1 (n X 1)
  bias = None

  # The reconstruction parameter matrix W_2 (2n X n)
  reconParamMatrix = None

  # The bias vector b_2 (2n X 1)
  reconBias = None

  # The activation function
  activation = None

  def __init__(self, n=100, activation=None):
    self.n = n
    if activation:
      self.activation = activation
    else:
      self.activation = activationfunction.ActivationFunction("tanh")
    self.paramMatrix = np.random.rand(n, 2*n)
    self.bias = np.random.rand(n, 1)
    self.reconParamMatrix = np.random.rand(2*n, n)
    self.reconBias = np.random.rand(2*n, 1)

  # Computes the parent vector from the input vectors
  # c1 and c2 are n-dimensional vectors
  # p = f(W * [c1;c2] + b)
  def computeParentVector(self, c1, c2):
    # Sanity check for dimensionality of input
    try:
      (np.shape(c1) == (self.n, 1)) & (np.shape(c2) == (self.n, 1))
    except:
      print "Input shapes are not consistent with expected dimensionality"
      print "Expected dimensionality : " + str(self.n)
      print "Got : c1 = " + str(np.shape(c1))
      print "Got : c2 = " + str(np.shape(c2))
      raise

    c1c2 = np.vstack((c1, c2))
    # Parent before the activation function is applied
    p_z = np.dot(self.paramMatrix, c1c2) + self.bias
    # Application of the activation function
    p = self.activation.apply(p_z)

    return p_z, p

  # Computes the reconstruction error between a parent and its children
  # for one training example
  # No regularization included here. It can be included when calculating the total error
  # [c1':c2'] = f(W_2 * p + b_2)
  def computeSingleReconstructionError(self, p, c1, c2):
    try:
      (np.shape(c1) == (self.n, 1)) & (np.shape(c2) == (self.n, 1)) & (np.shape(p) == (self.n, 1))
    except:
      print "Input shapes are not consistent with expected dimensionality"
      print "Expected dimensionality : " + str(self.n)
      print "Got : c1 = " + str(np.shape(c1))
      print "Got : c2 = " + str(np.shape(c2))
      print "Got : p = " + str(np.shape(c2))
      raise

    c1c2Prime_z = np.dot(self.reconParamMatrix, p) + self.reconBias
    c1c2Prime = self.activation.apply(c1c2Prime_z)
    reconstructionError = self.computeL2Norm(c1c2Prime, np.vstack((c1,c2)))

    return c1c2Prime_z, c1c2Prime, reconstructionError


  def computeL2Norm(self, input1, input2):
    try:
      (np.shape(input1) == np.shape(input2))
    except:
      print "Input shapes are not the same"
      print "Got : input1 = " + str(np.shape(input1))
      print "Got : input2 = " + str(np.shape(input2))
      raise

    inputDiff = input1 - input2

    return np.sqrt(np.sum(inputDiff**2))

  def getGradients(self, training, regularization=True, l=0.1):
    # Initialize weight and bias accumulators
    weightAccumulator_1 = np.zeros((self.n, self.n*2))
    weightAccumulator_2 = np.zeros((self.n*2, self.n))
    biasAccumulator_2 = np.zeros((self.n, 1))
    biasAccumulator_3 = np.zeros((self.n * 2, 1))
    for instance in training:
      # First compute forward activations for each layer for this training instance
      # 2n X 1
      x = instance[0]
      # n X 1
      activationValues_2 = instance[1]
      z_2 = instance[2]
      # 2n X 1
      activationValues_3 = instance[3]
      z_3 = instance[4]

      # Now calculate the delta terms for each layer and add to the accumulator
      # 2n X 1
      delta_3 = self.computeL2Norm(activationValues_3, x) * \
          self.activation.derivative(z_3)
      # Note the use of the Hadamard product here
      # n x 1
      delta_2 = np.dot(np.transpose(self.reconParamMatrix), delta_3) * \
          self.activation.derivative(z_2)

      # Now add these layer errors to the accumulator
      weightAccumulator_1 = weightAccumulator_1 + np.dot(delta_2, np.transpose(x))
      biasAccumulator_2 = biasAccumulator_2 + delta_2
      weightAccumulator_2 = weightAccumulator_2 + np.dot(delta_3, np.transpose(activationValues_2))
      biasAccumulator_3 = biasAccumulator_3 + delta_3

    # Compute derivatives now
    m = len(training)
    dJ_dW1 = (1./m) * weightAccumulator_1
    dJ_dW2 = (1./m) * weightAccumulator_2
    if (regularization):
      dJ_dW1 = dJ_dW1 + l * self.paramMatrix
      dJ_dW2 = dJ_dW2 + l * self.reconParamMatrix
    dJ_db2 = (1./m) * biasAccumulator_2
    dJ_db3 = (1./m) * biasAccumulator_3

    return dJ_dW1, dJ_dW2, dJ_db2, dJ_db3
