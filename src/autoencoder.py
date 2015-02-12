import numpy as np
import activationfunction as activationfunction

class AutoEncoder:

  def __init__(self, n=100, activation=None):
    self.n = n
    if activation:
      self.activation = activation
    else:
      self.activation = activationfunction.ActivationFunction("tanh")
    if self.activation.funcType == "tanh":
      # Initialization of weight matrix based on Xavier_10
      self.paramMatrix = np.random.uniform(-np.sqrt(6./(n + 2*n)), np.sqrt(6./(n + 2*n)), (n, 2*n))
      self.reconParamMatrix = np.random.uniform(-np.sqrt(6./(n + 2*n)), np.sqrt(6./(n + 2*n)), (2*n, n))
    elif self.activation.funcType == "sigmoid":
      # Initialization of weight matrix based on Xavier_10
      self.paramMatrix = np.random.uniform(-4. * np.sqrt(6./(n + 2*n)), 4. * np.sqrt(6./(n + 2*n)), (n, 2*n))
      self.reconParamMatrix = np.random.uniform(-4. * np.sqrt(6./(n + 2*n)), 4. * np.sqrt(6./(n + 2*n)), (2*n, n))
    self.bias = np.zeros((n, 1))
    self.reconBias = np.zeros((2*n, 1))

  # Computes the parent vector from the input vectors
  # c1 and c2 are n-dimensional vectors
  # p = f(W * [c1;c2] + b)
  def computeParentVector(self, c1, c2, paramMatrix, bias):
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
    p_z = np.dot(paramMatrix, c1c2) + bias
    # Application of the activation function
    p = self.activation.apply(p_z)

    return p_z, p

  # Computes the reconstruction error between a parent and its children
  # for one training example
  # No regularization included here. It can be included when calculating the total error
  # [c1':c2'] = f(W_2 * p + b_2)
  def computeSingleReconstructionError(self, p, c1, c2, reconParamMatrix, reconBias):
    try:
      (np.shape(c1) == (self.n, 1)) & (np.shape(c2) == (self.n, 1)) & (np.shape(p) == (self.n, 1))
    except:
      print "Input shapes are not consistent with expected dimensionality"
      print "Expected dimensionality : " + str(self.n)
      print "Got : c1 = " + str(np.shape(c1))
      print "Got : c2 = " + str(np.shape(c2))
      print "Got : p = " + str(np.shape(c2))
      raise

    c1c2Prime_z = np.dot(reconParamMatrix, p) + reconBias
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

  def getGradients(self, training, paramMatrix, reconParamMatrix, regularization=True, l=0.1, alpha=0.1, delta_3_s=None, W_s=None):
    """
    If alpha and delta_3_s are set, then we are trying to calculate semantic gradients
    That is, the size of W(2) now becomes, (3n X n) and the size of b(3) becomes (3n x 1)
    delta_3 = [delta_3_r, delta_3_s] -> 3n X 1
    """
    # Initialize weight and bias accumulators
    weightAccumulator_1 = np.zeros((self.n, self.n*2))
    weightAccumulator_2 = np.zeros((self.n*2, self.n))
    biasAccumulator_2 = np.zeros((self.n, 1))
    biasAccumulator_3 = np.zeros((self.n * 2, 1))
    for i, instance in enumerate(training):
      # First compute forward activations for each layer for this training instance
      # 2n X 1
      x = instance[0]
      # n X 1
      activationValues_2 = instance[1]
      z_2 = instance[2]
      # 2n X 1
      activationValues_3 = instance[3]
      z_3 = instance[4]

      # Skip training examples that are None. 
      if x is None:
        continue

      # Now calculate the delta terms for each layer and add to the accumulator
      # 2n X 1
      delta_3 = self.computeL2Norm(activationValues_3, x) * \
          self.activation.derivative(z_3)
      W_2 = reconParamMatrix

      if (delta_3_s is not None):
        # Semantic tuning
        if (delta_3_s[i] is None):
          continue
        # Now 3n X 1
        delta_3_tmp = np.vstack((alpha * delta_3, (1 - alpha) * delta_3_s[i]))
        # Now 3n X n
        W_2_tmp = np.vstack((W_2, W_s))
        delta_2 = np.dot(np.transpose(W_2_tmp), delta_3_tmp) * \
            self.activation.derivative(z_2)
      else:
        # Note the use of the Hadamard product here
        # n x 1
        delta_2 = np.dot(np.transpose(W_2), delta_3) * \
            self.activation.derivative(z_2)

      # Now add these layer errors to the accumulator
      weightAccumulator_1 = weightAccumulator_1 + np.dot(delta_2, np.transpose(x))
      biasAccumulator_2 = biasAccumulator_2 + delta_2
      weightAccumulator_2 = weightAccumulator_2 + np.dot(delta_3, np.transpose(activationValues_2))
      biasAccumulator_3 = biasAccumulator_3 + delta_3

#      print np.sqrt(np.sum(activationValues_3**2))
      #print np.sqrt(np.sum(x**2))

    # Compute derivatives now
    m = len(training)
    dJ_dW1 = (1./m) * weightAccumulator_1
    dJ_dW2 = (1./m) * weightAccumulator_2
    if (regularization):
      dJ_dW1 = dJ_dW1 + l * paramMatrix
      dJ_dW2 = dJ_dW2 + l * reconParamMatrix
    dJ_db2 = (1./m) * biasAccumulator_2
    dJ_db3 = (1./m) * biasAccumulator_3

    return dJ_dW1, dJ_dW2, dJ_db2, dJ_db3

