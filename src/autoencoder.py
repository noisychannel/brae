import numpy as np

class AutoEncoder:

  # Dimensionality of input
  n = 100

  # The parameter matrix W_1
  paramMatrix = None

  # The bias vector b_1
  bias = None

  # The reconstruction parameter matrix W_2
  reconParamMatrix = None

  # The bias vector b_2
  reconBias = None

  # The activation function
  activation = None

  def __init__(self, n=100, activation=None):
    self.n = n
    self.activation = activation
    self.paramMatrix = np.zeroes(n, 2*n)
    self.bias = np.zeroes(n, 1)
    self.reconParamMatrix = np.zeroes(n, 2*n)
    self.reconBias = np.zeroes(n, 1)

  # Computes the parent vector from the input vectors
  # c1 and c2 are n-dimensional vectors
  # p = f(W * [c1;c2] + b)
  def computeParentVector(c1, c2):
    # Sanity check for dimensionality of input
    try:
      (np.shape(c1) == (n, 1)) & (np.shape(c2) == (n, 1))
    except:
      print "Input shapes are not consistent with expected dimensionality"
      print "Expected dimensionality : " + str(n)
      print "Got : c1 = " + str(np.shape(c1))
      print "Got : c2 = " + str(np.shape(c2))
      raise

    c1c2 = np.hstack(c1, c2)
    p = activation.apply(self.paramMatrix * c1c2 + bias)

    return p

  # Computes the reconstruction error between a parent and its children
  # [c1':c2'] = f(W_2 * p + b_2)
  def computeReconstructionError(p, c1, c2):
    try:
      (np.shape(c1) == (n, 1)) & (np.shape(c2) == (n, 1) & np.shape(p) == (n, n))
    except:
      print "Input shapes are not consistent with expected dimensionality"
      print "Expected dimensionality : " + str(n)
      print "Got : c1 = " + str(np.shape(c1))
      print "Got : c2 = " + str(np.shape(c2))
      print "Got : p = " + str(np.shape(c2))
      raise

    c1c2Prime = activation.apply(self.reconParamMatrix * p + reconBias )

    return computeL2Norm(c1c2Prime, np.hstack(c1,c2))


  def computeL2Norm(input1, input2):
    try:
      (np.shape(input1) == np.shape(input2))
    except:
      print "Input shapes are not the same"
      print "Got : input1 = " + str(np.shape(input1))
      print "Got : input2 = " + str(np.shape(input2))
      raise

    inputDiff = input1 - input2

    return np.sqrt(np.sum(inputDiff**2))
