import numpy as np

class ActivationFunction:

  activationFunction = None
  activationFunction_vectorized = None
  funcType = None

  def __init__(self, type = ""):
    if type == "tanh":
      self.funcType = "tanh"
      self.activationFunction = np.tanh
      # np.tanh is already vectorized, operates on elements when an array is provided
      self.activationFunction_vectorized = self.activationFunction
    else:
      print "Unsupported activation function type : " + type
      raise


  def apply(self, input):
    return self.activationFunction(input)


  def derivative(self, input):
    return 1 - self.apply(input)**2
