import numpy as np

class ActivationFunction:

  activationFunction = None

  def __init__(self, type = ""):
    if type == "tanh":
      self.activationFunction = np.tanh
    else:
      print "Unsupported activation function type : " + type
      raise

  def apply(self, input):
    return self.activationFunction(input)
