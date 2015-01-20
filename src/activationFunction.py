import numpy as np

class ActivationFunction:

  activationFunction = None

  def __init__(type = ""):
    if type == "tanh":
      activationFunction = np.tanh
    else:
      print "Unsupported activation function type : " + type
      raise

  def apply(input):
    return self.activationFunction(input)
