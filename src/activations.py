import numpy as np
# Compute the forward and backward propagation for every actvations functions (except for the softmax function which is only used as the activation function of the last layer)
class ReLU:
    def Forward_prop(self, Z): 
      self.Z = Z
      self.A = np.maximum(0,Z)
      return np.maximum(0, Z)

    def Backward_prop(self, dA): 
      return dA*(self.A > 0)

class Sigmoid:
    def Forward_prop(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def Backward_prop(self, dA):
        return dA * (self.A * (1 - self.A))


class Softmax:
    def Forward_prop(self, Z): 
        exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))

        return exp / np.sum(exp, axis=1, keepdims=True)
