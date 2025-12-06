import numpy as np

class Neural_Net:
  def __init__(self,layers_dim,activations):
    # layers_dims: a list of the number of neurones of the corresponding layer 
    # activations: a list contaning the name of the activation function

    self.L = len(layers_dim) - 1
    self.activations = activations
    self.parameters = {}  # A dictionnary containing the weight and the bias 

    for l in range(1, self.L + 1):  # From the first layer to the last one
      self.parameters[f"W{l}"] = (np.random.rand(layers_dim[l-1],layers_dim[l])) * (np.sqrt(2 / layers_dim[l-1])) # The size of the weight matrix of the l-layer is the nb of neurone in the previous layer * the nb of neurone in the current one
      self.parameters[f"b{l}"] = np.zeros((1,layers_dim[l])) # The size of the bias matrix is 1 * the number of neurone in the current layer

  def Forward_prop(self, X):
    self.memory = {"A0": X}
    A = X

    for l in range(1, self.L + 1):  # Loop on the number of layer
      W = self.parameters[f"W{l}"]
      b = self.parameters[f"b{l}"]

      #print(f"\n---- Layer {l} ----")
      #print("A shape:", A.shape)
      #print("W shape:", W.shape)
      #print("b shape:", b.shape)


      Z = A @ W + b
      if l < self.L:  
      # Hidden layer => activation
        A = self.activations[l-1].Forward_prop(Z)
      else:
      # Output layer => softmax
        A = Softmax().Forward_prop(Z)

      self.memory[f"Z{l}"] = Z
      self.memory[f"A{l}"] = A

    return A

  def Backward_prop(self,X,y,lr):
    n = X.shape[0]
    grads = {}

    A_last =self.memory[f"A{self.L}"]     
    dZ = (A_last - compute_y_onehot(y,len(np.unique(y))))/n   # Grad of the softmax function
    n_classes = A_last.shape[0]

    for l in reversed(range(1,self.L +1)):  # We compute the gradient backward
        A_prev = self.memory[f"A{l-1}"]
        W = self.parameters[f"W{l}"]

        grads[f"W{l}"] = A_prev.T @ dZ                       # Compute and store the grad with respect to W[l]
        grads[f"b{l}"] = np.sum(dZ, axis=0, keepdims=True)   # Compute and store the grad with respect to b[l]

        self.parameters[f"W{l}"] -= lr*grads[f"W{l}"]    # Update the weight
        self.parameters[f"b{l}"] -= lr*grads[f"b{l}"]    # Update the bias

        if l > 1:
          dA_prev = dZ @ W.T
          dZ = self.activations[l-2].Backward_prop(dA_prev)  # l-2 and not l-1 because of the gap explain above 
    
  def train(self,X,y,lr,n_iters):
    X = np.array(X)
    y = np.array(y)

    n = len(y)
    losses = []
    y_oh = compute_y_onehot(y,len(np.unique(y)))

    for iter in range(n_iters):
      y_preds = self.Forward_prop(X)     # Compute and store all the value in the forward prop
      loss = -np.mean(np.sum(y_oh * np.log(y_preds + 1e-10), axis=1))      
      losses.append(loss)
      self.Backward_prop(X,y,lr)
      print(loss)  # Update the weight and bias
    return losses
  def predict(self,X):
    X = np.array(X)
    A_last = self.Forward_prop(X)

    return np.argmax(A_last,axis=1)   #return a list containing the class that has the highest prob for every sample
