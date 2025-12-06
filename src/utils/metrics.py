import numpy as np 
def accuracy(y,y_preds):  #compute the accuracy of the model predictions
  return np.mean(y == y_preds)
