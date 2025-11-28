import numpy as np 
def accuracy(y,y_preds):
  return np.mean(y == y_preds)