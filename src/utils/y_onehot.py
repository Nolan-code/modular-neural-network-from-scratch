import numpy as np

def compute_y_onehot(y,n_classes):             #instead of having the class (0 or 1 or 2) for every row, create a list of list where the class is the index of the 1
  y = np.array(y)
  y_onehot = np.zeros((len(y),n_classes))    #in this sublist (all the other term are set to 0)
  for k in range(len(y)):
    y_onehot[k][y[k]] = 1
  return y_onehot