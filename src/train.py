import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data import scale_features
from utils.metrics import accuracy
from NN import Neural_Net
from activations import ReLU, Sigmoid, Softmax

df_train = pd.read_csv("Data/pendigits.tra",sep=",")
df_train.head()

X_train = df_train.drop(["class"],axis=1)
#X_train.head()
y_train = df_train["class"]
#y_train.head()
print(len(y_train))

X_train,means,stds = scale_features(X_train)

df_test = pd.read_csv("Data/pendigits.tes",sep=",")
df_test.head()

X_test = df_test.drop(["class"],axis=1)
#X_train.head()
y_test = df_test["class"]
#y_train.head()
X_test = (X_test - means)/stds
X_test.head()

X_train,X_test,y_train,y_test = np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)

layers_dim = [16, 32, 16, len(np.unique(y_train))]
#print(np.unique(y_train))
activations = [ReLU(),ReLU()]
model = Neural_Net(layers_dim,activations)
losses = model.train(X_train,y_train,0.05,3500)
#print(losses)

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss evolution")
plt.grid(True)
plt.show()

y_preds = model.predict(X_test)
acc = np.mean(y_preds == y_test)
print("Test accuracy:", acc)
losses[:10]