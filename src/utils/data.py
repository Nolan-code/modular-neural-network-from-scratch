import pandas as pd

df_train = pd.read_csv("pendigits.tra",sep=",")
df_train.head()

X_train = df_train.drop(["class"],axis=1)
#X_train.head()
y_train = df_train["class"]
#y_train.head()
print(len(y_train))

X_train,means,stds = scale_features(X_train)

df_test = pd.read_csv("pendigits.tes",sep=",")
df_test.head()

X_test = df_test.drop(["class"],axis=1)
#X_train.head()
y_test = df_test["class"]
#y_train.head()
X_test = (X_test - means)/stds
X_test.head()

X_train,X_test,y_train,y_test = np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test),

def scale_features(X):    #normalize features
    X = np.array(X)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    return (X - means) / stds, means, stds  #return the normalized dataset, the mean and the variance to normalize the test set with the same parameters