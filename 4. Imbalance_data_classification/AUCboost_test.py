# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#### Importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# %%
bank = pd.read_csv('Personal Loan.csv')
input_idx = [1,2,3,5,6,7,8,10,11,12,13]
target_idx = 9

X = np.array(bank.iloc[:, input_idx])
y = np.array(bank.iloc[:, target_idx])

X = X[y < 2,:]
y = y[y < 2]

sss = StratifiedShuffleSplit(n_splits=1,train_size=0.7)

for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# %%
wgt = np.repeat(1/X_train.shape[0], X_train.shape[0])

model = DecisionTreeClassifier(max_depth=30, min_samples_split=20)
model.fit(X_train, y_train, sample_weight=wgt)

pred_y = model.predict(X_test)

x = confusion_matrix(y_true=y_test,y_pred=pred_y)
print("confusion matrix = \n", x)

tn, fp, fn, tp = confusion_matrix(y_true=y_test,y_pred=pred_y).ravel()
tpr = tp/(tp+fn)
fpr = fp/(tn+fp)
auc = (1+tpr-fpr)/2

print("TPR: {:.3f}, FPR: {:.3f}, AUC: {:.3f}".format(tpr,fpr,auc))


# %%
cit = np.array([])
ait = np.array([])
x_bst = np.array([])

iter = 20

for i in range(0,iter):
    model = DecisionTreeClassifier(max_depth=5, min_samples_split=20)
    model.fit(X_train, y_train, sample_weight=wgt)

    ci = model.predict(X_train)
    cit = np.append(cit, ci).reshape(i+1,-1)

    tn, fp, fn, tp = confusion_matrix(y_train,ci).ravel()
    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)
    auc = (1+tpr-fpr)/2

    ai = 0.5 * np.log(wgt[ci == y_train].sum()/wgt[ci != y_train].sum())
    ait = np.append(ait, ai)

    loss = np.exp(-ai*np.sign((ci == y_train)-0.5))
    wls = wgt * loss
    zi_pos = np.sum(wls[y_train==1])
    zi_neg = np.sum(wls[y_train==0])

    wls[y_train==1] *= 1/zi_pos
    wls[y_train==0] *= 1/zi_neg

    wgt = wls

    print("iteration {:3d},TPR: {:.3f}, FPR: {:.3f}, AUC: {:.3f}".format(i,tpr,fpr,auc))


# %%
pred = np.array([])
for i in range(0,iter):
    x = 0
    for j in range(0,i+1):
        x = x + ait[j]*cit[j,:]
    # print(x)
    pred = np.append(pred, x)

pred = np.sign(pred.reshape(iter,-1))

print(pred)

y_bst = np.array([])
y_train_tmp = np.sign((y_train==1)-0.5)
for i in range(0,iter):
    tn, fp, fn, tp = confusion_matrix(y_train,pred[i,:]).ravel()
    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)
    auc = (1+tpr-fpr)/2

    print("iteration {:3d},TPR: {:.3f}, FPR: {:.3f}, AUC: {:.3f}".format(i,tpr,fpr,auc))

# %%
plt.plot(range(0,iter), x_bst)
plt.title('Error from each boosting iterations')
plt.show()


# %%
plt.plot(range(0,iter), y_bst)
plt.title("Error arising from averaging predictions")
plt.show()


# %%



