import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def SMOTE(X,y,K=5,dup_size=1):
    X = X[y==1,:]
    nbrs = NearestNeighbors(n_neighbors=K).fit(X)
    _,indices = nbrs.kneighbors(X)

    minor_count = np.sum(y==1)
    rep = np.floor(dup_size).astype('int32')
    rest = int((dup_size % 1)*minor_count)

    syns = np.zeros((rep*minor_count+rest,X.shape[1]))
    idx = 0
    for _ in range(rep):
        for m in range(minor_count):
            cand = np.random.choice(indices[m, 1:],1)[0]
            dif = X[cand,:]-X[m,:]
            gap = np.random.uniform(0,1,1)[0]
            syns[idx,:] = X[m,:]+gap*dif
            idx += 1

    rest_idx = np.random.choice(range(minor_count),rest)
    for m in rest_idx:
        cand = np.random.choice(indices[m, 1:], 1)[0]
        dif = X[cand, :] - X[m, :]
        gap = np.random.uniform(0, 1, 1)[0]
        syns[idx, :] = X[m, :] + gap * dif
        idx += 1

    return syns.reshape((-1, X.shape[1]))

def RUS(X,y,p):
    pos_idx = np.where(y==1)[0]
    neg_idx = np.where(y==0)[0]

    sample_neg_idx = np.random.choice(neg_idx,np.int(len(neg_idx)*p))

    X = X[np.concatenate((sample_neg_idx,pos_idx)),:]
    y = y[np.concatenate((sample_neg_idx,pos_idx))]

    return X,y

def decision_plot(X,y,model):
    lows = np.min(X,axis=0)
    upps = np.max(X,axis=0)

    x1 = np.linspace(lows[0],upps[0],100)
    x2 = np.linspace(lows[1],upps[1],100)
    new_x = np.array(np.meshgrid(x1,x2)).reshape(2,100*100).T
    new_y = model.predict(new_x).reshape(100,100)

    fig, ax = plt.subplots()
    ax.contour(x1, x2, new_y,cmap='binary')
    ax.plot(X[y==0,0],X[y==0,1],'.',c='black')
    ax.plot(X[y==1,0],X[y==1,1],'.',c='red')
    fig.show()