import numpy as np
from sklearn.tree import DecisionTreeClassifier


class SMOTE(object):
    def __init__(self, X, k):
        self.k = k
        self.Synthetic = []
        self.new_index = 0
        self.Xs = X

    def smote(self, N):
        T = len(self.Xs)
        if N < 100:
            T = int((N/100)*T)
            N = 100
        N = N//100
        for i in range(T):
            idx = np.random.randint(0, len(self.Xs))
            diff = np.sum((self.Xs[idx] - self.Xs)**2, axis=1)
            diff[idx] = np.inf
            nnarray = (np.argsort(diff)[:self.k])
            self.populate(N, self.Xs[idx], nnarray)
        return self.Synthetic

    def populate(self, N, p, idxs):
        candids = np.random.choice(idxs, size=N)
        for c in candids:
            self.Synthetic.append(
                (self.Xs[c] - p) * np.random.uniform(0, 1))
            self.new_index += 1


class ADABOOST_M2(object):
    def __init__(self, X, y):
        self.Ds = {}
        self.models = {}
        self.X = X
        self.N = len(X)
        self.y = np.copy(y)
        self.y[self.y < 0] = 0
        self.preds = {}
        self.bts = {}
        self.N = len(X)
        self.num_class = len(np.unique(y))

    def train(self, T):
        # D = np.ones((self.N, self.num_class)) / (self.N * (self.num_class - 1))
        D = np.array([1/self.N]*self.N)
        # D[np.arange(self.N), self.y] = 0
        for i in range(T):
            # W = np.sum(D, axis=1)
            dt = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            model = dt.fit(self.X, self.y, sample_weight=D)
            self.models[i] = model
            pred = model.predict_proba(self.X)
            mask = (np.argmax(pred, axis=1) != self.y)
            true_preds = pred[np.arange(len(self.X)), self.y].reshape(-1, 1)
            pred = pred - true_preds
            pred += 1
            pred[np.arange(self.N), self.y] = 0
            e = (1 / 2) * np.sum(np.sum(pred, axis=1)*D*mask)
            bt = e / (1 - e)
            self.bts[i] = bt
            p = np.copy(-1 * pred)
            p += 2
            p[np.arange(self.N), self.y] = 0
            D = D * np.power(bt, 0.5 * np.sum(p, axis=1))
            D = D / D.sum()
            self.Ds[i] = D

    def predict(self, X):
        preds = np.zeros((len(X), 2), dtype=np.float64)
        for i in range(len(self.models)):
            pred = np.array(self.models[i].predict_proba(X))
            preds += (np.log(1 / (self.bts[i]+1e-5)) * pred)
        y_fin = np.argmax(preds, axis=1)
        y_fin[y_fin == 0] = -1
        return y_fin


class RUSBOOST(object):
    def __init__(self, X, y, N):
        self.Ws = {}
        self.models = {}
        self.X = X
        self.y = np.copy(y)
        self.y[self.y < 0] = 0
        self.preds = {}
        self.bts = {}
        self.min_idx = np.where(self.y == 1)[0]
        self.maj_idx = np.where(self.y == 0)[0]
        self.N_p = N

    def train(self, T):
        # np.random.seed(0)
        W = np.ones(len(self.X))/len(self.X)
        self.N_p /= 100
        M_p = int((1-self.N_p) / self.N_p * len(self.min_idx))
        for i in range(T):
            idx = np.concatenate(
                (self.min_idx, np.random.choice(self.maj_idx, size=M_p)))
            CX = self.X[idx]
            Cy = self.y[idx]
            Cw = W[idx]
            dt = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            model = dt.fit(CX, Cy, sample_weight=Cw)
            self.models[i] = model
            pred = model.predict_proba(self.X)
            mask = (np.argmax(pred, axis=1) != self.y)
            self.preds[i] = pred
            true_preds = pred[np.arange(len(self.X)), self.y].reshape(-1, 1)
            pred = (pred - true_preds)
            pred += 1
            pred[np.arange(len(self.X)), self.y] = 0
            e = (1/2) * np.sum(np.sum(pred, axis=1) * mask * W)
            bt = e/(1-e)
            self.bts[i] = bt
            p = np.copy(-1 * pred)
            p += 2
            p[np.arange(len(self.X)), self.y] = 0
            W = W * np.power(bt, 0.5*(np.sum(p, axis=1)))
            W = W / W.sum()
            self.Ws[i] = W

    def predict(self, X):
        preds = np.zeros((len(X), 2), dtype=np.float64)
        for i in range(len(self.models)):
            pred = np.array(self.models[i].predict_proba(X))
            preds += (np.log(1 / self.bts[i]) * pred)
        y_fin = np.argmax(preds, axis=1)
        y_fin[y_fin == 0] = -1
        return y_fin


class SMOTE_BOOST(object):
    def __init__(self, X, y, k, sm_N):
        self.Ds = {}
        self.models = {}
        self.X = X
        self.y = y
        self.y[self.y < 0] = 0
        self.preds = {}
        self.bts = {}
        self.N = len(X)
        self.k = k
        self.min_idx = np.where(self.y == 1)[0]
        self.sm_N = sm_N

    def train(self, T):
        D = np.array(([1/self.N]*self.N))
        for i in range(T):
            sm = SMOTE(self.X[self.min_idx], self.k)
            s = sm.smote(self.sm_N)
            sx = np.concatenate((self.X, s))
            sy = np.concatenate((self.y, [1]*len(s)))
            CW = np.array([1/len(sx)]*len(s))
            CW = np.concatenate((D, CW))
            CW = CW/CW.sum()
            dt = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            model = dt.fit(sx, sy, sample_weight=CW)
            self.models[i] = model
            pred = model.predict_proba(self.X)
            mask = (np.argmax(pred, axis=1) != self.y)
            self.preds[i] = pred
            true_preds = pred[np.arange(len(self.X)), self.y].reshape(-1, 1)
            pred = pred - true_preds
            pred += 1
            pred[np.arange(self.N), self.y] = 0
            e = (1 / 2) * np.sum(np.sum(pred, axis=1)*D*mask)
            bt = e / (1 - e)
            self.bts[i] = bt
            p = np.copy(-1 * pred)
            p += 2
            p[np.arange(self.N), self.y] = 0
            D = D * np.power(bt, 0.5 * np.sum(p, axis=1))
            D = D / D.sum()
            self.Ds[i] = D

    def predict(self, X):
        preds = np.zeros((len(X), 2), dtype=np.float64)
        for i in range(len(self.models)):
            pred = np.array(self.models[i].predict_proba(X))
            preds += (np.log(1 / self.bts[i]) * pred)
        y_fin = np.argmax(preds, axis=1)
        y_fin[y_fin == 0] = -1
        return y_fin


class RANDOM_BALANCEX(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N = len(X)
        self.min_idx = np.where(y == 1)[0]
        self.maj_idx = np.where(y == 0)[0]

    def rbx(self, k):
        min_size = len(self.min_idx)
        maj_size = len(self.maj_idx)
        n_maj_size = np.random.randint(2, self.N-2)
        n_min_size = self.N - n_maj_size
        S_p = []
        if n_maj_size < maj_size:
            A = self.X[self.min_idx]
            size = n_min_size-min_size
            size = int(np.floor(size/min_size))
            d = size*min_size + min_size
            r_maj_idx = np.random.choice(
                self.maj_idx, size=self.N - d, replace=False)
            C = self.X[r_maj_idx]
            idx = np.concatenate((self.min_idx, r_maj_idx))
            lb = [1]
            size *= 100

        else:
            A = self.X[self.maj_idx]
            size = n_maj_size-maj_size
            size = int(np.floor(size/maj_size))
            d = maj_size + size*maj_size
            r_min_idx = np.random.choice(
                self.min_idx, self.N-d, replace=False)

            idx = np.concatenate((self.maj_idx, r_min_idx))
            C = self.X[r_min_idx]
            size *= 100
            lb = [-1]

        sm = SMOTE(A, k)
        syn = sm.smote(size)
        S_p.extend(np.concatenate((A, C)))
        S_p.extend(syn)
        S_py = np.concatenate((self.y[idx], len(syn)*lb))
        S_py = np.int8(S_py)
        return S_p, S_py, idx


class RB_BOOST(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.models = {}
        self.bts = {}
        self.Ws = {}
        self.y = np.copy(y)
        self.y[self.y < 0] = 0

    def train(self, k, T):
        X = self.X
        y = self.y
        N = len(X)
        W = np.ones(N)/N
        proportions = []
        for i in range(T):
            CX, Cy, idx = RANDOM_BALANCEX(X, y).rbx(k)
            proportions.append(round(np.count_nonzero(Cy)/len(Cy), 2))
            Cw = np.ones_like(W)/N
            Cw[idx] = np.copy(W[idx])
            dt = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            model = dt.fit(CX, Cy, sample_weight=Cw)
            self.models[i] = model
            pred = model.predict_proba(CX)
            mask = (np.argmax(pred, axis=1) != Cy)
            true_preds = pred[np.arange(len(self.X)), Cy].reshape(-1, 1)
            pred = (pred - true_preds)
            pred += 1
            pred[np.arange(len(self.X)), Cy] = 0
            e = (1/2) * np.sum(np.sum(pred, axis=1) * mask * Cw)
            bt = e/(1-e)
            self.bts[i] = bt
            p = np.copy(-1 * pred)
            p += 2
            p[np.arange(len(self.X)), Cy] = 0
            Cw = Cw * np.power(bt, 0.5*(np.sum(p, axis=1)))
            W[idx] = np.copy(Cw[idx])
            W = W / W.sum()
            self.Ws[i] = W

    def predict(self, X):
        preds = np.zeros((len(X), 2), dtype=np.float64)
        for item in self.models.items():
            i = item[0]
            pred = np.array(item[1].predict_proba(X))
            preds += np.log(1 / (self.bts[i]+1e-6)) * pred
        y_fin = np.argmax(preds, axis=1)
        y_fin[(y_fin == 0)] = -1
        return y_fin
