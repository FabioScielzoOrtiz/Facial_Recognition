import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from itertools import  product
from sklearn.preprocessing import StandardScaler
import sys
import random
from sklearn.metrics import accuracy_score

sys.path.insert(0, 'C:/Users/fscielzo/Documents/DataScience-GitHub/Dimension Reduction/PCA')
from PCA import PCA_class

##################################################################################################

class FacialRecognitionKnn :

    def __init__(self, n_neighbors=10, metric='euclidean', threshold_dist=0):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.threshold_dist = threshold_dist

    def set_params(self, **params):
        # Method to set the parameters of the estimator
        for key, value in params.items():
            setattr(self, key, value)
        return self
       
    def fit(self, X, Y) :

        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric)
        knn.fit(X, Y)          
        self.knn = knn

    def predict(self, X) :

        # Predicting IN/OUT
        Z_hat = np.repeat(0, len(X))
        Y_hat = np.repeat(0, len(X))
        nearest_neigh_dist, nearest_neigh_index = self.knn.kneighbors(X, n_neighbors=1) 
        for i in range(0, len(X)):
            if nearest_neigh_dist[i] >= self.threshold_dist :
                Z_hat[i] = 0 # 0 = OUT
            else :
                Z_hat[i] = 1 # 1 = IN 
        
        num_images_IN = np.sum(Z_hat == 1)
        if  num_images_IN > 0 :
            Y_hat[Z_hat == 1] = self.knn.predict(X[Z_hat == 1,:])   
        
        return Y_hat

#######################################################################################################

class RandomSearchFacialRecognitionKnn :
   
    def __init__(self, param_grid, n_trials=10, random_state=123) :

       self.param_grid = param_grid
       self.n_trials = n_trials
       self.random_state = random_state

    def fit(self, X_train, Y_train, X_test, Y_test):      

        accuracy, accuracy_IN, accuracy_OUT = {}, {}, {}
        search_space = list(product(*[self.param_grid[x] for x in self.param_grid.keys()]))
        random.seed(self.random_state)
        random_search_space = random.sample(search_space, self.n_trials)
        random_search_space = [{x: random_search_space[j][i] for i, x in enumerate(self.param_grid.keys())} for j in range(0, len(random_search_space))]
        
        for i in range(0, self.n_trials):
            n_neighbors, metric, threshold_dist = random_search_space[i]['n_neighbors'], random_search_space[i]['metric'], random_search_space[i]['threshold_dist']
            face_recognition_knn = FacialRecognitionKnn(n_neighbors=n_neighbors, metric=metric, threshold_dist=threshold_dist)
            face_recognition_knn.fit(X_train, Y_train)
            Y_test_hat  = face_recognition_knn.predict(X_test)
            # General Accuracy 
            accuracy[(n_neighbors, metric, threshold_dist)] = np.round(accuracy_score(y_pred=Y_test_hat, y_true=Y_test), 3) 
            # Accuracy for IN images
            accuracy_IN[(n_neighbors, metric, threshold_dist)] = np.round(accuracy_score(y_pred=Y_test_hat[np.where(Y_test != 0)], y_true=Y_test[np.where(Y_test != 0)]), 3) 
            # Accuracy for OUT images
            accuracy_OUT[(n_neighbors, metric, threshold_dist)] = np.round(accuracy_score(y_pred=Y_test_hat[np.where(Y_test == 0)], y_true=Y_test[np.where(Y_test == 0)]), 3) 

        combinations = [x for x in accuracy.keys()]
        accuracy_values = np.array([x for x in accuracy.values()])
        accuracy_IN_values = np.array([x for x in accuracy_IN.values()])
        accuracy_OUT_values = np.array([x for x in accuracy_OUT.values()])

        results = pd.DataFrame({'combination': combinations, 'accuracy': accuracy_values, 
                                   'accuracy_IN': accuracy_IN_values, 'accuracy_OUT': accuracy_OUT_values})
        results = results.sort_values(by='accuracy', ascending=False)
        params_keys = [x for x in self.param_grid.keys()]
        results[params_keys] = results['combination'].apply(lambda x: pd.Series(x))

        best_params = dict()
        best_params['n_neighbors'] = results['n_neighbors'].iloc[0]
        best_params['metric'] = results['metric'].iloc[0]
        best_params['threshold_dist'] = results['threshold_dist'].iloc[0]

        self.best_params_ = best_params
        self.best_score_ = results.iloc[0,1]
        self.results = results 

##################################################################################################

class FacialRecognitionKnnPca :

    def __init__(self, n_neighbors=5, metric='euclidean', threshold_dist=0, n_components=3):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.threshold_dist = threshold_dist
        self.n_components = n_components

    def set_params(self, **params):
        # Method to set the parameters of the estimator
        for key, value in params.items():
            setattr(self, key, value)
        return self
     
    def fit(self, X, Y) :

        metric = self.metric
        n_neighbors = self.n_neighbors
        n_components = self.n_components

        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        pca = PCA_class(n_components=n_components)
        pca.fit(X=X_scaled, solver='spectral')
        X_pca = pca.transform(X_scaled)

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        knn.fit(X_pca, Y)          
        
        self.knn = knn
        self.pca = pca
        self.mu = mu
        self.sigma = sigma

    def predict(self, X) :

        T = self.pca.T
        X_scaled = (X - self.mu) / self.sigma
        X_pca = X_scaled @ T

        # Predicting IN/OUT
        Y_hat, Z_hat = np.repeat(0, len(X_pca)), np.repeat(0, len(X_pca))
        nearest_neigh_dist, nearest_neigh_index = self.knn.kneighbors(X_pca, n_neighbors=1) 
        for i in range(0, len(X_pca)):
            if nearest_neigh_dist[i] > self.threshold_dist :
                Z_hat[i] = 0 # 0 = OUT
            else :
                Z_hat[i] = 1 # 1 = IN 
        
        num_images_IN = np.sum(Z_hat == 1)
        if  num_images_IN > 0 :
            Y_hat[Z_hat == 1] = self.knn.predict(X_pca[Z_hat == 1,:])   
        
        return Y_hat
    
#######################################################################################################
    
class RandomSearchFacialRecognitionKnnPca :
   
    def __init__(self, param_grid, n_trials=10, random_state=123) :

       self.param_grid = param_grid
       self.n_trials = n_trials
       self.random_state = random_state

    def fit(self, X_train, Y_train, X_test, Y_test):      

        accuracy, accuracy_IN, accuracy_OUT = {}, {}, {}
        search_space = list(product(*[self.param_grid[x] for x in self.param_grid.keys()]))
        random.seed(self.random_state)
        random_search_space = random.sample(search_space, self.n_trials)
        random_search_space = [{x: random_search_space[j][i] for i, x in enumerate(self.param_grid.keys())} for j in range(0, len(random_search_space))]
        
        for i in range(0, self.n_trials):
            face_recognition_knn_pca = FacialRecognitionKnnPca()
            face_recognition_knn_pca.set_params(**random_search_space[i])
            face_recognition_knn_pca.fit(X_train, Y_train)
            Y_test_hat  = face_recognition_knn_pca.predict(X_test)
            # General Accuracy 
            accuracy[tuple(random_search_space[i].values())] = np.round(accuracy_score(y_pred=Y_test_hat, y_true=Y_test), 3) 
            # Accuracy for IN images
            accuracy_IN[tuple(random_search_space[i].values())] = np.round(accuracy_score(y_pred=Y_test_hat[np.where(Y_test != 0)], y_true=Y_test[np.where(Y_test != 0)]), 3) 
            # Accuracy for OUT images
            accuracy_OUT[tuple(random_search_space[i].values())] = np.round(accuracy_score(y_pred=Y_test_hat[np.where(Y_test == 0)], y_true=Y_test[np.where(Y_test == 0)]), 3) 

        combinations = [x for x in accuracy.keys()]
        accuracy_values = np.array([x for x in accuracy.values()])
        accuracy_IN_values = np.array([x for x in accuracy_IN.values()])
        accuracy_OUT_values = np.array([x for x in accuracy_OUT.values()])

        results = pd.DataFrame({'combination': combinations, 'accuracy': accuracy_values, 
                                   'accuracy_IN': accuracy_IN_values, 'accuracy_OUT': accuracy_OUT_values})
        results = results.sort_values(by='accuracy', ascending=False)
        params_keys = [x for x in self.param_grid.keys()]
        results[params_keys] = results['combination'].apply(lambda x: pd.Series(x))

        best_params = dict()
        best_params['n_neighbors'] = results['n_neighbors'].iloc[0]
        best_params['metric'] = results['metric'].iloc[0]
        best_params['n_components'] = results['n_components'].iloc[0]
        best_params['threshold_dist'] = results['threshold_dist'].iloc[0]
        
        self.best_params_ = best_params
        self.best_score_ = results.iloc[0,1]
        self.results = results 

#######################################################################################################################

def FisherLDA(X, Y):

    X_, n, H, C = {}, {}, {}, {}

    for r in np.unique(Y) :
        X_[r] = X[Y==r,:]
        n[r] = len(X_[r])
        H[r] = np.identity(n[r]) - (1/n[r])*np.ones((n[r], n[r]))
        C[r] = X_[r].T @ H[r] @ X_[r]

    Sw = np.sum([C[r] for r in np.unique(Y)], axis=0)
    n = len(X)
    H = np.identity(n) - (1/n)*np.ones((n, n))
    St = X.T @ H @ X
    Sb = St - Sw
    Sw_inv = np.linalg.inv(Sw)
    eigval, eigvec = np.linalg.eig(Sw_inv @ Sb)
    eigval = np.real(eigval)
    eigvec = np.real(eigvec)
    eigval[np.isclose(eigval, 0, atol=0.0001)] = 0 
    eigval_idx_sorted = np.argsort(-eigval)
    g = len(np.unique(Y))
    W = eigvec[:, eigval_idx_sorted[0:g-1]] 
    X_fisher = X @ W  

    return X_fisher, W

#######################################################################################################################

class FacialRecognitionKnnPcaFisher :

    def __init__(self, n_neighbors=3, metric='euclidean', threshold_dist=0, n_components=3):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.threshold_dist = threshold_dist
        self.n_components = n_components

    def set_params(self, **params):
        # Method to set the parameters of the estimator
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def fit(self, X, Y) :

        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0)
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        self.pca = PCA_class(n_components=self.n_components)
        self.pca.fit(X=X_scaled, solver='spectral')
        X_pca = self.pca.transform(X_scaled)
        try:
            X_fisher, self.W = FisherLDA(X_pca, Y)
        except:
            return 'Error: Sw not invertible'

        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric)
        self.knn.fit(X_fisher, Y)          

    def predict(self, X) :

        T = self.pca.T
        X_scaled = (X - self.mu) / self.sigma
        X_pca = X_scaled @ T
        X_fisher = X_pca @ self.W
        # Predicting IN/OUT
        Z_hat = np.repeat(0, len(X_fisher))
        Y_hat = np.repeat(0, len(X_fisher))
        nearest_neigh_dist, nearest_neigh_index = self.knn.kneighbors(X_fisher, n_neighbors=1) 
        for i in range(0, len(X_fisher)):
            if nearest_neigh_dist[i] > self.threshold_dist :
                Z_hat[i] = 0 # 0 = OUT
            else :
                Z_hat[i] = 1 # 1 = IN 
        
        num_images_IN = np.sum(Z_hat == 1)
        if  num_images_IN > 0 :
            Y_hat[Z_hat == 1] = self.knn.predict(X_fisher[Z_hat == 1,:])   
        
        return Y_hat

#######################################################################################################################
    
class RandomSearchFacialRecognitionKnnPcaFisher :
   
    def __init__(self, param_grid, n_trials=10, random_state=123) :

       self.param_grid = param_grid
       self.n_trials = n_trials
       self.random_state = random_state

    def fit(self, X_train, Y_train, X_test, Y_test):      

        accuracy, accuracy_IN, accuracy_OUT = {}, {}, {}
        search_space = list(product(*[self.param_grid[x] for x in self.param_grid.keys()]))
        random.seed(self.random_state)
        random_search_space = random.sample(search_space, self.n_trials)
        random_search_space = [{x: random_search_space[j][i] for i, x in enumerate(self.param_grid.keys())} for j in range(0, len(random_search_space))]
        
        for i in range(0, self.n_trials):
            face_recognition_knn_pca_fisher = FacialRecognitionKnnPcaFisher()
            face_recognition_knn_pca_fisher.set_params(**random_search_space[i])
            face_recognition_knn_pca_fisher.fit(X_train, Y_train)
            Y_test_hat  = face_recognition_knn_pca_fisher.predict(X_test)
            # General Accuracy 
            accuracy[tuple(random_search_space[i].values())] = np.round(accuracy_score(y_pred=Y_test_hat, y_true=Y_test), 3) 
            # Accuracy for IN images
            accuracy_IN[tuple(random_search_space[i].values())] = np.round(accuracy_score(y_pred=Y_test_hat[np.where(Y_test != 0)], y_true=Y_test[np.where(Y_test != 0)]), 3) 
            # Accuracy for OUT images
            accuracy_OUT[tuple(random_search_space[i].values())] = np.round(accuracy_score(y_pred=Y_test_hat[np.where(Y_test == 0)], y_true=Y_test[np.where(Y_test == 0)]), 3) 

        combinations = [x for x in accuracy.keys()]
        accuracy_values = np.array([x for x in accuracy.values()])
        accuracy_IN_values = np.array([x for x in accuracy_IN.values()])
        accuracy_OUT_values = np.array([x for x in accuracy_OUT.values()])

        results = pd.DataFrame({'combination': combinations, 'accuracy': accuracy_values, 
                                   'accuracy_IN': accuracy_IN_values, 'accuracy_OUT': accuracy_OUT_values})
        results = results.sort_values(by='accuracy', ascending=False)
        params_keys = [x for x in self.param_grid.keys()]
        results[params_keys] = results['combination'].apply(lambda x: pd.Series(x))

        best_params = dict()
        best_params['n_neighbors'] = results['n_neighbors'].iloc[0]
        best_params['metric'] = results['metric'].iloc[0]
        best_params['n_components'] = results['n_components'].iloc[0]
        best_params['threshold_dist'] = results['threshold_dist'].iloc[0]
        
        self.best_params_ = best_params
        self.best_score_ = results.iloc[0,1]
        self.results = results 