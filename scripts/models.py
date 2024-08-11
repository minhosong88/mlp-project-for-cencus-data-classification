from scipy.special import expit
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import sys

class TwoLayerPerceptron(object):
    def __init__(self, n_hidden=30,
                 C=0.0, epochs=500, eta=0.001,shuffle=True, minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.n_hidden = n_hidden
        self.l2_C = C
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatches = minibatches
        
    @staticmethod
    def _encode_labels(y):
        """Encode labels into one-hot representation"""
        onehot = pd.get_dummies(y).values.T
            
        return onehot

    def _initialize_weights(self):
        """Initialize weights Glorot and He normalization."""
        init_bound = 4*np.sqrt(6. / (self.n_hidden + self.n_features_))
        W1 = np.random.uniform(-init_bound, init_bound,(self.n_hidden, self.n_features_))

        # reduce the final layer magnitude in order to balance the size of the gradients
        # between 
        init_bound = 4*np.sqrt(6 / (self.n_output_ + self.n_hidden))
        W2 = np.random.uniform(-init_bound, init_bound,(self.n_output_, self.n_hidden)) 
        
        # set these to zero to start so that
        # they do not immediately saturate the neurons
        b1 = np.zeros((self.n_hidden, 1))
        b2 = np.zeros((self.n_output_, 1))
        
        return W1, W2, b1, b2
    
    @staticmethod
    def _sigmoid(z):
        """Use scipy.special.expit to avoid overflow"""
        # 1.0 / (1.0 + np.exp(-z))
        return expit(z)
    
    
    @staticmethod
    def _L2_reg(lambda_, W1, W2):
        """Compute L2-regularization cost"""
        # only compute for non-bias terms
        return (lambda_/2.0) * np.sqrt(np.mean(W1[:, 1:] ** 2) + np.mean(W2[:, 1:] ** 2))
    
    def _cost(self,A3,y,W1,W2):
        '''Get the objective function value'''
        cost = -np.mean(np.nan_to_num((y*np.log(A3+1e-7)+(1-y)*np.log(1-A3+1e-7))))
        L2_term = self._L2_reg(self.l2_C, W1, W2)
        return cost + L2_term
    
    def _feedforward(self, X, W1, W2, b1, b2):
        """Compute feedforward step
        -----------
        X : Input layer with original features.
        W1: Weight matrix for input layer -> hidden layer.
        W2: Weight matrix for hidden layer -> output layer.
        ----------
        a1-a3 : activations into layer (or output layer)
        z1-z2 : layer inputs 

        """
        A1 = X.T
        Z1 = W1 @ A1 + b1
        A2 = self._sigmoid(Z1)
        Z2 = W2 @ A2 + b2
        A3 = self._sigmoid(Z2)
        return A1, Z1, A2, Z2, A3
    
    def fit(self, X, y, print_progress=False, XY_test=None):
        """ Learn weights from training data. With mini-batch"""
        X_data, y_data = X.copy(), y.copy()
        
        # Ensure X_data and y_data are numpy arrays for compatibility with numpy operations
        X_data = X_data.values if isinstance(X_data, pd.DataFrame) else X_data
        y_data = y_data.values if isinstance(y_data, pd.Series) else y_data
               
        # init weights and setup matrices
        self.n_features_ = X_data.shape[1]
        self.n_output_ = np.unique(y_data).shape[0]
        self.W1, self.W2, self.b1, self.b2 = self._initialize_weights()

        self.cost_ = []
        self.score_ = []
        # get starting acc
        self.score_.append(accuracy_score(y_data,self.predict(X_data)))
        # keep track of validation, if given
        if XY_test is not None:
            X_test = XY_test[0].copy()
            y_test = XY_test[1].copy()
            
            X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
            
            self.val_score_ = []
            self.val_score_.append(accuracy_score(y_test,self.predict(X_test)))
            self.val_cost_ = []
            
        for i in range(self.epochs):

            if print_progress>0 and (i+1)%print_progress==0:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx_shuffle = np.random.permutation(y_data.shape[0])
                X_data, y_data = X_data[idx_shuffle], y_data[idx_shuffle]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            mini_cost = []
            for idx in mini:

                # feedforward
                A1, Z1, A2, Z2, A3 = self._feedforward(X_data[idx],
                                                       self.W1,
                                                       self.W2,
                                                       self.b1,
                                                       self.b2
                                                      )
                
                cost = self._cost(A3,y_data[idx],self.W1,self.W2)
                mini_cost.append(cost) # this appends cost of mini-batch only

                # compute gradient via backpropagation
                gradW1, gradW2, gradb1, gradb2 = self._get_gradient(A1=A1, A2=A2, A3=A3, Z1=Z1, Z2=Z2, 
                                                  y=y_data[idx],
                                                  W1=self.W1,W2=self.W2)
                # Update weights
                self.W1 -= self.eta * gradW1
                self.W2 -= self.eta * gradW2

                # Update biases
                self.b1 -= self.eta * gradb1
                self.b2 -= self.eta * gradb2

            self.cost_.append(np.mean(mini_cost))
            self.score_.append(accuracy_score(y_data,self.predict(X_data)))
            
            # update if a validation set was provided
            if XY_test is not None:
                yhat = self.predict(X_test)
                self.val_score_.append(accuracy_score(y_test,yhat))
            
        return self    
    
    def _get_gradient(self, A1, A2, A3, Z1, Z2, y, W1, W2):
        """ Compute gradient step using backpropagation.
        """
        # vectorized backpropagation
        V2 = (A3-y) # <- this is only line that changed
        V1 = A2*(1-A2)*(W2.T @ V2)
        
        gradW2 = V2 @ A2.T
        gradW1 = V1 @ A1.T
        
        gradb2 = np.sum(V2, axis=1).reshape((-1,1))
        gradb1 = np.sum(V1, axis=1).reshape((-1,1))
        
        # regularize weights that are not bias terms
        gradW1 += W1 * self.l2_C
        gradW2 += W2 * self.l2_C

        return gradW1, gradW2, gradb1, gradb2
    
    def predict(self, X):
        """Predict class labels"""
        _, _, _, _, A3 = self._feedforward(X, self.W1, self.W2, self.b1, self.b2)
        y_pred = np.argmax(A3, axis=0)
        return y_pred

# One-Hot-Coding
class TwoLayerPerceptron_ohc(object):
    def __init__(self, n_hidden=30,
                 C=0.0, epochs=500, eta=0.001, random_state=None, shuffle = True, minibatches=1):
        np.random.seed(random_state)
        self.n_hidden = n_hidden
        self.l2_C = C
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatches = minibatches
        
    @staticmethod
    def _encode_labels(y):
        """Encode labels into one-hot representation"""
        onehot = pd.get_dummies(y).values.T
            
        return onehot

    def _initialize_weights(self):
        """Initialize weights Glorot and He normalization."""
        init_bound = 4*np.sqrt(6. / (self.n_hidden + self.n_features_))
        W1 = np.random.uniform(-init_bound, init_bound,(self.n_hidden, self.n_features_))

        # reduce the final layer magnitude in order to balance the size of the gradients
        # between 
        init_bound = 4*np.sqrt(6 / (self.n_output_ + self.n_hidden))
        W2 = np.random.uniform(-init_bound, init_bound,(self.n_output_, self.n_hidden)) 
        
        # set these to zero to start so that
        # they do not immediately saturate the neurons
        b1 = np.zeros((self.n_hidden, 1))
        b2 = np.zeros((self.n_output_, 1))
        
        return W1, W2, b1, b2
    
    @staticmethod
    def _sigmoid(z):
        """Use scipy.special.expit to avoid overflow"""
        # 1.0 / (1.0 + np.exp(-z))
        return expit(z)
    
    
    @staticmethod
    def _L2_reg(lambda_, W1, W2):
        """Compute L2-regularization cost"""
        # only compute for non-bias terms
        return (lambda_/2.0) * np.sqrt(np.mean(W1[:, 1:] ** 2) + np.mean(W2[:, 1:] ** 2))
    
    def _cost(self,A3,Y_enc,W1,W2):
        '''Get the objective function value'''
        cost = -np.mean(np.nan_to_num((Y_enc*np.log(A3+1e-7)+(1-Y_enc)*np.log(1-A3+1e-7))))
        L2_term = self._L2_reg(self.l2_C, W1, W2)
        return cost + L2_term
    
    def _feedforward(self, X, W1, W2, b1, b2):
        """Compute feedforward step
        -----------
        X : Input layer with original features.
        W1: Weight matrix for input layer -> hidden layer.
        W2: Weight matrix for hidden layer -> output layer.
        ----------
        a1-a3 : activations into layer (or output layer)
        z1-z2 : layer inputs 

        """
        A1 = X.T
        Z1 = W1 @ A1 + b1
        A2 = self._sigmoid(Z1)
        Z2 = W2 @ A2 + b2
        A3 = self._sigmoid(Z2)
        return A1, Z1, A2, Z2, A3
    
    def fit(self, X, y, print_progress=False, XY_test=None):
        """ Learn weights from training data. With mini-batch"""
        X_data, y_data = X.copy(), y.copy()
        Y_enc = self._encode_labels(y)
        
        # Ensure X_data and y_data are numpy arrays for compatibility with numpy operations
        X_data = X_data.values if isinstance(X_data, pd.DataFrame) else X_data
        y_data = y_data.values if isinstance(y_data, pd.Series) else y_data
        
        # init weights and setup matrices
        self.n_features_ = X_data.shape[1]
        self.n_output_ = Y_enc.shape[0]
        self.W1, self.W2, self.b1, self.b2 = self._initialize_weights()

        self.cost_ = []
        self.score_ = []
        # get starting acc
        self.score_.append(accuracy_score(y_data,self.predict(X_data)))
        # keep track of validation, if given
        if XY_test is not None:
            X_test = XY_test[0].copy()
            y_test = XY_test[1].copy()
            
            X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
            
            self.val_score_ = []
            self.val_score_.append(accuracy_score(y_test,self.predict(X_test)))
            self.val_cost_ = []
            
        for i in range(self.epochs):


            if print_progress>0 and (i+1)%print_progress==0:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx_shuffle = np.random.permutation(y_data.shape[0])
                X_data, Y_enc, y_data = X_data[idx_shuffle], Y_enc[:, idx_shuffle], y_data[idx_shuffle]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            mini_cost = []
            for idx in mini:

                # feedforward
                A1, Z1, A2, Z2, A3 = self._feedforward(X_data[idx],
                                                       self.W1,
                                                       self.W2,
                                                       self.b1,
                                                       self.b2
                                                      )
                
                cost = self._cost(A3,Y_enc[:, idx],self.W1,self.W2)
                mini_cost.append(cost) # this appends cost of mini-batch only

                # compute gradient via backpropagation
                gradW1, gradW2, gradb1, gradb2 = self._get_gradient(A1=A1, A2=A2, A3=A3, Z1=Z1, Z2=Z2, 
                                                  Y_enc=Y_enc[:, idx],
                                                  W1=self.W1,W2=self.W2)
                # Update weights
                self.W1 -= self.eta * gradW1
                self.W2 -= self.eta * gradW2

                # Update biases
                self.b1 -= self.eta * gradb1
                self.b2 -= self.eta * gradb2
                
            self.cost_.append(np.mean(mini_cost))
            self.score_.append(accuracy_score(y_data,self.predict(X_data)))
            
            # update if a validation set was provided
            if XY_test is not None:
                yhat = self.predict(X_test)
                self.val_score_.append(accuracy_score(y_test,yhat))
            
        return self
    
    
    def _get_gradient(self, A1, A2, A3, Z1, Z2, Y_enc, W1, W2):
        """ Compute gradient step using backpropagation.
        """
        # vectorized backpropagation
        V2 = (A3-Y_enc) # <- this is only line that changed
        V1 = A2*(1-A2)*(W2.T @ V2)
        
        gradW2 = V2 @ A2.T
        gradW1 = V1 @ A1.T
        
        gradb2 = np.sum(V2, axis=1).reshape((-1,1))
        gradb1 = np.sum(V1, axis=1).reshape((-1,1))
        
        # regularize weights that are not bias terms
        gradW1 += W1 * self.l2_C
        gradW2 += W2 * self.l2_C

        return gradW1, gradW2, gradb1, gradb2
    
    def predict(self, X):
        """Predict class labels"""
        _, _, _, _, A3 = self._feedforward(X, self.W1, self.W2, self.b1, self.b2)
        y_pred = np.argmax(A3, axis=0)
        return y_pred
    
class ThreeLayerPerceptron(object):
    def __init__(self, n_hidden=30,
                 C=0.0, epochs=500, eta=0.001, random_state=None, shuffle = True, minibatches=1):
        np.random.seed(random_state)
        self.n_hidden1 = n_hidden
        self.n_hidden2 = n_hidden
        self.l2_C = C
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatches = minibatches
        self.grad_magnitudes = {'W1': [], 'W2': [], 'W3': []}
        
    @staticmethod
    def _encode_labels(y):
        """Encode labels into one-hot representation"""
        onehot = pd.get_dummies(y).values.T
            
        return onehot

    def _initialize_weights(self):
        """Initialize weights Glorot and He normalization."""
        init_bound = 4*np.sqrt(6. / (self.n_hidden1 + self.n_features_))
        W1 = np.random.uniform(-init_bound, init_bound,(self.n_hidden1, self.n_features_))
 
        init_bound = 4*np.sqrt(6 / (self.n_hidden2 + self.n_hidden1))
        W2 = np.random.uniform(-init_bound, init_bound,(self.n_hidden2, self.n_hidden1)) 
        
        init_bound = 4*np.sqrt(6 / (self.n_output_  + self.n_hidden2))
        W3 = np.random.uniform(-init_bound, init_bound,(self.n_output_, self.n_hidden2))         
        

        b1 = np.zeros((self.n_hidden1, 1))
        b2 = np.zeros((self.n_hidden2, 1))
        b3 = np.zeros((self.n_output_, 1))
        
        
        return W1, W2, W3, b1, b2, b3
    
    @staticmethod
    def _sigmoid(z):
        """Use scipy.special.expit to avoid overflow"""
        # 1.0 / (1.0 + np.exp(-z))
        return expit(z)
    
    
    @staticmethod
    def _L2_reg(lambda_, W1, W2, W3):
        """Compute L2-regularization cost"""
        # only compute for non-bias terms
        return (lambda_/2.0) * np.sqrt(np.mean(W1[:, 1:] ** 2) + np.mean(W2[:, 1:] ** 2) + np.mean(W3[:, 1:] ** 2))
    
    def _cost(self,A4,Y_enc,W1,W2,W3):
        '''Get the objective function value'''
        cost = -np.mean(np.nan_to_num((Y_enc*np.log(A4+1e-7)+(1-Y_enc)*np.log(1-A4+1e-7))))
        L2_term = self._L2_reg(self.l2_C, W1, W2, W3)
        return cost + L2_term
    
    def _feedforward(self, X, W1, W2, W3, b1, b2, b3):
 
        A1 = X.T
        Z1 = W1 @ A1 + b1
        A2 = self._sigmoid(Z1)
        Z2 = W2 @ A2 + b2
        A3 = self._sigmoid(Z2)
        Z3 = W3 @ A3 + b3
        A4 = self._sigmoid(Z3)
        return A1, Z1, A2, Z2, A3, Z3, A4
    
    def fit(self, X, y, print_progress=False, XY_test=None):
        """ Learn weights from training data. With mini-batch"""
        X_data, y_data = X.copy(), y.copy()
        Y_enc = self._encode_labels(y)
        
        # Ensure X_data and y_data are numpy arrays for compatibility with numpy operations
        X_data = X_data.values if isinstance(X_data, pd.DataFrame) else X_data
        y_data = y_data.values if isinstance(y_data, pd.Series) else y_data
        
        # init weights and setup matrices
        self.n_features_ = X_data.shape[1]
        self.n_output_ = Y_enc.shape[0]
        self.W1, self.W2, self.W3, self.b1, self.b2, self.b3 = self._initialize_weights()

        self.cost_ = []
        self.score_ = []
        # get starting acc
        self.score_.append(accuracy_score(y_data,self.predict(X_data)))
        # keep track of validation, if given
        if XY_test is not None:
            X_test = XY_test[0].copy()
            y_test = XY_test[1].copy()
            
            X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
            
            self.val_score_ = []
            self.val_score_.append(accuracy_score(y_test,self.predict(X_test)))
            self.val_cost_ = []
            
        for i in range(self.epochs):


            if print_progress>0 and (i+1)%print_progress==0:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx_shuffle = np.random.permutation(y_data.shape[0])
                X_data, Y_enc, y_data = X_data[idx_shuffle], Y_enc[:, idx_shuffle], y_data[idx_shuffle]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            mini_cost = []
            for idx in mini:

                # feedforward
                A1, Z1, A2, Z2, A3, Z3, A4 = self._feedforward(X_data[idx], self.W1, self.W2, self.W3, self.b1, self.b2, self.b3)
                
                cost = self._cost(A4, Y_enc[:, idx], self.W1, self.W2, self.W3)
                mini_cost.append(cost) # this appends cost of mini-batch only

                # compute gradient via backpropagation
                gradW1, gradW2, gradW3, gradb1, gradb2, gradb3 = self._get_gradient(A1, A2, A3, A4, Z1, Z2, Z3, Y_enc[:, idx],
                                                  W1=self.W1,W2=self.W2, W3=self.W3)
                # Update weights
                self.W1 -= self.eta * gradW1
                self.W2 -= self.eta * gradW2
                self.W3 -= self.eta * gradW3

                # Update biases
                self.b1 -= self.eta * gradb1
                self.b2 -= self.eta * gradb2
                self.b3 -= self.eta * gradb3
                # store gradient magnitude with average absolute values
                self.grad_magnitudes['W1'].append(np.mean(np.abs(gradW1)))
                self.grad_magnitudes['W2'].append(np.mean(np.abs(gradW2)))
                self.grad_magnitudes['W3'].append(np.mean(np.abs(gradW3)))
                
            self.cost_.append(np.mean(mini_cost))
            self.score_.append(accuracy_score(y_data,self.predict(X_data)))
 
            # update if a validation set was provided
            if XY_test is not None:
                yhat = self.predict(X_test)
                self.val_score_.append(accuracy_score(y_test,yhat))
            
        return self
    
    
    def _get_gradient(self, A1, A2, A3, A4, Z1, Z2, Z3, Y_enc, W1, W2, W3):
        """ Compute gradient step using backpropagation.
        """
        # vectorized backpropagation
        V3 = (A4 - Y_enc)
        V2 = A3 * (1 - A3) * (self.W3.T @ V3) 
        V1 = A2 * (1 - A2) * (self.W2.T @ V2)
        
        gradW3 = V3 @ A3.T
        gradW2 = V2 @ A2.T
        gradW1 = V1 @ A1.T
        
        gradb3 = np.sum(V3, axis=1).reshape((-1, 1))
        gradb2 = np.sum(V2, axis=1).reshape((-1,1))
        gradb1 = np.sum(V1, axis=1).reshape((-1,1))
        
        # regularize weights that are not bias terms
        gradW1 += W1 * self.l2_C
        gradW2 += W2 * self.l2_C
        gradW3 += W3 * self.l2_C
        

        return gradW1, gradW2, gradW3, gradb1, gradb2, gradb3
    
    def predict(self, X):
        """Predict class labels"""
        _, _, _, _, _, _, A4 = self._feedforward(X, self.W1, self.W2, self.W3, self.b1, self.b2, self.b3)
        y_pred = np.argmax(A4, axis=0)
        return y_pred

class FourLayerPerceptron(object):
    def __init__(self, n_hidden=30,
                 C=0.0, epochs=500, eta=0.001, random_state=None, shuffle = True, minibatches=1):
        np.random.seed(random_state)
        self.n_hidden1 = n_hidden
        self.n_hidden2 = n_hidden
        self.n_hidden3 = n_hidden
        self.l2_C = C
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatches = minibatches
        self.grad_magnitudes = {'W1': [], 'W2': [], 'W3': [],'W4': [] }
        
    @staticmethod
    def _encode_labels(y):
        """Encode labels into one-hot representation"""
        onehot = pd.get_dummies(y).values.T
            
        return onehot

    def _initialize_weights(self):
        """Initialize weights Glorot and He normalization."""
        init_bound = 4*np.sqrt(6. / (self.n_hidden1 + self.n_features_))
        W1 = np.random.uniform(-init_bound, init_bound,(self.n_hidden1, self.n_features_))
 
        init_bound = 4*np.sqrt(6 / (self.n_hidden1 + self.n_hidden2))
        W2 = np.random.uniform(-init_bound, init_bound,(self.n_hidden2, self.n_hidden1)) 
        
        init_bound = 4*np.sqrt(6 / (self.n_hidden2  + self.n_hidden3))
        W3 = np.random.uniform(-init_bound, init_bound,(self.n_hidden3, self.n_hidden2))
        
        init_bound = 4*np.sqrt(6 / (self.n_hidden3  + self.n_output_))
        W4 = np.random.uniform(-init_bound, init_bound,(self.n_output_, self.n_hidden3))           
        

        b1 = np.zeros((self.n_hidden1, 1))
        b2 = np.zeros((self.n_hidden2, 1))
        b3 = np.zeros((self.n_hidden3, 1))
        b4 = np.zeros((self.n_output_, 1))

        
        
        return W1, W2, W3, W4, b1, b2, b3, b4
    
    @staticmethod
    def _sigmoid(z):
        """Use scipy.special.expit to avoid overflow"""
        # 1.0 / (1.0 + np.exp(-z))
        return expit(z)
    
    
    @staticmethod
    def _L2_reg(lambda_, W1, W2, W3, W4):
        """Compute L2-regularization cost"""
        # only compute for non-bias terms
        return (lambda_/2.0) * np.sqrt(np.mean(W1[:, 1:] ** 2) + np.mean(W2[:, 1:] ** 2) + np.mean(W3[:, 1:] ** 2)+ np.mean(W4[:, 1:] ** 2))
    
    def _cost(self,A5,Y_enc,W1,W2,W3,W4):
        '''Get the objective function value'''
        cost = -np.mean(np.nan_to_num((Y_enc*np.log(A5+1e-7)+(1-Y_enc)*np.log(1-A5+1e-7))))
        L2_term = self._L2_reg(self.l2_C, W1, W2, W3, W4)
        return cost + L2_term
    
    def _feedforward(self, X, W1, W2, W3, W4, b1, b2, b3, b4):

        A1 = X.T
        Z1 = W1 @ A1 + b1
        A2 = self._sigmoid(Z1)
        Z2 = W2 @ A2 + b2
        A3 = self._sigmoid(Z2)
        Z3 = W3 @ A3 + b3
        A4 = self._sigmoid(Z3)
        Z4 = W4 @ A4 + b4
        A5 = self._sigmoid(Z4)
        
        return A1, Z1, A2, Z2, A3, Z3, A4, Z4, A5
    
    def fit(self, X, y, print_progress=False, XY_test=None):
        """ Learn weights from training data. With mini-batch"""
        X_data, y_data = X.copy(), y.copy()
        Y_enc = self._encode_labels(y)
        
        # Ensure X_data and y_data are numpy arrays for compatibility with numpy operations
        X_data = X_data.values if isinstance(X_data, pd.DataFrame) else X_data
        y_data = y_data.values if isinstance(y_data, pd.Series) else y_data
        
        # init weights and setup matrices
        self.n_features_ = X_data.shape[1]
        self.n_output_ = Y_enc.shape[0]
        self.W1, self.W2, self.W3, self.W4, self.b1, self.b2, self.b3, self.b4 = self._initialize_weights()

        self.cost_ = []
        self.score_ = []
        # get starting acc
        self.score_.append(accuracy_score(y_data,self.predict(X_data)))
        # keep track of validation, if given
        if XY_test is not None:
            X_test = XY_test[0].copy()
            y_test = XY_test[1].copy()
            
            X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
            
            self.val_score_ = []
            self.val_score_.append(accuracy_score(y_test,self.predict(X_test)))
            self.val_cost_ = []
            
        for i in range(self.epochs):


            if print_progress>0 and (i+1)%print_progress==0:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx_shuffle = np.random.permutation(y_data.shape[0])
                X_data, Y_enc, y_data = X_data[idx_shuffle], Y_enc[:, idx_shuffle], y_data[idx_shuffle]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            mini_cost = []
            for idx in mini:

                # feedforward
                A1, Z1, A2, Z2, A3, Z3, A4, Z4, A5 = self._feedforward(X_data[idx], self.W1, self.W2, self.W3, self.W4, self.b1, self.b2, self.b3, self.b4)
                
                cost = self._cost(A5, Y_enc[:, idx], self.W1, self.W2, self.W3, self.W4)
                mini_cost.append(cost) # this appends cost of mini-batch only

                # compute gradient via backpropagation
                gradW1, gradW2, gradW3, gradW4, gradb1, gradb2, gradb3, gradb4 = self._get_gradient(A1, A2, A3, A4, A5, Z1, Z2, Z3, Z4, Y_enc[:,idx], W1=self.W1, W2=self.W2, W3=self.W3, W4=self.W4 )
                
                # Update weights
                self.W1 -= self.eta * gradW1
                self.W2 -= self.eta * gradW2
                self.W3 -= self.eta * gradW3
                self.W4 -= self.eta * gradW4

                # Update biases
                self.b1 -= self.eta * gradb1
                self.b2 -= self.eta * gradb2
                self.b3 -= self.eta * gradb3
                self.b4 -= self.eta * gradb4
                
                # store gradient magnitude with average absolute values
                self.grad_magnitudes['W1'].append(np.mean(np.abs(gradW1)))
                self.grad_magnitudes['W2'].append(np.mean(np.abs(gradW2)))
                self.grad_magnitudes['W3'].append(np.mean(np.abs(gradW3)))
                self.grad_magnitudes['W4'].append(np.mean(np.abs(gradW4)))
                
            self.cost_.append(np.mean(mini_cost))
            self.score_.append(accuracy_score(y_data,self.predict(X_data)))
 
            # update if a validation set was provided
            if XY_test is not None:
                yhat = self.predict(X_test)
                self.val_score_.append(accuracy_score(y_test,yhat))
            
        return self
    
    
    def _get_gradient(self, A1, A2, A3, A4, A5, Z1, Z2, Z3, Z4, Y_enc, W1, W2, W3, W4):
        """ Compute gradient step using backpropagation.
        """
        # vectorized backpropagation
        V4 = (A5 - Y_enc)
        V3 = A4 * (1 - A4) * (self.W4.T @ V4) 
        V2 = A3 * (1 - A3) * (self.W3.T @ V3) 
        V1 = A2 * (1 - A2) * (self.W2.T @ V2)
        
        gradW4 = V4 @ A4.T
        gradW3 = V3 @ A3.T
        gradW2 = V2 @ A2.T
        gradW1 = V1 @ A1.T
        
        gradb4 = np.sum(V4, axis=1).reshape((-1, 1))
        gradb3 = np.sum(V3, axis=1).reshape((-1, 1))
        gradb2 = np.sum(V2, axis=1).reshape((-1,1))
        gradb1 = np.sum(V1, axis=1).reshape((-1,1))
        
        # regularize weights that are not bias terms
        gradW1 += W1 * self.l2_C
        gradW2 += W2 * self.l2_C
        gradW3 += W3 * self.l2_C
        gradW4 += W4 * self.l2_C

        return gradW1, gradW2, gradW3, gradW4, gradb1, gradb2, gradb3, gradb4
    
    def predict(self, X):
        """Predict class labels"""
        _, _, _, _, _, _, _, _, A5 = self._feedforward(X, self.W1, self.W2, self.W3, self.W4, self.b1, self.b2, self.b3, self.b4)
        y_pred = np.argmax(A5, axis=0)
        return y_pred


# Start from this 
class FiveLayerPerceptron(object):
    def __init__(self, n_hidden=30,
                 C=0.0, epochs=500, eta=0.001, random_state=None, shuffle = True, minibatches=1):
        np.random.seed(random_state)
        self.n_hidden1 = n_hidden
        self.n_hidden2 = n_hidden
        self.n_hidden3 = n_hidden
        self.n_hidden4 = n_hidden
        self.l2_C = C
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatches = minibatches
        self.grad_magnitudes = {'W1': [], 'W2': [], 'W3': [],'W4': [], 'W5': [] }
        
    @staticmethod
    def _encode_labels(y):
        """Encode labels into one-hot representation"""
        onehot = pd.get_dummies(y).values.T
            
        return onehot

    def _initialize_weights(self):
        """Initialize weights Glorot and He normalization."""
        init_bound = 4*np.sqrt(6. / (self.n_hidden1 + self.n_features_))
        W1 = np.random.uniform(-init_bound, init_bound,(self.n_hidden1, self.n_features_))
 
        init_bound = 4*np.sqrt(6 / (self.n_hidden1 + self.n_hidden2))
        W2 = np.random.uniform(-init_bound, init_bound,(self.n_hidden2, self.n_hidden1)) 
        
        init_bound = 4*np.sqrt(6 / (self.n_hidden2  + self.n_hidden3))
        W3 = np.random.uniform(-init_bound, init_bound,(self.n_hidden3, self.n_hidden2))
        
        init_bound = 4*np.sqrt(6 / (self.n_hidden3  + self.n_hidden4))
        W4 = np.random.uniform(-init_bound, init_bound,(self.n_hidden4, self.n_hidden3))           
        
        init_bound = 4*np.sqrt(6 / (self.n_hidden4  + self.n_output_))
        W5 = np.random.uniform(-init_bound, init_bound,(self.n_output_, self.n_hidden4))

        b1 = np.zeros((self.n_hidden1, 1))
        b2 = np.zeros((self.n_hidden2, 1))
        b3 = np.zeros((self.n_hidden3, 1))
        b4 = np.zeros((self.n_hidden4, 1))
        b5 = np.zeros((self.n_output_, 1))
        
        
        return W1, W2, W3, W4, W5, b1, b2, b3, b4, b5
    
    @staticmethod
    def _sigmoid(z):
        """Use scipy.special.expit to avoid overflow"""
        # 1.0 / (1.0 + np.exp(-z))
        return expit(z)
    
    
    @staticmethod
    def _L2_reg(lambda_, W1, W2, W3, W4, W5):
        """Compute L2-regularization cost"""
        # only compute for non-bias terms
        return (lambda_/2.0) * np.sqrt(np.mean(W1[:, 1:] ** 2) + np.mean(W2[:, 1:] ** 2) + np.mean(W3[:, 1:] ** 2)+ np.mean(W4[:, 1:] ** 2)+np.mean(W5[:, 1:] ** 2))
    
    def _cost(self,A6,Y_enc,W1,W2,W3,W4,W5):
        '''Get the objective function value'''
        cost = -np.mean(np.nan_to_num((Y_enc*np.log(A6+1e-7)+(1-Y_enc)*np.log(1-A6+1e-7))))
        L2_term = self._L2_reg(self.l2_C, W1, W2, W3, W4, W5)
        return cost + L2_term
    
    def _feedforward(self, X, W1, W2, W3, W4, W5, b1, b2, b3, b4, b5):

        A1 = X.T
        Z1 = W1 @ A1 + b1
        A2 = self._sigmoid(Z1)
        Z2 = W2 @ A2 + b2
        A3 = self._sigmoid(Z2)
        Z3 = W3 @ A3 + b3
        A4 = self._sigmoid(Z3)
        Z4 = W4 @ A4 + b4
        A5 = self._sigmoid(Z4)
        Z5 = W5 @ A5 + b5
        A6 = self._sigmoid(Z5)
        
        return A1, Z1, A2, Z2, A3, Z3, A4, Z4, A5, Z5, A6
    
    def fit(self, X, y, print_progress=False, XY_test=None):
        """ Learn weights from training data. With mini-batch"""
        X_data, y_data = X.copy(), y.copy()
        Y_enc = self._encode_labels(y)
        
        # Ensure X_data and y_data are numpy arrays for compatibility with numpy operations
        X_data = X_data.values if isinstance(X_data, pd.DataFrame) else X_data
        y_data = y_data.values if isinstance(y_data, pd.Series) else y_data
        
        # init weights and setup matrices
        self.n_features_ = X_data.shape[1]
        self.n_output_ = Y_enc.shape[0]
        self.W1, self.W2, self.W3, self.W4, self.W5, self.b1, self.b2, self.b3, self.b4, self.b5 = self._initialize_weights()

        self.cost_ = []
        self.score_ = []
        # get starting acc
        self.score_.append(accuracy_score(y_data,self.predict(X_data)))
        # keep track of validation, if given
        if XY_test is not None:
            X_test = XY_test[0].copy()
            y_test = XY_test[1].copy()
            
            X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
            
            self.val_score_ = []
            self.val_score_.append(accuracy_score(y_test,self.predict(X_test)))
            self.val_cost_ = []
            
        for i in range(self.epochs):


            if print_progress>0 and (i+1)%print_progress==0:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx_shuffle = np.random.permutation(y_data.shape[0])
                X_data, Y_enc, y_data = X_data[idx_shuffle], Y_enc[:, idx_shuffle], y_data[idx_shuffle]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            mini_cost = []
            for idx in mini:

                # feedforward
                A1, Z1, A2, Z2, A3, Z3, A4, Z4, A5, Z5, A6 = self._feedforward(X_data[idx], self.W1, self.W2, self.W3, self.W4, self.W5, self.b1, self.b2, self.b3, self.b4, self.b5)
                
                cost = self._cost(A6, Y_enc[:, idx], self.W1, self.W2, self.W3, self.W4, self.W5)
                mini_cost.append(cost) # this appends cost of mini-batch only

                # compute gradient via backpropagation
                gradW1, gradW2, gradW3, gradW4,gradW5, gradb1, gradb2, gradb3, gradb4, gradb5 = self._get_gradient(A1, A2, A3, A4, A5, A6, Z1, Z2, Z3, Z4, Z5, Y_enc[:,idx], W1=self.W1, W2=self.W2, W3=self.W3, W4=self.W4, W5=self.W5 )
                
                # Update weights
                self.W1 -= self.eta * gradW1
                self.W2 -= self.eta * gradW2
                self.W3 -= self.eta * gradW3
                self.W4 -= self.eta * gradW4
                self.W5 -= self.eta * gradW5

                # Update biases
                self.b1 -= self.eta * gradb1
                self.b2 -= self.eta * gradb2
                self.b3 -= self.eta * gradb3
                self.b4 -= self.eta * gradb4
                self.b5 -= self.eta * gradb5
                
                # store gradient magnitude with average absolute values
                self.grad_magnitudes['W1'].append(np.mean(np.abs(gradW1)))
                self.grad_magnitudes['W2'].append(np.mean(np.abs(gradW2)))
                self.grad_magnitudes['W3'].append(np.mean(np.abs(gradW3)))
                self.grad_magnitudes['W4'].append(np.mean(np.abs(gradW4)))
                self.grad_magnitudes['W5'].append(np.mean(np.abs(gradW5)))

                
            self.cost_.append(np.mean(mini_cost))
            self.score_.append(accuracy_score(y_data,self.predict(X_data)))
 
            # update if a validation set was provided
            if XY_test is not None:
                yhat = self.predict(X_test)
                self.val_score_.append(accuracy_score(y_test,yhat))
            
        return self
    
    
    def _get_gradient(self, A1, A2, A3, A4, A5, A6, Z1, Z2, Z3, Z4, Z5, Y_enc, W1, W2, W3, W4, W5):
        """ Compute gradient step using backpropagation.
        """
        # vectorized backpropagation
        V5 = (A6 - Y_enc)
        V4 = A5 * (1 - A5) * (self.W5.T @ V5)
        V3 = A4 * (1 - A4) * (self.W4.T @ V4) 
        V2 = A3 * (1 - A3) * (self.W3.T @ V3) 
        V1 = A2 * (1 - A2) * (self.W2.T @ V2)
        
        gradW5 = V5 @ A5.T
        gradW4 = V4 @ A4.T
        gradW3 = V3 @ A3.T
        gradW2 = V2 @ A2.T
        gradW1 = V1 @ A1.T
        
        gradb5 = np.sum(V5, axis=1).reshape((-1, 1))
        gradb4 = np.sum(V4, axis=1).reshape((-1, 1))
        gradb3 = np.sum(V3, axis=1).reshape((-1, 1))
        gradb2 = np.sum(V2, axis=1).reshape((-1 ,1))
        gradb1 = np.sum(V1, axis=1).reshape((-1, 1))
        
        # regularize weights that are not bias terms
        gradW1 += W1 * self.l2_C
        gradW2 += W2 * self.l2_C
        gradW3 += W3 * self.l2_C
        gradW4 += W4 * self.l2_C
        gradW5 += W5 * self.l2_C
        
        return gradW1, gradW2, gradW3, gradW4, gradW5, gradb1, gradb2, gradb3, gradb4, gradb5
    
    def predict(self, X):
        """Predict class labels"""
        _, _, _, _, _, _, _, _, _, _, A6 = self._feedforward(X, self.W1, self.W2, self.W3, self.W4, self.W5, self.b1, self.b2, self.b3, self.b4, self.b5)
        y_pred = np.argmax(A6, axis=0)
        return y_pred

# Start from this 
class FLPRMSProp(object):
    def __init__(self, n_hidden=30,
                 C=0.0, epochs=500, eta=0.001, random_state=None, shuffle = True, minibatches=1, gamma = 0.9):
        np.random.seed(random_state)
        self.n_hidden1 = n_hidden
        self.n_hidden2 = n_hidden
        self.n_hidden3 = n_hidden
        self.n_hidden4 = n_hidden
        self.l2_C = C
        self.epochs = epochs
        self.gamma = gamma
        self.eta = eta
        self.shuffle = shuffle
        self.minibatches = minibatches
        self.grad_magnitudes = {'W1': [], 'W2': [], 'W3': [],'W4': [], 'W5': [] }
        
    @staticmethod
    def _encode_labels(y):
        """Encode labels into one-hot representation"""
        onehot = pd.get_dummies(y).values.T
            
        return onehot

    def _initialize_weights(self):
        """Initialize weights Glorot and He normalization."""
        init_bound = 4*np.sqrt(6. / (self.n_hidden1 + self.n_features_))
        W1 = np.random.uniform(-init_bound, init_bound,(self.n_hidden1, self.n_features_))
 
        init_bound = 4*np.sqrt(6 / (self.n_hidden1 + self.n_hidden2))
        W2 = np.random.uniform(-init_bound, init_bound,(self.n_hidden2, self.n_hidden1)) 
        
        init_bound = 4*np.sqrt(6 / (self.n_hidden2  + self.n_hidden3))
        W3 = np.random.uniform(-init_bound, init_bound,(self.n_hidden3, self.n_hidden2))
        
        init_bound = 4*np.sqrt(6 / (self.n_hidden3  + self.n_hidden4))
        W4 = np.random.uniform(-init_bound, init_bound,(self.n_hidden4, self.n_hidden3))           
        
        init_bound = 4*np.sqrt(6 / (self.n_hidden4  + self.n_output_))
        W5 = np.random.uniform(-init_bound, init_bound,(self.n_output_, self.n_hidden4))

        b1 = np.zeros((self.n_hidden1, 1))
        b2 = np.zeros((self.n_hidden2, 1))
        b3 = np.zeros((self.n_hidden3, 1))
        b4 = np.zeros((self.n_hidden4, 1))
        b5 = np.zeros((self.n_output_, 1))
        
        
        return W1, W2, W3, W4, W5, b1, b2, b3, b4, b5
    
    @staticmethod
    def _sigmoid(z):
        """Use scipy.special.expit to avoid overflow"""
        # 1.0 / (1.0 + np.exp(-z))
        return expit(z)
    
    
    @staticmethod
    def _L2_reg(lambda_, W1, W2, W3, W4, W5):
        """Compute L2-regularization cost"""
        # only compute for non-bias terms
        return (lambda_/2.0) * np.sqrt(np.mean(W1[:, 1:] ** 2) + np.mean(W2[:, 1:] ** 2) + np.mean(W3[:, 1:] ** 2)+ np.mean(W4[:, 1:] ** 2)+np.mean(W5[:, 1:] ** 2))
    
    def _cost(self,A6,Y_enc,W1,W2,W3,W4,W5):
        '''Get the objective function value'''
        cost = -np.mean(np.nan_to_num((Y_enc*np.log(A6+1e-7)+(1-Y_enc)*np.log(1-A6+1e-7))))
        L2_term = self._L2_reg(self.l2_C, W1, W2, W3, W4, W5)
        return cost + L2_term
    
    def _feedforward(self, X, W1, W2, W3, W4, W5, b1, b2, b3, b4, b5):

        A1 = X.T
        Z1 = W1 @ A1 + b1
        A2 = self._sigmoid(Z1)
        Z2 = W2 @ A2 + b2
        A3 = self._sigmoid(Z2)
        Z3 = W3 @ A3 + b3
        A4 = self._sigmoid(Z3)
        Z4 = W4 @ A4 + b4
        A5 = self._sigmoid(Z4)
        Z5 = W5 @ A5 + b5
        A6 = self._sigmoid(Z5)
        
        return A1, Z1, A2, Z2, A3, Z3, A4, Z4, A5, Z5, A6
    
    def fit(self, X, y, print_progress=False, XY_test=None):
        """ Learn weights from training data. With mini-batch"""
        X_data, y_data = X.copy(), y.copy()
        Y_enc = self._encode_labels(y)
        
        # Ensure X_data and y_data are numpy arrays for compatibility with numpy operations
        X_data = X_data.values if isinstance(X_data, pd.DataFrame) else X_data
        y_data = y_data.values if isinstance(y_data, pd.Series) else y_data
        
        # init weights and setup matrices
        self.n_features_ = X_data.shape[1]
        self.n_output_ = Y_enc.shape[0]
        self.W1, self.W2, self.W3, self.W4, self.W5, self.b1, self.b2, self.b3, self.b4, self.b5 = self._initialize_weights()
        
        # Initialize accumulators Vk for RMSProp
        self.V1 = np.zeros_like(self.W1)
        self.V2 = np.zeros_like(self.W2)
        self.V3 = np.zeros_like(self.W3)
        self.V4 = np.zeros_like(self.W4)
        self.V5 = np.zeros_like(self.W5) 

        self.cost_ = []
        self.score_ = []
        # get starting acc
        self.score_.append(accuracy_score(y_data,self.predict(X_data)))
        # keep track of validation, if given
        if XY_test is not None:
            X_test = XY_test[0].copy()
            y_test = XY_test[1].copy()
            
            X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
            
            self.val_score_ = []
            self.val_score_.append(accuracy_score(y_test,self.predict(X_test)))
            self.val_cost_ = []
            
        for i in range(self.epochs):
            
            eta = self.eta

            if print_progress>0 and (i+1)%print_progress==0:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx_shuffle = np.random.permutation(y_data.shape[0])
                X_data, Y_enc, y_data = X_data[idx_shuffle], Y_enc[:, idx_shuffle], y_data[idx_shuffle]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            mini_cost = []
            for idx in mini:

                # feedforward
                A1, Z1, A2, Z2, A3, Z3, A4, Z4, A5, Z5, A6 = self._feedforward(X_data[idx], self.W1, self.W2, self.W3, self.W4, self.W5, self.b1, self.b2, self.b3, self.b4, self.b5)
                
                cost = self._cost(A6, Y_enc[:, idx], self.W1, self.W2, self.W3, self.W4, self.W5)
                mini_cost.append(cost) # this appends cost of mini-batch only

                # compute gradient via backpropagation
                gradW1, gradW2, gradW3, gradW4,gradW5, gradb1, gradb2, gradb3, gradb4, gradb5 = self._get_gradient(A1, A2, A3, A4, A5, A6, Z1, Z2, Z3, Z4, Z5, Y_enc[:,idx], W1=self.W1, W2=self.W2, W3=self.W3, W4=self.W4, W5=self.W5 )
                
                # Eta-grad implementation:
                G1 = gradW1*gradW1
                G2 = gradW2*gradW2
                G3 = gradW3*gradW3
                G4 = gradW4*gradW4
                G5 = gradW5*gradW5
                
                # RMSProp adjustments
                self.V1 = self.gamma * self.V1 + (1 - self.gamma) * G1
                self.V2 = self.gamma * self.V2 + (1 - self.gamma) * G2
                self.V3 = self.gamma * self.V3 + (1 - self.gamma) * G3
                self.V4 = self.gamma * self.V4 + (1 - self.gamma) * G4
                self.V5 = self.gamma * self.V5 + (1 - self.gamma) * G5

                 
                # Update weights
                self.W1 -= eta * gradW1/np.sqrt(self.V1)
                self.W2 -= eta * gradW2/np.sqrt(self.V2)
                self.W3 -= eta * gradW3/np.sqrt(self.V3)
                self.W4 -= eta * gradW4/np.sqrt(self.V4)
                self.W5 -= eta * gradW5/np.sqrt(self.V5)

                # Update biases
                self.b1 -= self.eta * gradb1
                self.b2 -= self.eta * gradb2
                self.b3 -= self.eta * gradb3
                self.b4 -= self.eta * gradb4
                self.b5 -= self.eta * gradb5
                
                # store gradient magnitude with average absolute values
                self.grad_magnitudes['W1'].append(np.mean(np.abs(gradW1)))
                self.grad_magnitudes['W2'].append(np.mean(np.abs(gradW2)))
                self.grad_magnitudes['W3'].append(np.mean(np.abs(gradW3)))
                self.grad_magnitudes['W4'].append(np.mean(np.abs(gradW4)))
                self.grad_magnitudes['W5'].append(np.mean(np.abs(gradW5)))

                
            self.cost_.append(np.mean(mini_cost))
            self.score_.append(accuracy_score(y_data,self.predict(X_data)))
 
            # update if a validation set was provided
            if XY_test is not None:
                yhat = self.predict(X_test)
                self.val_score_.append(accuracy_score(y_test,yhat))
            
        return self
    
    
    def _get_gradient(self, A1, A2, A3, A4, A5, A6, Z1, Z2, Z3, Z4, Z5, Y_enc, W1, W2, W3, W4, W5):
        """ Compute gradient step using backpropagation.
        """
        # vectorized backpropagation
        V5 = (A6 - Y_enc)
        V4 = A5 * (1 - A5) * (self.W5.T @ V5)
        V3 = A4 * (1 - A4) * (self.W4.T @ V4) 
        V2 = A3 * (1 - A3) * (self.W3.T @ V3) 
        V1 = A2 * (1 - A2) * (self.W2.T @ V2)
        
        gradW5 = V5 @ A5.T
        gradW4 = V4 @ A4.T
        gradW3 = V3 @ A3.T
        gradW2 = V2 @ A2.T
        gradW1 = V1 @ A1.T
        
        gradb5 = np.sum(V5, axis=1).reshape((-1, 1))
        gradb4 = np.sum(V4, axis=1).reshape((-1, 1))
        gradb3 = np.sum(V3, axis=1).reshape((-1, 1))
        gradb2 = np.sum(V2, axis=1).reshape((-1 ,1))
        gradb1 = np.sum(V1, axis=1).reshape((-1, 1))
        
        # regularize weights that are not bias terms
        gradW1 += W1 * self.l2_C
        gradW2 += W2 * self.l2_C
        gradW3 += W3 * self.l2_C
        gradW4 += W4 * self.l2_C
        gradW5 += W5 * self.l2_C
        
        return gradW1, gradW2, gradW3, gradW4, gradW5, gradb1, gradb2, gradb3, gradb4, gradb5
    
    def predict(self, X):
        """Predict class labels"""
        _, _, _, _, _, _, _, _, _, _, A6 = self._feedforward(X, self.W1, self.W2, self.W3, self.W4, self.W5, self.b1, self.b2, self.b3, self.b4, self.b5)
        y_pred = np.argmax(A6, axis=0)
        return y_pred
    
# Start from this 
class FLPAdaM(object):
    def __init__(self, n_hidden=30,
                 C=0.0, epochs=500, eta=0.001, random_state=None, shuffle = True, minibatches=1, beta1 = 0.9, beta2=0.999):
        np.random.seed(random_state)
        self.n_hidden1 = n_hidden
        self.n_hidden2 = n_hidden
        self.n_hidden3 = n_hidden
        self.n_hidden4 = n_hidden
        self.l2_C = C
        self.epochs = epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.shuffle = shuffle
        self.minibatches = minibatches
        self.grad_magnitudes = {'W1': [], 'W2': [], 'W3': [],'W4': [], 'W5': [] }
        
    @staticmethod
    def _encode_labels(y):
        """Encode labels into one-hot representation"""
        onehot = pd.get_dummies(y).values.T
            
        return onehot

    def _initialize_weights(self):
        """Initialize weights Glorot and He normalization."""
        init_bound = 4*np.sqrt(6. / (self.n_hidden1 + self.n_features_))
        W1 = np.random.uniform(-init_bound, init_bound,(self.n_hidden1, self.n_features_))
 
        init_bound = 4*np.sqrt(6 / (self.n_hidden1 + self.n_hidden2))
        W2 = np.random.uniform(-init_bound, init_bound,(self.n_hidden2, self.n_hidden1)) 
        
        init_bound = 4*np.sqrt(6 / (self.n_hidden2  + self.n_hidden3))
        W3 = np.random.uniform(-init_bound, init_bound,(self.n_hidden3, self.n_hidden2))
        
        init_bound = 4*np.sqrt(6 / (self.n_hidden3  + self.n_hidden4))
        W4 = np.random.uniform(-init_bound, init_bound,(self.n_hidden4, self.n_hidden3))           
        
        init_bound = 4*np.sqrt(6 / (self.n_hidden4  + self.n_output_))
        W5 = np.random.uniform(-init_bound, init_bound,(self.n_output_, self.n_hidden4))

        b1 = np.zeros((self.n_hidden1, 1))
        b2 = np.zeros((self.n_hidden2, 1))
        b3 = np.zeros((self.n_hidden3, 1))
        b4 = np.zeros((self.n_hidden4, 1))
        b5 = np.zeros((self.n_output_, 1))
        
        
        return W1, W2, W3, W4, W5, b1, b2, b3, b4, b5
    
    @staticmethod
    def _sigmoid(z):
        """Use scipy.special.expit to avoid overflow"""
        # 1.0 / (1.0 + np.exp(-z))
        return expit(z)
    
    
    @staticmethod
    def _L2_reg(lambda_, W1, W2, W3, W4, W5):
        """Compute L2-regularization cost"""
        # only compute for non-bias terms
        return (lambda_/2.0) * np.sqrt(np.mean(W1[:, 1:] ** 2) + np.mean(W2[:, 1:] ** 2) + np.mean(W3[:, 1:] ** 2)+ np.mean(W4[:, 1:] ** 2)+np.mean(W5[:, 1:] ** 2))
    
    def _cost(self,A6,Y_enc,W1,W2,W3,W4,W5):
        '''Get the objective function value'''
        cost = -np.mean(np.nan_to_num((Y_enc*np.log(A6+1e-7)+(1-Y_enc)*np.log(1-A6+1e-7))))
        L2_term = self._L2_reg(self.l2_C, W1, W2, W3, W4, W5)
        return cost + L2_term
    
    def _feedforward(self, X, W1, W2, W3, W4, W5, b1, b2, b3, b4, b5):

        A1 = X.T
        Z1 = W1 @ A1 + b1
        A2 = self._sigmoid(Z1)
        Z2 = W2 @ A2 + b2
        A3 = self._sigmoid(Z2)
        Z3 = W3 @ A3 + b3
        A4 = self._sigmoid(Z3)
        Z4 = W4 @ A4 + b4
        A5 = self._sigmoid(Z4)
        Z5 = W5 @ A5 + b5
        A6 = self._sigmoid(Z5)
        
        return A1, Z1, A2, Z2, A3, Z3, A4, Z4, A5, Z5, A6
    
    def fit(self, X, y, print_progress=False, XY_test=None):
        """ Learn weights from training data. With mini-batch"""
        X_data, y_data = X.copy(), y.copy()
        Y_enc = self._encode_labels(y)
        
        # Ensure X_data and y_data are numpy arrays for compatibility with numpy operations
        X_data = X_data.values if isinstance(X_data, pd.DataFrame) else X_data
        y_data = y_data.values if isinstance(y_data, pd.Series) else y_data
        
        # init weights and setup matrices
        self.n_features_ = X_data.shape[1]
        self.n_output_ = Y_enc.shape[0]
        self.W1, self.W2, self.W3, self.W4, self.W5, self.b1, self.b2, self.b3, self.b4, self.b5 = self._initialize_weights()
        
        # Initialize accumulators Vk for RMSProp
        self.V1, self.M1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.V2, self.M2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.V3, self.M3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.V4, self.M4 = np.zeros_like(self.W4), np.zeros_like(self.W4)
        self.V5, self.M5 = np.zeros_like(self.W5), np.zeros_like(self.W5)

        self.cost_ = []
        self.score_ = []
        # get starting acc
        self.score_.append(accuracy_score(y_data,self.predict(X_data)))
        # keep track of validation, if given
        if XY_test is not None:
            X_test = XY_test[0].copy()
            y_test = XY_test[1].copy()
            
            X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
            
            self.val_score_ = []
            self.val_score_.append(accuracy_score(y_test,self.predict(X_test)))
            self.val_cost_ = []
            
        for i in range(self.epochs):
            
            eta = self.eta

            if print_progress>0 and (i+1)%print_progress==0:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx_shuffle = np.random.permutation(y_data.shape[0])
                X_data, Y_enc, y_data = X_data[idx_shuffle], Y_enc[:, idx_shuffle], y_data[idx_shuffle]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            mini_cost = []
            for idx in mini:

                # feedforward
                A1, Z1, A2, Z2, A3, Z3, A4, Z4, A5, Z5, A6 = self._feedforward(X_data[idx], self.W1, self.W2, self.W3, self.W4, self.W5, self.b1, self.b2, self.b3, self.b4, self.b5)
                
                cost = self._cost(A6, Y_enc[:, idx], self.W1, self.W2, self.W3, self.W4, self.W5)
                mini_cost.append(cost) # this appends cost of mini-batch only

                # compute gradient via backpropagation
                gradW1, gradW2, gradW3, gradW4,gradW5, gradb1, gradb2, gradb3, gradb4, gradb5 = self._get_gradient(A1, A2, A3, A4, A5, A6, Z1, Z2, Z3, Z4, Z5, Y_enc[:,idx], W1=self.W1, W2=self.W2, W3=self.W3, W4=self.W4, W5=self.W5 )
                
                # Implement Eta-grad
                G1 = gradW1*gradW1
                G2 = gradW2*gradW2
                G3 = gradW3*gradW3
                G4 = gradW4*gradW4
                G5 = gradW5*gradW5
                
                # accumulated gradient
                self.M1 = self.beta1 * self.M1 + (1 - self.beta1) * gradW1
                self.M2 = self.beta1 * self.M2 + (1 - self.beta1) * gradW2
                self.M3 = self.beta1 * self.M3 + (1 - self.beta1) * gradW3
                self.M4 = self.beta1 * self.M4 + (1 - self.beta1) * gradW4
                self.M5 = self.beta1 * self.M5 + (1 - self.beta1) * gradW5
                # accumulated squared gradient
                self.V1 = self.beta2 * self.V1 + (1 - self.beta2) * G1
                self.V2 = self.beta2 * self.V2 + (1 - self.beta2) * G2
                self.V3 = self.beta2 * self.V3 + (1 - self.beta2) * G3
                self.V4 = self.beta2 * self.V4 + (1 - self.beta2) * G4
                self.V5 = self.beta2 * self.V5 + (1 - self.beta2) * G5

                # boost moments magnitudes
                M_hat1 = self.M1 / (1 - self.beta1 ** (i+1))
                M_hat2 = self.M2 / (1 - self.beta1 ** (i+1))
                M_hat3 = self.M3 / (1 - self.beta1 ** (i+1))
                M_hat4 = self.M4 / (1 - self.beta1 ** (i+1))
                M_hat5 = self.M5 / (1 - self.beta1 ** (i+1))
                
                V_hat1 = self.V1 / (1 - self.beta2 ** (i+1))
                V_hat2 = self.V2 / (1 - self.beta2 ** (i+1))
                V_hat3 = self.V3 / (1 - self.beta2 ** (i+1))
                V_hat4 = self.V4 / (1 - self.beta2 ** (i+1))
                V_hat5 = self.V5 / (1 - self.beta2 ** (i+1))
                 
                # Update weights
                self.W1 -= eta * M_hat1/np.sqrt(V_hat1)
                self.W2 -= eta * M_hat2/np.sqrt(V_hat2)
                self.W3 -= eta * M_hat3/np.sqrt(V_hat3)
                self.W4 -= eta * M_hat4/np.sqrt(V_hat4)
                self.W5 -= eta * M_hat5/np.sqrt(V_hat5)

                # Update biases
                self.b1 -= self.eta * gradb1
                self.b2 -= self.eta * gradb2
                self.b3 -= self.eta * gradb3
                self.b4 -= self.eta * gradb4
                self.b5 -= self.eta * gradb5
                
                # store gradient magnitude with average absolute values
                self.grad_magnitudes['W1'].append(np.mean(np.abs(gradW1)))
                self.grad_magnitudes['W2'].append(np.mean(np.abs(gradW2)))
                self.grad_magnitudes['W3'].append(np.mean(np.abs(gradW3)))
                self.grad_magnitudes['W4'].append(np.mean(np.abs(gradW4)))
                self.grad_magnitudes['W5'].append(np.mean(np.abs(gradW5)))

                
            self.cost_.append(np.mean(mini_cost))
            self.score_.append(accuracy_score(y_data,self.predict(X_data)))
 
            # update if a validation set was provided
            if XY_test is not None:
                yhat = self.predict(X_test)
                self.val_score_.append(accuracy_score(y_test,yhat))
            
        return self
    
    
    def _get_gradient(self, A1, A2, A3, A4, A5, A6, Z1, Z2, Z3, Z4, Z5, Y_enc, W1, W2, W3, W4, W5):
        """ Compute gradient step using backpropagation.
        """
        # vectorized backpropagation
        V5 = (A6 - Y_enc)
        V4 = A5 * (1 - A5) * (self.W5.T @ V5)
        V3 = A4 * (1 - A4) * (self.W4.T @ V4) 
        V2 = A3 * (1 - A3) * (self.W3.T @ V3) 
        V1 = A2 * (1 - A2) * (self.W2.T @ V2)
        
        gradW5 = V5 @ A5.T
        gradW4 = V4 @ A4.T
        gradW3 = V3 @ A3.T
        gradW2 = V2 @ A2.T
        gradW1 = V1 @ A1.T
        
        gradb5 = np.sum(V5, axis=1).reshape((-1, 1))
        gradb4 = np.sum(V4, axis=1).reshape((-1, 1))
        gradb3 = np.sum(V3, axis=1).reshape((-1, 1))
        gradb2 = np.sum(V2, axis=1).reshape((-1 ,1))
        gradb1 = np.sum(V1, axis=1).reshape((-1, 1))
        
        # regularize weights that are not bias terms
        gradW1 += W1 * self.l2_C
        gradW2 += W2 * self.l2_C
        gradW3 += W3 * self.l2_C
        gradW4 += W4 * self.l2_C
        gradW5 += W5 * self.l2_C
        
        return gradW1, gradW2, gradW3, gradW4, gradW5, gradb1, gradb2, gradb3, gradb4, gradb5
    
    def predict(self, X):
        """Predict class labels"""
        _, _, _, _, _, _, _, _, _, _, A6 = self._feedforward(X, self.W1, self.W2, self.W3, self.W4, self.W5, self.b1, self.b2, self.b3, self.b4, self.b5)
        y_pred = np.argmax(A6, axis=0)
        return y_pred