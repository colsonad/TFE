import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class Modele: 
    def __init__(self, lambda_param: float = 1.0):
        self.lambda_param = lambda_param

    def kernel(self, X: np.ndarray, Z: np.ndarray, kernel_type: str = 'rbf', gamma: float = 1/(30), degree: int = 3) -> np.ndarray:
        """Compute the kernel matrix"""
        if kernel_type == 'linear':
            return X @ Z.T
        elif kernel_type == 'rbf':
            return np.exp(-gamma * cdist(X, Z, 'sqeuclidean'))
        elif kernel_type == 'poly':
            return (X @ Z.T + 1)**degree
        
    def loss_function(self, X: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> float:
        """Calculate the training loss function value"""
        n = X.shape[0]
        K = self.kernel(X, X)
        residual = y - K @ alpha
        loss = (residual.T @ residual) / (2 * n) + (self.lambda_param / 2) * (alpha.T @ K @ alpha)
        return loss
    
    def predictions(self, X_train: np.ndarray, X_test: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return np.sign(self.kernel(X_test, X_train) @ alpha)
    
    def accuracy(self, X_train: np.ndarray, X_test: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> float:
        """Calculate the accuracy"""
        return accuracy_score(y, self.predictions(X_train, X_test, alpha))
    
    def test_loss_function(self, X_train: np.ndarray, X_test: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> float:
        """Calculate the test loss function value"""
        n = X_test.shape[0]
        loss = 1 / (2 * n) * np.linalg.norm(y - self.kernel(X_test, X_train) @ alpha)**2
        return loss
    
    def gradient(self, X: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the loss function"""
        n = X.shape[0]
        K = self.kernel(X, X)
        grad = K @ ((1 / n * K + self.lambda_param * np.identity(n)) @ alpha - 1 / n * y)
        return grad

    def constante_L(self, X: np.ndarray) -> tuple:
        """Compute the Lipschitz constant"""
        n = X.shape[0]
        K = self.kernel(X, X)
        eigenvals, _ = np.linalg.eigh(K @ ((1 / n * K + self.lambda_param * np.identity(n))))
        L = np.max(eigenvals)
        mu = np.min(eigenvals)
        return L, mu
    
    def alpha_opt(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute the optimal alpha for the training set"""
        n = X.shape[0]
        alpha = np.linalg.solve(self.kernel(X, X) + n * self.lambda_param * np.identity(n), y)
        return alpha
