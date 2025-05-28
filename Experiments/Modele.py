import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from numba import njit,prange


# use of numba to speed up the computation of the kernel and the loss function
@njit(parallel=True)
def rbf_kernel(X, Z, gamma):
    n1, d1 = X.shape
    n2, d2 = Z.shape
    K = np.empty((n1, n2))
    for i in prange(n1):
        for j in range(n2):
            sq_dist = 0.0
            for k in range(d1):
                diff = X[i, k] - Z[j, k]
                sq_dist += diff * diff
            K[i, j] = np.exp(-gamma * sq_dist)
    return K

@njit
def gradient(K, y, alpha, lambda_param):
    n = K.shape[0]
    I = np.identity(n)
    A = (1 / n) * K + lambda_param * I
    grad = K @ (A @ alpha - (1 / n) * y)
    return grad
@njit
def loss(K, y, alpha, lambda_param):
    n = K.shape[0]
    residual = y - K @ alpha
    loss = (residual.T @ residual) / (2 * n) + (lambda_param / 2) * (alpha.T @ K @ alpha)
    return loss
@njit
def test_loss(K_train,K_test, y, alpha, lambda_param):
    n = K_train.shape[0]
    loss = 1 / (2 * n) * np.linalg.norm(y - K_test @ alpha)**2 + (lambda_param / 2) * np.linalg.norm(K_train @ alpha)**2
    return loss
@njit
def accuracy(y_true, K, alpha):
    y_pred = np.sign(K @ alpha)
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)

@njit
def fast_compute_all(X,x,y,y_,alphas,lambda_param,gamma):
    n = X.shape[0]
    K = rbf_kernel(X, X,gamma)
    K_ = rbf_kernel(x, X,gamma) 

    loss = np.zeros(len(alphas))
    test_loss = np.zeros(len(alphas))
    accuracy = np.zeros(len(alphas))
    train_accuracy = np.zeros(len(alphas))
    norm = np.zeros(len(alphas))
    j=0
    for alpha in alphas:
        residual = y - K @ alpha
        loss[j] = (residual.T @ residual) / (2 * n) + (lambda_param / 2) * (alpha.T @ K @ alpha)

        test_loss[j] = 1 / (2 * n) * np.linalg.norm(y_ - K_ @ alpha)**2 + (lambda_param / 2) * np.linalg.norm(K_ @ alpha)**2

        y_pred = np.sign(K_ @ alpha)
        correct = len(y_)
        for i in range(len(y_)):
            if y_[i] != y_pred[i]:
                correct -= 1
        accuracy[j] = correct / len(y_)
        y_pred = np.sign(K @ alpha)
        correct = 0
        for i in range(len(y)):
            if y[i] == y_pred[i]:
                correct += 1

        train_accuracy[j] = correct / len(y)
        I = np.identity(n)
        A = (1 / n) * K + lambda_param * I
        grad = K @ (A @ alpha - (1 / n) * y)

        norm[j] = np.linalg.norm(grad)

        
        j+=1
    
    return loss, test_loss, norm, train_accuracy, accuracy

class Modele: 
    """Base class for kernel models"""

    def __init__(self, lambda_param: float = 1.0,gamma: float = 25):
        self.lambda_param = lambda_param
        self.gamma = gamma

    def kernel(self, X: np.ndarray, Z: np.ndarray, kernel_type: str = 'rbf', gamma: float = 25, degree: int = 3) -> np.ndarray:
        """Compute the kernel matrix"""
        if kernel_type == 'linear':
            return X @ Z.T
        elif kernel_type == 'rbf':
            return np.exp(-self.gamma * cdist(X, Z, 'sqeuclidean'))
        elif kernel_type == 'poly':
            return rbf_kernel(X, Z, self.gamma)
        
    def loss_function(self, X: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> float:
        """Calculate the training loss function value"""
        K = self.kernel(X, X)
        return loss(K, y, alpha, self.lambda_param)
    
    def predictions(self, X_train: np.ndarray, X_test: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return np.sign(self.kernel(X_test, X_train) @ alpha)
    
    def accuracy(self, X_train: np.ndarray, X_test: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> float:
        """Calculate the accuracy"""
        K = self.kernel(X_test, X_train)
        return accuracy(y, K, alpha)
    
    def test_loss_function(self, X_train: np.ndarray, X_test: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> float:
        """Calculate the test loss function value"""
        K_test = self.kernel(X_test, X_train)
        K_train = self.kernel(X_train, X_train)
        return test_loss(K_train,K_test, y, alpha, self.lambda_param)
    
    def gradient(self, X: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the loss function"""
        n = X.shape[0]
        K = self.kernel(X, X)
        return gradient(K, y, alpha, self.lambda_param)
        

    def constante_L(self, X: np.ndarray) -> tuple:
        """Compute the Lipschitz constant"""
        n = X.shape[0]
        K = self.kernel(X, X)
        eigenvals, _ = np.linalg.eigh(K @ ((1 / n * K + self.lambda_param * np.identity(n)))) #Hessian
        L = np.max(eigenvals)
        mu = np.min(eigenvals)
        return L, mu
    
    def alpha_opt(self, X: np.ndarray, y: np.ndarray, gamma: float = 30) -> np.ndarray:
        """Compute the optimal alpha for the training set"""
        n = X.shape[0]
        alpha = np.dot(np.linalg.pinv(self.kernel(X, X, gamma=gamma) + n * self.lambda_param * np.identity(n)), y)

        return alpha

    def compute_all(self, X: np.ndarray, x: np.ndarray, y: np.ndarray, y_: np.ndarray, alpha: np.ndarray) -> tuple:
        return fast_compute_all(X,x,y,y_,alpha,self.lambda_param,self.gamma)