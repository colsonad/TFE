import numpy as np
from Modele import Modele
from numba import njit

# Numba-accelerated gradient descent on precomputed kernel matrix
@njit  
def fast_gradient_descent(K: np.ndarray, y: np.ndarray, alpha_init: np.ndarray,
                          lr: float, max_iters: int, target: float,
                          lambda_param: float, criterion: int) -> tuple:
    """
    Numba-accelerated gradient descent on precomputed kernel matrix.
    criterion: 0 for 'norm', 1 for 'loss'
    """
    n = K.shape[0]
    alpha = alpha_init.copy()
    alpha_values = [alpha.copy()]
    i = 0

    while i < max_iters - 1:
        I = np.identity(n)
        A = (1 / n) * K + lambda_param * I
        grad = K @ (A @ alpha - (1 / n) * y)  #compute gradient

        if criterion == 0:  # 'norm'
            if np.linalg.norm(grad) <= target:
                break
        elif criterion == 1:  # 'loss'
            Ka = K @ alpha
            residual = y - Ka
            loss = (residual.T @ residual) / (2 * n) + (lambda_param / 2) * (alpha.T @ Ka) #compute loss
            if loss <= target:
                break

        alpha = alpha - lr * grad
        alpha_values.append(alpha.copy())
        i += 1
    return alpha, alpha_values
    
@njit  
def fast_dynamic_stepsize(K: np.ndarray, y: np.ndarray, alpha_init: np.ndarray,
                          L: float, max_iters: int, target: float,
                          lambda_param: float, criterion: int) -> tuple:
    """
    Numba-accelerated gradient descent on precomputed kernel matrix.
    criterion: 0 for 'norm', 1 for 'loss'
    """
    n = K.shape[0]
    alpha = alpha_init.copy()
    alpha_values = [alpha.copy()]
    i = 0

    t = np.sqrt(2) / L
    T = 0

    stepsizes = [t]

    while i < max_iters - 1:
        I = np.identity(n)
        A = (1 / n) * K + lambda_param * I
        grad = K @ (A @ alpha - (1 / n) * y)  #compute gradient

        if criterion == 0:  # 'norm'
            if np.linalg.norm(grad) <= target:
                break
        elif criterion == 1:  # 'loss'
            Ka = K @ alpha
            residual = y - Ka
            loss = (residual.T @ residual) / (2 * n) + (lambda_param / 2) * (alpha.T @ Ka) #compute loss
            if loss <= target:
                break

        alpha = alpha - t * grad
        alpha_values.append(alpha.copy())
        i += 1
        T += t
        t = (-L * T + np.sqrt((L * T)**2 + 8 * (L * T + 1))) / (2 * L)
        stepsizes.append(t)
    return alpha, alpha_values,stepsizes

@njit
def fast_optimal(K: np.ndarray, y: np.ndarray, alpha_init: np.ndarray,
                          L: float, max_iters: int, target: float,
                          lambda_param: float, criterion: int) -> tuple:
    n = K.shape[0]
    alpha = alpha_init.copy()
    alpha_values = [alpha.copy()]
    i = 0

    stepsizes = []

    A_ = (K.T @ K /n + lambda_param * K)

    while i < max_iters - 1:
        I = np.identity(n)
        A = (1 / n) * K + lambda_param * I
        grad = K @ (A @ alpha - (1 / n) * y)  #compute gradient

        if criterion == 0:  # 'norm'
            if np.linalg.norm(grad) <= target:
                break
        elif criterion == 1:  # 'loss'
            Ka = K @ alpha
            residual = y - Ka
            loss = (residual.T @ residual) / (2 * n) + (lambda_param / 2) * (alpha.T @ Ka) #compute loss
            if loss <= target:
                break
        learning_rate = (grad.T @ grad) / (grad.T @ A_ @ grad)
        alpha = alpha - learning_rate * grad
        stepsizes.append(learning_rate)
        alpha_values.append(alpha.copy())
        i+= 1
    stepsizes.append(learning_rate)
    return alpha, alpha_values,stepsizes

@njit
def fast_exact(K: np.ndarray, y: np.ndarray, alpha_init: np.ndarray,
                          L: float, max_iters: int, target: float,
                          lambda_param: float, criterion: int) -> tuple:
    n = K.shape[0]
    alpha = alpha_init.copy()
    alpha_values = [alpha.copy()]
    i = 0

    stepsizes = []

    A_ = (K.T @ K /n + lambda_param * K)

    while i < max_iters - 1:
        I = np.identity(n)
        A = (1 / n) * K + lambda_param * I
        grad = K @ (A @ alpha - (1 / n) * y)  #compute gradient

        if criterion == 0:  # 'norm'
            if np.linalg.norm(grad) <= target:
                break
        elif criterion == 1:  # 'loss'
            Ka = K @ alpha
            residual = y - Ka
            loss = (residual.T @ residual) / (2 * n) + (lambda_param / 2) * (alpha.T @ Ka) #compute loss
            if loss <= target:
                break
        learning_rate = (grad.T @ A_.T @ grad) / (grad.T @ A_.T @ A_ @ grad)
        alpha = alpha - learning_rate * grad
        stepsizes.append(learning_rate)
        alpha_values.append(alpha.copy())
        i+= 1
    stepsizes.append(learning_rate)
    return alpha, alpha_values,stepsizes

@njit
def fast_periodic(K: np.ndarray, y: np.ndarray, alpha_init: np.ndarray,
                          L: float, max_iters: int, target: float,
                          lambda_param: float, criterion: int) -> tuple:
    n = K.shape[0]
    alpha = alpha_init.copy()
    alpha_values = [alpha.copy()]
    i = 0

    if max_iters == 3: stepsize = np.array([1.5, 4.9, 1.5])
    elif max_iters == 7: stepsize = np.array([1.5, 2.2, 1.5, 12.0, 1.5, 2.2, 1.5])
    elif max_iters == 15: stepsize = np.array([1.4, 2.0, 1.4, 4.5, 1.4, 2.0, 1.4, 29.7, 1.4, 2.0, 1.4, 4.5, 1.4, 2.0, 1.4])
    elif max_iters == 31: stepsize = np.array([1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 8.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4,
                                        72.3, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 8.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4])
    elif max_iters == 63: stepsize = np.array([1.4, 2.0, 1.4, 3.9, 1.4,2.0,1.4,7.2, 1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,14.2,
                                        1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,7.2, 1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,164.0,
                                        1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,7.2,1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,14.2,
                                        1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,7.2,1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4])   
    elif max_iters == 127: stepsize = np.array([1.4, 2.0, 1.4, 3.9, 1.4,2.0,1.4,7.2,1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,12.6,
                                        1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,7.2,1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,23.5,
                                        1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,7.2,1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,12.6,
                                        1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,7.2,1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,370.0,
                                        1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,7.2,1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,12.6,
                                        1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,7.2,1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,23.5,
                                        1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,7.2,1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,12.6,
                                        1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4,7.2,1.4, 2.0, 1.4, 3.9,1.4,2.0,1.4])


    while i < max_iters - 1:
        I = np.identity(n)
        A = (1 / n) * K + lambda_param * I
        grad = K @ (A @ alpha - (1 / n) * y)  #compute gradient

        if criterion == 0:  # 'norm'
            if np.linalg.norm(grad) <= target:
                break
        elif criterion == 1:  # 'loss'
            Ka = K @ alpha
            residual = y - Ka
            loss = (residual.T @ residual) / (2 * n) + (lambda_param / 2) * (alpha.T @ Ka) #compute loss
            if loss <= target:
                break
        alpha = alpha - stepsize[i]/L * grad
        alpha_values.append(alpha.copy())
        i+= 1
    stepsize = stepsize[:i+1]
    return alpha, alpha_values,stepsize/L

@njit
def fast_chebyshev(K: np.ndarray, y: np.ndarray, alpha_init: np.ndarray,
                          L: float, mu: float, max_iters: int, target: float,
                          lambda_param: float, criterion: int) -> tuple:
    
    n = K.shape[0]
    alpha = alpha_init.copy()
    alpha_values = [alpha.copy()]
    i = 0

    rho = (L - mu) / (L + mu)
    omega = 2
    step = 2 / (L + mu)
    alpha_old = alpha.copy()



    while i < max_iters - 1:
        I = np.identity(n)
        A = (1 / n) * K + lambda_param * I
        grad = K @ (A @ alpha - (1 / n) * y)  #compute gradient

        if criterion == 0:  # 'norm'
            if np.linalg.norm(grad) <= target:
                break
        elif criterion == 1:  # 'loss'
            Ka = K @ alpha
            residual = y - Ka
            loss = (residual.T @ residual) / (2 * n) + (lambda_param / 2) * (alpha.T @ Ka) #compute loss
            if loss <= target:
                break
        if i==0: alpha -= step * grad 
        else:
            omega = 1 / (1 - (rho**2 / 4) * omega)
            vt = alpha_old - alpha
            alpha_old = alpha.copy()
            alpha += (1 - omega) * vt - omega * step * grad

        alpha_values.append(alpha.copy())
        i+= 1
        
    return alpha, alpha_values


class Optimiseur:
    """Class to perform different optimization algorithms on a given model."""

    def __init__(self, modele: Modele):
        self.modele = modele

    def gradient_descent(self, X: np.ndarray, y: np.ndarray, alpha_init: np.ndarray, learning_rate: float, max_iters: int, target: float, criterion: str = 'norm') -> tuple:
        """Perform the Gradient Descent algorithm with a fixed step size"""
        K = self.modele.kernel(X, X)
        criterion_flag = 0 if criterion == 'norm' else 1
        alpha, alpha_list = fast_gradient_descent(K, y, alpha_init, learning_rate, max_iters, target, self.modele.lambda_param, criterion_flag)
        return alpha, alpha_list

    def dynamic_stepsize(self, X: np.ndarray, y: np.ndarray, alpha_init: np.ndarray, max_iters: int, target: float, criterion: str = 'norm', get_stepsize: bool=False) -> tuple:
        """Perform the Gradient Descent algorithm with a dynamic step size"""

        K = self.modele.kernel(X, X)
        L, mu = self.modele.constante_L(X)
        criterion_flag = 0 if criterion == 'norm' else 1
        alpha, alpha_lists, stepsizes = fast_dynamic_stepsize(K, y, alpha_init, L, max_iters, target, self.modele.lambda_param, criterion_flag)
        if get_stepsize: 
            return alpha, alpha_lists, stepsizes
        return alpha, alpha_lists

    def optimal_stepsize(self, X:np.ndarray,y:np.ndarray,alpha_init:np.ndarray,max_iters:int,target:float,criterion:str='norm',get_stepsize: bool=False) -> tuple:
        """Perform the Gradient Descent algorithm with optimal step size"""
        K = self.modele.kernel(X, X)
        L, mu = self.modele.constante_L(X)
        criterion_flag = 0 if criterion == 'norm' else 1
        alpha, alpha_lists, stepsizes = fast_optimal(K, y, alpha_init, L, max_iters, target, self.modele.lambda_param, criterion_flag)
        if get_stepsize: 
            return alpha, alpha_lists, stepsizes
        return alpha, alpha_lists
    
    def exact_stepsize(self, X:np.ndarray,y:np.ndarray,alpha_init:np.ndarray,max_iters:int,target:float,criterion:str='norm',get_stepsize: bool=False) -> tuple:
        """Perform the Gradient Descent algorithm with optimal step size"""
        K = self.modele.kernel(X, X)
        L, mu = self.modele.constante_L(X)
        criterion_flag = 0 if criterion == 'norm' else 1
        alpha, alpha_lists, stepsizes = fast_exact(K, y, alpha_init, L, max_iters, target, self.modele.lambda_param, criterion_flag)
        if get_stepsize: 
            return alpha, alpha_lists, stepsizes
        return alpha, alpha_lists
    
    def Periodic(self, X: np.ndarray, y: np.ndarray, alpha_init: np.ndarray, max_iters: int, target: float, criterion: str = 'norm', get_stepsize: bool=False) -> tuple:
        """Perform the Gradient Descent algorithm with a periodic large step size"""
        K = self.modele.kernel(X, X)
        L, mu = self.modele.constante_L(X)
        criterion_flag = 0 if criterion == 'norm' else 1
        iter = max_iters
        alphas = []
        stepsizes_final = []
        while iter >= 127:
            alpha, alpha_lists, stepsizes = fast_periodic(K, y, alpha_init, L, 127, target, self.modele.lambda_param, criterion_flag)
            alpha_init = alpha.copy()
            alphas.append(alpha_lists)
            stepsizes_final.append(stepsizes)
            if len(alpha_lists) < 127:
                iter = 0
            else: 
                iter -= 127
        while iter >= 63: 
            alpha, alpha_lists, stepsizes = fast_periodic(K, y, alpha_init, L, 63, target, self.modele.lambda_param, criterion_flag)
            alpha_init = alpha.copy()
            alphas.append(alpha_lists)
            stepsizes_final.append(stepsizes)
            if len(alpha_lists) < 63:
                iter = 0
            else: 
                iter -= 63

        while iter >= 31:
            alpha, alpha_lists, stepsizes = fast_periodic(K, y, alpha_init, L, 31, target, self.modele.lambda_param, criterion_flag)
            alpha_init = alpha.copy()
            alphas.append(alpha_lists)
            stepsizes_final.append(stepsizes)
            if len(alpha_lists) < 31:
                iter = 0
            else: 
                iter -= 31

        while iter >= 15:
            alpha, alpha_lists, stepsizes = fast_periodic(K, y, alpha_init, L, 15, target, self.modele.lambda_param, criterion_flag)
            alpha_init = alpha.copy()
            alphas.append(alpha_lists)
            stepsizes_final.append(stepsizes)
            if len(alpha_lists) < 15:
                iter = 0
            else: 
                iter -= 15
        while iter >= 7:
            alpha, alpha_lists, stepsizes = fast_periodic(K, y, alpha_init, L, 7, target, self.modele.lambda_param, criterion_flag)
            alpha_init = alpha.copy()
            alphas.append(alpha_lists)
            stepsizes_final.append(stepsizes)
            if len(alpha_lists) < 7:
                iter = 0
            else: 
                iter -= 7
        while iter >= 3:
            alpha, alpha_lists, stepsizes = fast_periodic(K, y, alpha_init, L, 3, target, self.modele.lambda_param, criterion_flag)
            alpha_init = alpha.copy()
            alphas.append(alpha_lists)
            stepsizes_final.append(stepsizes)
            if len(alpha_lists) < 3:
                iter = 0
            else: 
                iter -= 3
        alpha_lists = [] 
        for list in alphas: 
            for elem in list:
                alpha_lists.append(elem)
        stepsizes_list = []
        for list in stepsizes_final:
            for elem in list:
                stepsizes_list.append(elem)
        if get_stepsize: 
            return alpha_init, alpha_lists, stepsizes_list
        return alpha_init, alpha_lists
    