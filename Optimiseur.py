import numpy as np
from Modele import Modele

class Optimiseur:
    def __init__(self, modele: Modele):
        self.modele = modele

    def gradient_descent(self, X: np.ndarray, y: np.ndarray, alpha_init: np.ndarray, learning_rate: float, max_iters: int, target: float, criterion: str = 'norm') -> tuple:
        """Perform the Gradient Descent algorithm with a fixed step size"""
        alpha_values = []
        alpha = alpha_init
        i = 0
        alpha_values.append(alpha)

        if criterion == 'loss':
            condition = lambda: self.modele.loss_function(X, y, alpha) > target
        elif criterion == 'norm':
            grad = self.modele.gradient(X, y, alpha)
            condition = lambda: np.linalg.norm(grad) > target

        while condition() and i < max_iters - 1:
            grad = self.modele.gradient(X, y, alpha)
            alpha = alpha - learning_rate * grad
            i += 1
            alpha_values.append(alpha)

        return alpha, alpha_values

    def dynamic_stepsize(self, X: np.ndarray, y: np.ndarray, alpha_init: np.ndarray, max_iters: int, target: float, criterion: str = 'norm') -> tuple:
        """Perform the Gradient Descent algorithm with a dynamic step size"""
        alpha_values = []
        alpha = alpha_init
        L, mu = self.modele.constante_L(X)
        i = 0
        alpha_values.append(alpha)
        t = np.sqrt(2) / L
        T = 0
        grad = self.modele.gradient(X, y, alpha)

        if criterion == 'loss':
            condition = lambda: self.modele.loss_function(X, y, alpha) > target
        elif criterion == 'norm':
            grad = self.modele.gradient(X, y, alpha)
            condition = lambda: np.linalg.norm(grad) > target

        while condition() and i < max_iters - 1:
            grad = self.modele.gradient(X, y, alpha)
            alpha =alpha - t * grad
            i += 1
            alpha_values.append(alpha)
            T += t
            t = (-L * T + np.sqrt((L * T)**2 + 8 * (L * T + 1))) / (2 * L)
        return alpha, alpha_values
    
    def chebychev_root(self, L: float, mu: float, t: int, i: int) -> float:
        """Compute the Chebychev roots"""
        w = L + mu + (L - mu) * np.cos((i + 1 / 2) * np.pi / (t))
        return w / 2

    def permutation(self, previous: list, T: int) -> list:
        """Compute the permutation of the Chebychev roots"""
        change = [2 * T + 1 - p for p in previous]
        new = []
        for a in range(np.size(previous)):
            new.append(previous[a])
            new.append(change[a])
        return new

    def perm(self, T: int) -> list:
        """Compute the permutation of the Chebychev roots"""
        perm = [1]
        i = 1
        while i < T:
            perm = self.permutation(perm, i)
            i *= 2
        return perm

    def Chebychev(self, X: np.ndarray, y: np.ndarray, alpha_init: np.ndarray, max_iters: int, target: float, criterion: str = 'norm', fractal: bool = False) -> tuple:
        """Perform the Gradient Descent algorithm with a Chebychev step size"""
        alpha_values = []
        alpha = alpha_init
        i = 0
        alpha_values.append(alpha)
        L, mu = self.modele.constante_L(X)
        
        roots = [self.chebychev_root(L, mu, max_iters - 1, i) for i in range(max_iters)]

        permute = self.perm(max_iters - 1)
        permuted_roots = [roots[i - 1] for i in permute]
        
        grad = self.modele.gradient(X, y, alpha)

        if criterion == 'loss':
            condition = lambda: self.modele.loss_function(X, y, alpha) > target
        elif criterion == 'norm':
            grad = self.modele.gradient(X, y, alpha)
            condition = lambda: np.linalg.norm(grad) > target

        while condition() and i < max_iters - 1:
            grad = self.modele.gradient(X, y, alpha)
            if fractal:
                alpha = alpha - 1 / permuted_roots[i] * grad
            else:
                alpha = alpha - 1 / roots[i] * grad
            i += 1
            alpha_values.append(alpha)
        return alpha, alpha_values
