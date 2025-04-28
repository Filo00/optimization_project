import numpy as np

def x_squared(x):
    return x ** 2

def grad_x_squared(x):
    return 2 * x

def logistic_loss(X, y, w, lam):
    z = y * X.dot(w)
    loss = np.mean(np.log(1 + np.exp(-z)))  # Log-loss  log(1-e^(-y(w^Tx)))
    regul = 0.5 * np.linalg.norm(w) ** 2  # Regolarizzazione L2
    return loss + lam * regul # => log-loss + lambda/2 * ||w||^2

def logistic_gradient(X, y, w, lam):
    z = y * X.dot(w)
    grad_loss = -X.T.dot(y * (1 - 1 / (1 + np.exp(-z)))) / len(y)
    grad_regul = lam * w
    return grad_loss + grad_regul

def logistic_hessian(X, y, w, lam):
    z = y * X.dot(w)
    sigma = 1 / (1 + np.exp(z))
    D = sigma * (1 - sigma)
    D_matrix = np.diag(D)
    H = (1 / X.shape[0]) * X.T @ D_matrix @ X + lam * np.eye(X.shape[1]) # 1/N XTDX + lam*I
    return H


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

