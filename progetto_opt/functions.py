import numpy as np

def x_squared(x):
    return x ** 2

def grad_x_squared(x):
    return 2 * x

def logistic_loss(X, y, w, lam):
    """
    Calcola la log-loss con regolarizzazione L2.

    Parameters:
    w : numpy.ndarray
        Vettore dei pesi di dimensione (p,).
    X : numpy.ndarray
        Matrice delle caratteristiche di dimensione (N, p).
    y : numpy.ndarray
        Vettore dei target di dimensione (N,).
    lam : float
        Parametro di regolarizzazione L2.

    Returns:
    float
        Valore della funzione di loss logistica con regolarizzazione L2.
    """
    z = y * X.dot(w)
    loss = np.mean(np.log(1 + np.exp(-z)))  # Log-loss
    regul = 0.5 * np.linalg.norm(w) ** 2  # Regolarizzazione L2
    return loss + lam * regul


def logistic_gradient(X, y, w, lam):
    """
    Calcola il gradiente della log-loss con regolarizzazione L2.

    Parameters:
    w : numpy.ndarray
        Vettore dei pesi di dimensione (p,).
    X : numpy.ndarray
        Matrice delle caratteristiche di dimensione (N, p).
    y : numpy.ndarray
        Vettore dei target di dimensione (N,).
    lam : float
        Parametro di regolarizzazione L2.

    Returns:
    numpy.ndarray
        Gradiente della funzione di perdita rispetto ai pesi w.
    """
    z = y * X.dot(w)
    grad_loss = -X.T.dot(y * (1 - 1 / (1 + np.exp(-z)))) / len(y)
    grad_regul = lam * w
    return grad_loss + grad_regul

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

