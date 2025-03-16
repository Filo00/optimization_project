import numpy as np
import functions
import utils
from tqdm import tqdm

def gradient_descent(X, y, loss_fun, grad_fun, lam, tol, max_iter, step_method):
    """
    Esegue il metodo del gradiente per minimizzare una funzione di loss.

    Parameters:
    X : numpy.ndarray
        Matrice delle caratteristiche di dimensione (N, p).
    y : numpy.ndarray
        Vettore dei target di dimensione (N,).
    loss_fun : callable
        Funzione di loss.
    grad_fun : callable
        Gradiente della funzione di loss.
    lam : float
        Parametro di regolarizzazione L2.
    tol : float
        Tolleranza per la convergenza.
    max_iter : int
        Numero massimo di iterazioni
    step_method : callable
        Metodo per calcolare il passo.

    Returns:
    tuple
        Vettore dei pesi di dimensione (p,) e lista delle perdite.
    """
    w = np.random.rand(X.shape[1])
    losses = []
    accuracy = []
    steps = []
    with tqdm(range(max_iter), unit="iter", total=max_iter) as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            grad = grad_fun(X, y, w, lam)
            alpha, backtrack= step_method(X, y, w, lam, loss_fun, -grad, grad_fun)
            w -= alpha * grad
            loss = loss_fun(X, y, w, lam)
            losses.append(loss)
            steps.append(alpha)
            accuracy.append(utils.evaluate_accuracy(X, y, w))
            tepoch.set_postfix(loss=loss, step_iter=backtrack, step_value = alpha, grad_norm=np.linalg.norm(grad), accuracy = accuracy[-1])
            tepoch.update()
            if np.linalg.norm(grad) <= tol:
                print("Tolleranza raggiunta - num iterazioni: " + str(epoch))
                break
    return w, losses, accuracy, steps

def armijo_line_search(X, y, w, lam, f, d, grad_f, delta=0.5, gamma=0.5):   # gamme = 1e-4
    """
    Cerca un passo che soddisfa la condizione di Armijo.
    """
    alpha = 1 # Passo iniziale   - 0.02
    i = 0
    while f(X, y, w + alpha * d, lam) > f(X, y, w, lam) + gamma * alpha * np.dot(grad_f(X, y, w, lam), d): # OTTIMIZZABILE sostituendo con i valori già calcolati
        alpha *= delta  # Riduzione del passo
        i += 1
    #print("it per step: " + str(i))
    return alpha, i

def fixed_step(X, y, w, lam, f, d, grad_f, delta=0.5, gamma=1e-4):
    return 0.02, 0


def armijo_line_search_euristic_initial_step(X, y, w, lam, f, d, grad_f, delta=0.5, gamma=0.5, initial_step = 1, num_backtrack = 0):
    """
    Cerca un passo che soddisfa la condizione di Armijo con passo iniziale scelto con euristica
    """
    #print("Passo iniziale K: " + str(initial_step) + " - Num backtrack: " + str(num_backtrack))
    alpha = euristic_initial_step(initial_step, num_backtrack)
    i = 0
    while f(X, y, w + alpha * d, lam) > f(X, y, w, lam) + gamma * alpha * np.dot(grad_f(X, y, w, lam), d):
        alpha *= delta  # Riduzione del passo
        i += 1
    #print("Passo finale k+1: " + str(alpha) + " - Num backtrack: " + str(i))
    return alpha, i

def euristic_initial_step(alpha, num_backtrack):
    if num_backtrack == 0:
        delta = np.clip(np.random.normal(0.5, 1), 0.1, 0.9)
        #print("Passo ingrandito di un coefficiente 1/" + str(delta))
        alpha = alpha * (1 / (delta))
        alpha = min(alpha, 10) # Il passo esplodeva
        return alpha
    else:
        return alpha


def gradient_descent_euristic_initial_step_armijo(X, y, loss_fun, grad_fun, lam, tol, max_iter, step_method):
    w = np.random.rand(X.shape[1])
    losses = []
    accuracy = []
    steps = []
    num_backtrack = 0
    alpha = 1 # Initial step for euristic
    #for i in range(max_iter):
    with tqdm(range(max_iter), unit="iter", total=max_iter) as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            grad = grad_fun(X, y, w, lam)
            alpha, num_backtrack = step_method(X, y, w, lam, loss_fun, -grad, grad_fun, initial_step = alpha, num_backtrack = num_backtrack)
            w -= alpha * grad
            loss = loss_fun(X, y, w, lam)
            losses.append(loss)
            steps.append(alpha)
            accuracy.append(utils.evaluate_accuracy(X, y, w))
            tepoch.set_postfix(loss=loss, step_iter=num_backtrack, step_value = alpha, grad_norm=np.linalg.norm(grad), accuracy=accuracy[-1])
            tepoch.update()
            if np.linalg.norm(grad) <= tol:
                print("Tolleranza raggiunta - num iterazioni: " + str(epoch))
                break
    return w, losses, accuracy, steps

def polyak_initial_step(X, y, w, lam, f, grad_fun, f_min=1e-6):
    grad_norm_sq = np.linalg.norm(grad_fun(X, y, w, lam)) ** 2
    if grad_norm_sq == 0:
        return 1e-3  # Evita divisioni per zero se si trova in un minimo, restituendo un passo piccolo
    return max((f(X, y, w, lam) - f_min) / grad_norm_sq, 1e-6)  # Imposta un valore minimo per stabilità

def armijo_line_search_polyak_initial_step(X, y, w, lam, f, d, grad_f, tol, delta=0.5, gamma=0.5):
    """
    Cerca un passo che soddisfa la condizione di Armijo con passo iniziale scelto con euristica
    """
    #print("Passo iniziale K: " + str(initial_step) + " - Num backtrack: " + str(num_backtrack))
    alpha = polyak_initial_step(X, y, w, lam, f, grad_f, tol)
    i = 0
    while f(X, y, w + alpha * d, lam) > f(X, y, w, lam) + gamma * alpha * np.dot(grad_f(X, y, w, lam), d):
        alpha *= delta  # Riduzione del passo
        i += 1
    #print("Passo finale k+1: " + str(alpha) + " - Num backtrack: " + str(i))
    return alpha, i

def gradient_descent_polyak_initial_step_armijo(X, y, loss_fun, grad_fun, lam, tol, max_iter, step_method):
    w = np.random.rand(X.shape[1])
    losses = []
    accuracy = []
    steps = []
    num_backtrack = 0
    alpha = 1 # Initial step for euristic
    #for i in range(max_iter):
    with tqdm(range(max_iter), unit="iter", total=max_iter) as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            grad = grad_fun(X, y, w, lam)
            alpha, num_backtrack = step_method(X, y, w, lam, loss_fun, -grad, grad_fun, tol)
            w -= alpha * grad
            loss = loss_fun(X, y, w, lam)
            losses.append(loss)
            steps.append(alpha)
            accuracy.append(utils.evaluate_accuracy(X, y, w))
            tepoch.set_postfix(loss=loss, step_iter=num_backtrack, step_value = alpha, grad_norm=np.linalg.norm(grad), accuracy=accuracy[-1])
            tepoch.update()
            if np.linalg.norm(grad) <= tol:
                print("Tolleranza raggiunta - num iterazioni: " + str(epoch))
                break
    return w, losses, accuracy, steps