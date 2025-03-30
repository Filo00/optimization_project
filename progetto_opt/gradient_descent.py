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
    sharp_list = []
    sharp_stepsize = []
    with tqdm(range(max_iter), unit="iter", total=max_iter) as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            grad = grad_fun(X, y, w, lam)
            w_old = w
            alpha, backtrack= step_method(X, y, w, lam, loss_fun, -grad, grad_fun)
            w -= alpha * grad
            sharp = evaluate_sharpness(X, y, w_old, w, lam, loss_fun, grad_fun, alpha)
            loss = loss_fun(X, y, w, lam)
            losses.append(loss)
            steps.append(alpha)
            sharp_list.append(sharp)
            sharp_stepsize.append(sharp * alpha)
            accuracy.append(utils.evaluate_accuracy(X, y, w))
            tepoch.set_postfix(loss=loss, step_iter=backtrack, step_value = alpha, grad_norm=np.linalg.norm(grad), accuracy = accuracy[-1])
            tepoch.update()
            if np.linalg.norm(grad) <= tol:
                print("Tolleranza raggiunta - num iterazioni: " + str(epoch))
                break
    return w, losses, accuracy, steps, sharp_list, sharp_stepsize

def armijo_line_search(X, y, w, lam, f, d, grad_f, delta=0.5, gamma=0.8):   # gamme = 1e-4
    """
    Cerca un passo che soddisfa la condizione di Armijo.
    """
    alpha = 1 # Passo iniziale   - 0.02
    i = 0
    while f(X, y, w + alpha * d, lam) > f(X, y, w, lam) + gamma * alpha * np.dot(grad_f(X, y, w, lam), d): # OTTIMIZZABILE sostituendo con i valori già calcolati
        alpha *= delta  # Riduzione del passo
        i += 1
    #print("it per step: " + str(i))
    #alpha = max(alpha, 10)
    return alpha, i

def fixed_step(X, y, w, lam, f, d, grad_f, delta=0.5, gamma=1e-4):
    return 0.02, 0

def evaluate_sharpness(X, y, w_old, w, lam, loss_fun, grad_fun, alpha):
    # DUBBIO, l'alpha deve essere del w o del w_old?
    squared_norm_funct = np.linalg.norm(grad_fun(X, y, w_old, lam)) ** 2
    Lapprox = ((2 * (loss_fun(X, y, w, lam) - loss_fun(X, y, w_old, lam))) /
               ((alpha ** 2) * squared_norm_funct)) + 2 / alpha
    return Lapprox

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
    #alpha = max(alpha, 10)
    return alpha, i

def euristic_initial_step(alpha, num_backtrack):
    if num_backtrack == 0:
        delta = np.clip(np.random.normal(0.5, 1), 0.1, 0.9)
        #print("Passo ingrandito di un coefficiente 1/" + str(delta))
        alpha = alpha * (1 / (delta))
        #alpha = min(alpha, 10)
        return min(alpha, 20)
    else:
        return alpha


def gradient_descent_euristic_initial_step_armijo(X, y, loss_fun, grad_fun, lam, tol, max_iter, step_method):
    w = np.random.rand(X.shape[1])
    losses = []
    accuracy = []
    steps = []
    sharp_list = []
    sharp_stepsize = []
    num_backtrack = 0
    alpha = 1 # Initial step for euristic
    #for i in range(max_iter):
    with tqdm(range(max_iter), unit="iter", total=max_iter) as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            grad = grad_fun(X, y, w, lam)
            w_old = w
            alpha, num_backtrack = step_method(X, y, w, lam, loss_fun, -grad, grad_fun, initial_step = alpha, num_backtrack = num_backtrack)
            w -= alpha * grad
            sharp = evaluate_sharpness(X, y, w_old, w, lam, loss_fun, grad_fun, alpha)
            loss = loss_fun(X, y, w, lam)
            losses.append(loss)
            steps.append(alpha)
            sharp_list.append(sharp)
            sharp_stepsize.append(sharp * alpha)
            accuracy.append(utils.evaluate_accuracy(X, y, w))
            tepoch.set_postfix(loss=loss, step_iter=num_backtrack, step_value = alpha, grad_norm=np.linalg.norm(grad), accuracy=accuracy[-1])
            tepoch.update()
            if np.linalg.norm(grad) <= tol:
                print("Tolleranza raggiunta - num iterazioni: " + str(epoch))
                break
    return w, losses, accuracy, steps, sharp_list, sharp_stepsize

def polyak_initial_step(X, y, w, lam, f, grad_fun, f_min=1e-6):
    grad_norm_sq = np.linalg.norm(grad_fun(X, y, w, lam)) ** 2
    if grad_norm_sq == 0:
        return 1e-6  # Evita divisioni per zero se si trova in un minimo, restituendo un passo piccolo
    alpha = (f(X, y, w, lam) - f_min) / grad_norm_sq # f(x) - f_min / ||grad f(x)||^2
    alpha = min(alpha, 20)
    return alpha

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
    #alpha = max(alpha, 10) # Nel paper limitano a 10
    return alpha, i

def gradient_descent_polyak_initial_step_armijo(X, y, loss_fun, grad_fun, lam, tol, max_iter, step_method):
    w = np.random.rand(X.shape[1])
    losses = []
    accuracy = []
    steps = []
    sharp_list = []
    sharp_stepsize = []
    #for i in range(max_iter):
    with tqdm(range(max_iter), unit="iter", total=max_iter) as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            grad = grad_fun(X, y, w, lam)
            alpha, num_backtrack = step_method(X, y, w, lam, loss_fun, -grad, grad_fun, tol)
            w_old = w
            w -= alpha * grad
            sharp = evaluate_sharpness(X, y, w_old, w, lam, loss_fun, grad_fun, alpha)
            loss = loss_fun(X, y, w, lam)
            losses.append(loss)
            steps.append(alpha)
            sharp_list.append(sharp)
            sharp_stepsize.append(sharp * alpha)
            accuracy.append(utils.evaluate_accuracy(X, y, w))
            tepoch.set_postfix(loss=loss, step_iter=num_backtrack, step_value = alpha, grad_norm=np.linalg.norm(grad), accuracy=accuracy[-1])
            tepoch.update()
            if np.linalg.norm(grad) <= tol:
                print("Tolleranza raggiunta - num iterazioni: " + str(epoch))
                break
    return w, losses, accuracy, steps, sharp_list, sharp_stepsize


def nonmonotone_line_search(X, y, w, lam, f, d, grad_f, Ck, Qk, xi=0.5, delta=0.5, gamma=0.5):
    alpha = 1
    i = 0
    Qk_new = xi * Qk + 1
    C_tilde = (xi * Qk * Ck + f(X, y, w, lam)) / (Qk_new) # Ck_tilde = (xi * Qk * Ck + f(x)) / (Qk + 1)
    Ck_new = max(C_tilde, f(X, y, w, lam)) # Ck = max {Ck_tilde, f(x)}

    # Condizione nonmonotona
    while f(X, y, w + alpha * d, lam) > Ck_new + gamma * alpha * np.dot(grad_f(X, y, w, lam), d): # f(x + alpha * d) > Ck - gamma * alpha * grad_f(x)^T * d
        alpha *= delta  # Backtracking
        i += 1
    #print(f"Passo finale: {alpha} - Num backtrack: {i}")
    #alpha = max(alpha, 10)
    return alpha, i, Ck_new, Qk_new

def gradient_descent_nonmonotone(X, y, loss_fun, grad_fun, lam, tol, max_iter, step_method):
    w = np.random.rand(X.shape[1])
    losses = []
    accuracy = []
    steps = []
    sharp_list = []
    sharp_stepsize = []
    Ck = 0
    Qk = 0
    with tqdm(range(max_iter), unit="iter", total=max_iter) as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            grad = grad_fun(X, y, w, lam)
            alpha, num_backtrack, Ck, Qk = step_method(X, y, w, lam, loss_fun, -grad, grad_fun, Ck, Qk)
            w_old = w
            w -= alpha * grad
            sharp = evaluate_sharpness(X, y, w_old, w, lam, loss_fun, grad_fun, alpha)
            loss = loss_fun(X, y, w, lam)
            losses.append(loss)
            steps.append(alpha)
            sharp_list.append(sharp)
            sharp_stepsize.append(sharp * alpha)
            accuracy.append(utils.evaluate_accuracy(X, y, w))
            tepoch.set_postfix(loss=loss, step_iter=num_backtrack, step_value = alpha, grad_norm=np.linalg.norm(grad), accuracy=accuracy[-1], Ck=Ck, Qk=Qk)
            tepoch.update()
            if np.linalg.norm(grad) <= tol:
                print("Tolleranza raggiunta - num iterazioni: " + str(epoch))
                break
    return w, losses, accuracy, steps, sharp_list, sharp_stepsize


def nonmonotone_line_search_euristic_initial_step(X, y, w, lam, f, d, grad_f, Ck, Qk, xi=0.5, delta=0.5, gamma=0.5, initial_step = 1, num_backtrack = 0):
    """
    Cerca un passo che soddisfa la condizione nonmonotona con passo iniziale scelto con euristica
    """
    #print("Passo iniziale K: " + str(initial_step) + " - Num backtrack: " + str(num_backtrack))
    alpha = euristic_initial_step(initial_step, num_backtrack)
    i = 0
    Qk_new = xi * Qk + 1
    C_tilde = (xi * Qk * Ck + f(X, y, w, lam)) / (Qk_new)
    Ck_new = max(C_tilde, f(X, y, w, lam))
    # Condizione nonmonotona
    while f(X, y, w + alpha * d, lam) > Ck_new + gamma * alpha * np.dot(grad_f(X, y, w, lam), d):
        alpha *= delta  # Backtracking
        i += 1
    #print("Passo finale k+1: " + str(alpha) + " - Num backtrack: " + str(i))
    #alpha = max(alpha, 10)
    return alpha, i, Ck_new, Qk_new

def gradient_descent_euristic_initial_step_nonmonotone(X, y, loss_fun, grad_fun, lam, tol, max_iter, step_method):
    w = np.random.rand(X.shape[1])
    losses = []
    accuracy = []
    steps = []
    sharp_list = []
    sharp_stepsize = []
    Ck = 0
    Qk = 0
    num_backtrack = 0
    alpha = 1 # Initial step for euristic
    with tqdm(range(max_iter), unit="iter", total=max_iter) as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            grad = grad_fun(X, y, w, lam)
            alpha, num_backtrack, Ck, Qk = step_method(X, y, w, lam, loss_fun, -grad, grad_fun, Ck, Qk, xi=0.5, delta=0.5, gamma=0.5, initial_step = alpha, num_backtrack = num_backtrack)
            w_old = w
            w -= alpha * grad
            sharp = evaluate_sharpness(X, y, w_old, w, lam, loss_fun, grad_fun, alpha)
            loss = loss_fun(X, y, w, lam)
            losses.append(loss)
            steps.append(alpha)
            sharp_list.append(sharp)
            sharp_stepsize.append(sharp * alpha)
            accuracy.append(utils.evaluate_accuracy(X, y, w))
            tepoch.set_postfix(loss=loss, step_iter=num_backtrack, step_value = alpha, grad_norm=np.linalg.norm(grad), accuracy=accuracy[-1], Ck=Ck, Qk=Qk)
            tepoch.update()
            if np.linalg.norm(grad) <= tol:
                print("Tolleranza raggiunta - num iterazioni: " + str(epoch))
                break
    return w, losses, accuracy, steps, sharp_list, sharp_stepsize

def nonmonotone_line_search_polyak_initial_step(X, y, w, lam, f, d, grad_f, Ck, Qk, xi=0.5, delta=0.5, gamma=0.5):
    #print("Passo iniziale K: " + str(initial_step) + " - Num backtrack: " + str(num_backtrack))
    alpha = polyak_initial_step(X, y, w, lam, f, grad_f)
    i = 0
    Qk_new = xi * Qk + 1
    C_tilde = (xi * Qk * Ck + f(X, y, w, lam)) / (Qk_new)
    Ck_new = max(C_tilde, f(X, y, w, lam))
    # Condizione nonmonotona
    while f(X, y, w + alpha * d, lam) > Ck_new + gamma * alpha * np.dot(grad_f(X, y, w, lam), d):
        alpha *= delta  # Backtracking
        i += 1
    #print("Passo finale k+1: " + str(alpha) + " - Num backtrack: " + str(i))
    #alpha = max(alpha, 10)
    return alpha, i, Ck_new, Qk_new

def gradient_descent_polyak_initial_step_nonmonotone(X, y, loss_fun, grad_fun, lam, tol, max_iter, step_method):
    w = np.random.rand(X.shape[1])
    losses = []
    accuracy = []
    steps = []
    sharp_list = []
    sharp_stepsize = []
    Ck = 0
    Qk = 0
    with tqdm(range(max_iter), unit="iter", total=max_iter) as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            grad = grad_fun(X, y, w, lam)
            alpha, num_backtrack, Ck, Qk = step_method(X, y, w, lam, loss_fun, -grad, grad_fun, Ck, Qk)
            w_old = w
            w -= alpha * grad
            sharp = evaluate_sharpness(X, y, w_old, w, lam, loss_fun, grad_fun, alpha)
            loss = loss_fun(X, y, w, lam)
            losses.append(loss)
            steps.append(alpha)
            sharp_list.append(sharp)
            sharp_stepsize.append(sharp * alpha)
            accuracy.append(utils.evaluate_accuracy(X, y, w))
            tepoch.set_postfix(loss=loss, step_iter=num_backtrack, step_value = alpha, grad_norm=np.linalg.norm(grad), accuracy=accuracy[-1], Ck=Ck, Qk=Qk)
            tepoch.update()
            if np.linalg.norm(grad) <= tol:
                print("Tolleranza raggiunta - num iterazioni: " + str(epoch))
                break
    return w, losses, accuracy, steps, sharp_list, sharp_stepsize