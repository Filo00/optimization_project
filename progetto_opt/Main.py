import functions
import gradient_descent
from gradient_descent import *
import utils
import numpy as np
if __name__ == '__main__':
    losses = []
    accuracy = []
    steps = []
    X, y = utils.generate_synthetic_data(1000, 100, 77)
    X_test, y_test = utils.generate_synthetic_data(1000, 100, 12)
    w = np.random.rand(X.shape[1])
    w, losses, accuracy, steps = gradient_descent(X, y, functions.logistic_loss, functions.logistic_gradient, 0.1, 1e-6, 3000, armijo_line_search)
    print("Norma dei pesi: " + str(np.linalg.norm(w)))
    utils.plot(losses, title="loss - GD with Armijo")
    print("Accuracy sui dati di test: " + str(utils.evaluate_accuracy(X_test, y_test, w)))
    utils.plot(accuracy, title="accuracy su training - GD with Armijo")
    utils.plot(steps, title="Andamento steps - GD with Armijo")
    losses = []
    accuracy = []
    steps = []
    w, losses, accuracy, steps = gradient_descent_euristic_initial_step_armijo(X, y, functions.logistic_loss, functions.logistic_gradient, 0.01, 1e-6, 3000, armijo_line_search_euristic_initial_step)
    print("Norma dei pesi: " + str(np.linalg.norm(w)))
    utils.plot(losses, title = "loss - GD with Armijo and euristic initial step")
    print("Accuracy sui dati di test: " + str(utils.evaluate_accuracy(X_test, y_test, w)))
    utils.plot(accuracy, title = "accuracy su training - GD with Armijo and euristic initial step")
    utils.plot(steps, title="Andamento steps - GD with Armijo and euristic initial step")