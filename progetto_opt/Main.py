import functions
import gradient_descent
from gradient_descent import *
import utils
import numpy as np
if __name__ == '__main__':
    X, y = utils.generate_synthetic_data(1000, 100, 42)
    X_test, y_test = utils.generate_synthetic_data(1000, 100, 12)
    w = np.random.rand(X.shape[1])
    w, losses, accuracy, steps = gradient_descent(X, y, functions.logistic_loss, functions.logistic_gradient, 1, 1e-9, 3000, armijo_line_search)
    print("Norma dei pesi: " + str(np.linalg.norm(w)))
    utils.plot(losses, "loss - GD with Armijo", "loss_GD_armijo")
    print("Accuracy sui dati di test: " + str(utils.evaluate_accuracy(X_test, y_test, w)))
    utils.plot(accuracy, "accuracy su training - GD with Armijo", "acc_GD_armijo")
    utils.plot(steps, "Andamento steps - GD with Armijo", "step_GD_armijo")
    w, losses, accuracy, steps = gradient_descent_euristic_initial_step_armijo(X, y, functions.logistic_loss, functions.logistic_gradient, 1, 1e-9, 3000, armijo_line_search_euristic_initial_step)
    print("Norma dei pesi: " + str(np.linalg.norm(w)))
    utils.plot(losses, "loss - GD with Armijo and euristic initial step", "loss_GD_Armijo_noTune")
    print("Accuracy sui dati di test: " + str(utils.evaluate_accuracy(X_test, y_test, w)))
    utils.plot(accuracy, "accuracy su training - GD with Armijo and euristic initial step", "acc_GD_Armijo_noTune")
    utils.plot(steps, "Andamento steps - GD with Armijo and euristic initial step", "Step_GD_Armijo_noTune")