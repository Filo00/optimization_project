import functions
import gradient_descent
from gradient_descent import *
import utils
import numpy as np
if __name__ == '__main__':
    X, y = utils.generate_synthetic_data(1000, 100, 77)
    X_test, y_test = utils.generate_synthetic_data(1000, 100, 12)
    w = np.random.rand(X.shape[1])

    print("GD ARMIJO")
    w, losses, accuracy, steps = gradient_descent(X, y, functions.logistic_loss, functions.logistic_gradient, 1, 1e-9, 3000, armijo_line_search)
    print("Norma dei pesi - GD Armijo: " + str(np.linalg.norm(w)))
    utils.plot(losses, "loss - GD with Armijo", "loss_GD_armijo")
    print("Accuracy sui dati di test - GD with Armijo: " + str(utils.evaluate_accuracy(X_test, y_test, w)))
    utils.plot(accuracy, "accuracy su training - GD with Armijo", "acc_GD_armijo")
    utils.plot(steps, "Andamento steps - GD with Armijo", "step_GD_armijo")

    print("GD ARMIJO NOTUNE")
    w, losses, accuracy, steps = gradient_descent_euristic_initial_step_armijo(X, y, functions.logistic_loss, functions.logistic_gradient, 1, 1e-9, 3000, armijo_line_search_euristic_initial_step)
    print("Norma dei pesi - GD Armijo noTune: " + str(np.linalg.norm(w)))
    utils.plot(losses, "loss - GD Armijo noTune", "loss_GD_Armijo_noTune")
    print("Accuracy sui dati di test - GD Armijo noTune: " + str(utils.evaluate_accuracy(X_test, y_test, w)))
    utils.plot(accuracy, "accuracy su training - GD Armijo noTune", "acc_GD_Armijo_noTune")
    utils.plot(steps, "Andamento steps - GD Armijo noTune", "Step_GD_Armijo_noTune")

    print("GD ARMIJO POLYAK")
    w, losses, accuracy, steps = gradient_descent_polyak_initial_step_armijo(X, y, functions.logistic_loss, functions.logistic_gradient, 1, 1e-9, 3000, armijo_line_search_polyak_initial_step)
    print("Norma dei pesi - GD Armijo polyak: " + str(np.linalg.norm(w)))
    utils.plot(losses, "loss - GD Armijo polyak", "loss_GD_Armijo_polyak")
    print("Accuracy sui dati di test - GD Armijo polyak: " + str(utils.evaluate_accuracy(X_test, y_test, w)))
    utils.plot(accuracy, "accuracy su training - GD Armijo polyak", "acc_GD_Armijo_polyak")
    utils.plot(steps, "Andamento steps - GD Armijo polyak", "Step_GD_Armijo_polyak")

    print("GD nonmonotone")
    w, losses, accuracy, steps = gradient_descent(X, y, functions.logistic_loss, functions.logistic_gradient, 1, 1e-9, 3000, nonmonotone_line_search)
    print("Norma dei pesi - GD nonmonotone: " + str(np.linalg.norm(w)))
    utils.plot(losses, "loss - GD nonmonotone", "loss_GD_nonmonotone")
    print("Accuracy sui dati di test - GD nonmonotone: " + str(utils.evaluate_accuracy(X_test, y_test, w)))
    utils.plot(accuracy, "accuracy su training - GD nonmonotone", "acc_GD_nonmonotone")
    utils.plot(steps, "Andamento steps - GD nonmonotone", "Step_GD_nonmonotone")

    print("GD nonmonotone noTune")
    w, losses, accuracy, steps = gradient_descent_euristic_initial_step_nonmonotone(X, y, functions.logistic_loss, functions.logistic_gradient, 1, 1e-9, 3000, nonmonotone_line_search_euristic_initial_step)
    print("Norma dei pesi - GD nonmonotone noTune: " + str(np.linalg.norm(w)))
    utils.plot(losses, "loss - GD nonmonotone noTune", "loss_GD_nonmonotone_noTune")
    print("Accuracy sui dati di test - GD nonmonotone noTune: " + str(utils.evaluate_accuracy(X_test, y_test, w)))
    utils.plot(accuracy, "accuracy su training - GD nonmonotone noTune", "acc_GD_nonmonotone_noTune")
    utils.plot(steps, "Andamento steps - GD nonmonotone noTune", "Step_GD_nonmonotone_noTune")

    print("GD nonmonotone polyak")
    w, losses, accuracy, steps = gradient_descent_polyak_initial_step_nonmonotone(X, y, functions.logistic_loss, functions.logistic_gradient, 1, 1e-9, 3000, nonmonotone_line_search_polyak_initial_step)
    print("Norma dei pesi - GD nonmonotone polyak: " + str(np.linalg.norm(w)))
    utils.plot(losses, "loss - GD nonmonotone polyak", "loss_GD_nonmonotone_polyak")
    print("Accuracy sui dati di test - GD nonmonotone polyak: " + str(utils.evaluate_accuracy(X_test, y_test, w)))
    utils.plot(accuracy, "accuracy su training - GD nonmonotone polyak", "acc_GD_nonmonotone_polyak")
    utils.plot(steps, "Andamento steps - GD nonmonotone polyak", "Step_GD_nonmonotone_polyak")