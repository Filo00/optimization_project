import functions
import gradient_descent
from gradient_descent import *
import utils
import numpy as np


if __name__ == '__main__':
    #X, y = utils.generate_synthetic_data(10000, 100, 69)
    #X_test, y_test = utils.generate_synthetic_data(10000, 100, 12)
    X, y, X_test, y_test = utils.load_a4a()
    w = np.random.rand(X.shape[1])

    print("GD ARMIJO")
    w, losses, accuracy, steps, sharp, sharp_stepsize = gradient_descent(X, y, functions.logistic_loss, functions.logistic_gradient, 0.01, 1e-6, 10000, armijo_line_search)
    print("Norma dei pesi - GD Armijo: " + str(np.linalg.norm(w)))
    utils.plot(losses, "Armijo","loss - GD with Armijo", "loss_GD_armijo", "log", "log")
    print("Accuracy sui dati di test - GD with Armijo: " + str(utils.evaluate_accuracy(X_test, y_test, w)))
    utils.plot(accuracy, "Armijo","accuracy su training - GD with Armijo", "acc_GD_armijo")
    utils.plot(steps, "Armijo","Andamento steps - GD with Armijo", "step_GD_armijo", "log", "log")
    utils.plot(sharp, "Armijo", "Andamento sharpness - GD with Armijo", "sharpness_GD_armijo", "log", "log")
    utils.plot(sharp_stepsize, "Armijo", "Shaprness * stepsize - GD with Armijo", "sharpness_stepsize_GD_armijo", "log")

    print("GD ARMIJO NOTUNE")
    w, losses, accuracy, steps, sharp, sharp_stepsize = gradient_descent_euristic_initial_step_armijo(X, y, functions.logistic_loss, functions.logistic_gradient, 0.01, 1e-6, 10000, armijo_line_search_euristic_initial_step)
    print("Norma dei pesi - GD Armijo noTune: " + str(np.linalg.norm(w)))
    utils.plot(losses, "Armijo_noTune", "loss - GD Armijo noTune", "loss_GD_Armijo_noTune", "log")
    print("Accuracy sui dati di test - GD Armijo noTune: " + str(utils.evaluate_accuracy(X_test, y_test, w)))
    utils.plot(accuracy, "Armijo_noTune",  "accuracy su training - GD Armijo noTune", "acc_GD_Armijo_noTune")
    utils.plot(steps, "Armijo_noTune", "Andamento steps - GD Armijo noTune", "Step_GD_Armijo_noTune")
    utils.plot(sharp, "Armijo_noTune", "Andamento sharpness - GD with Armijo noTune", "sharpness_GD_armijo_noTune", "log")
    utils.plot(sharp_stepsize, "Armijo_noTune", "Shaprness * stepsize - GD with Armijo noTune", "sharpness_stepsize_GD_armijo_noTune", "log")

    print("GD ARMIJO POLYAK")
    w, losses, accuracy, steps, sharp, sharp_stepsize = gradient_descent_polyak_initial_step_armijo(X, y, functions.logistic_loss, functions.logistic_gradient, 0.01, 1e-6, 10000, armijo_line_search_polyak_initial_step)
    print("Norma dei pesi - GD Armijo polyak: " + str(np.linalg.norm(w)))
    utils.plot(losses, "Armijo_Polyak", "loss - GD Armijo polyak", "loss_GD_Armijo_polyak", "log")
    print("Accuracy sui dati di test - GD Armijo polyak: " + str(utils.evaluate_accuracy(X_test, y_test, w)))
    utils.plot(accuracy, "Armijo_Polyak","accuracy su training - GD Armijo polyak", "acc_GD_Armijo_polyak")
    utils.plot(steps, "Armijo_Polyak","Andamento steps - GD Armijo polyak", "Step_GD_Armijo_polyak")
    utils.plot(sharp, "Armijo_Polyak", "Andamento sharpness - GD Armijo polyak", "sharpness_GD_armijo_polyak",
               "log")
    utils.plot(sharp_stepsize, "Armijo_Polyak", "Shaprness * stepsize - GD Armijo polyak",
               "sharpness_stepsize_GD_armijo_polyak", "log")

    print("GD nonmonotone")
    w, losses, accuracy, steps, sharp, sharp_stepsize = gradient_descent_nonmonotone(X, y, functions.logistic_loss, functions.logistic_gradient, 0.01, 1e-6, 10000, nonmonotone_line_search)
    print("Norma dei pesi - GD nonmonotone: " + str(np.linalg.norm(w)))
    utils.plot(losses, "Nonmonotone", "loss - GD nonmonotone", "loss_GD_nonmonotone", "log")
    print("Accuracy sui dati di test - GD nonmonotone: " + str(utils.evaluate_accuracy(X_test, y_test, w)))
    utils.plot(accuracy,"Nonmonotone",  "accuracy su training - GD nonmonotone", "acc_GD_nonmonotone")
    utils.plot(steps, "Nonmonotone", "Andamento steps - GD nonmonotone", "Step_GD_nonmonotone")
    utils.plot(sharp, "Nonmonotone", "Andamento sharpness - GD nonmonotone", "sharpness_GD_nonmonotone",
               "log")
    utils.plot(sharp_stepsize, "Nonmonotone", "Shaprness * stepsize - GD nonmonotone",
               "sharpness_stepsize_GD_nonmonotone", "log")

    print("GD nonmonotone noTune")
    w, losses, accuracy, steps, sharp, sharp_stepsize = gradient_descent_euristic_initial_step_nonmonotone(X, y, functions.logistic_loss, functions.logistic_gradient, 0.01, 1e-6, 10000, nonmonotone_line_search_euristic_initial_step)
    print("Norma dei pesi - GD nonmonotone noTune: " + str(np.linalg.norm(w)))
    utils.plot(losses, "Nonmonotone_noTune", "loss - GD nonmonotone noTune", "loss_GD_nonmonotone_noTune", "log")
    print("Accuracy sui dati di test - GD nonmonotone noTune: " + str(utils.evaluate_accuracy(X_test, y_test, w)))
    utils.plot(accuracy, "Nonmonotone_noTune", "accuracy su training - GD nonmonotone noTune", "acc_GD_nonmonotone_noTune")
    utils.plot(steps, "Nonmonotone_noTune",  "Andamento steps - GD nonmonotone noTune", "Step_GD_nonmonotone_noTune", "log")
    utils.plot(sharp, "Nonmonotone_noTune", "Andamento sharpness - GD nonmonotone noTune", "sharpness_GD_nonmonotone_noTune",
               "log")
    utils.plot(sharp_stepsize, "Nonmonotone_noTune", "Shaprness * stepsize - GD nonmonotone noTune",
               "sharpness_stepsize_GD_nonmonotone_noTune", "log")

    print("GD nonmonotone polyak")
    w, losses, accuracy, steps, sharp, sharp_stepsize = gradient_descent_polyak_initial_step_nonmonotone(X, y, functions.logistic_loss, functions.logistic_gradient, 0.01, 1e-6, 10000, nonmonotone_line_search_polyak_initial_step)
    print("Norma dei pesi - GD nonmonotone polyak: " + str(np.linalg.norm(w)))
    utils.plot(losses, "Nonmonotone_Polyak", "loss - GD nonmonotone polyak", "loss_GD_nonmonotone_polyak", "log")
    print("Accuracy sui dati di test - GD nonmonotone polyak: " + str(utils.evaluate_accuracy(X_test, y_test, w)))
    utils.plot(accuracy,"Nonmonotone_Polyak", "accuracy su training - GD nonmonotone polyak", "acc_GD_nonmonotone_polyak")
    utils.plot(steps,"Nonmonotone_Polyak", "Andamento steps - GD nonmonotone polyak", "Step_GD_nonmonotone_polyak", "log")
    utils.plot(sharp, "Nonmonotone_Polyak", "Andamento sharpness - GD nonmonotone polyak","sharpness_GD_nonmonotone_polyak",
               "log")
    utils.plot(sharp_stepsize, "Nonmonotone_Polyak", "Shaprness * stepsize - GD nonmonotone polyak","sharpness_stepsize_GD_nonmonotone_polyak", "log")
