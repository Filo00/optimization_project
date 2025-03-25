import numpy as np
import matplotlib.pyplot as plt
from functions import sigmoid
from joblib import Memory
from sklearn.datasets import load_svmlight_file
from scipy.sparse import lil_matrix, hstack

def generate_synthetic_data(N=100, p=10, seed=42):
    np.random.seed(seed)
    X = np.random.randn(N, p)
    w_true = np.random.randn(p)
    y = sigmoid(X.dot(w_true) + np.random.randn(N))
    y = np.where(y > 0.5, 1, 0)
    return X, y

def plot_fun(losses, title="", nameFile=""):
    path = "./plot/" + nameFile + ".pdf"
    plt.plot(losses, label=title)
    plt.xlabel('Iterazioni')
    plt.ylabel(title)
    plt.title("Andamento " + str(title))
    plt.xscale("log")
    plt.legend()
    plt.savefig(path)
    #plt.show()


def plot(losses, dir, title="", nameFile=""):
    path = "./plot/" + dir + "/" + nameFile + ".pdf"
    fig, ax = plt.subplots()
    ax.plot(losses, color="b")

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('Iterazioni')
    ax.set_ylabel(title)

    # Add a grid
    ax.grid(True)
    plt.savefig(path)

def evaluate_accuracy(X_test, y_test, w):
    from functions import sigmoid
    prob = sigmoid(X_test.dot(w)) # probabilità (input logits)
    y_pred = np.where(prob > 0.5, 1, -1)
    accuracy = np.mean(y_pred == y_test)
    return accuracy


mem = Memory("./mycache")
@mem.cache
def load_a4a(): # a4a dataset, 123 features, 4781 samples, test set 27780 samples, 2 class
    path_train = "./dataset/a4a"
    X_train, y_train = load_svmlight_file(path_train)

    path_test = "./dataset/a4a.t"
    X_test, y_test = load_svmlight_file(path_test)

    # add constant column
    X_train_prep = add_intercept(X_train)[:, :120]
    X_test_prep = add_intercept(X_test)[:, :120]
    return X_train_prep, y_train, X_test_prep, y_test

def add_intercept(X):
    ones = lil_matrix(np.ones((X.shape[0], 1)))
    X_prep = hstack([ones, X.tolil()], format="csr")
    return X_prep
