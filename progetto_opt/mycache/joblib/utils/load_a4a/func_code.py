# first line: 67
@mem.cache
def load_a4a(disp=False):
    path_train = "./dataset/a4a"
    X_train, y_train = load_svmlight_file(path_train)

    path_test = "./dataset/a4a.t"
    X_test, y_test = load_svmlight_file(path_test)

    # add constant column
    X_train_prep = add_intercept(X_train)[:, :120]
    X_test_prep = add_intercept(X_test)[:, :120]

    if disp:
        train_sklearn_log(X_train_prep, y_train, X_test_prep, y_test)

    return X_train_prep, y_train, X_test_prep, y_test
