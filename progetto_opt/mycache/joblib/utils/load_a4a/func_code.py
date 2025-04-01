# first line: 56
@mem.cache
def load_a4a():
    path_train = "./dataset/a4a"
    X_train, y_train = load_svmlight_file(path_train)

    path_test = "./dataset/a4a.t"
    X_test, y_test = load_svmlight_file(path_test)

    # add constant column
    X_train_prep = add_intercept(X_train)[:, :120]
    X_test_prep = add_intercept(X_test)[:, :120]
    return X_train_prep, y_train, X_test_prep, y_test, "a4a"
