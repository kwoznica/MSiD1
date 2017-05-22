# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from utils import polynomial


def mean_squared_error(x, y, w):
    '''
    :param x: ciag wejsciowy Nx1
    :param y: ciag wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: blad sredniokwadratowy pomiedzy wyjsciami y
    oraz wyjsciami uzyskanymi z wielowamiu o parametrach w dla wejsc x
    '''

    return (sum((y - polynomial(x, w)) ** 2) / y.shape[0])[0]


def design_matrix(x_train, M):
    '''
    :param x_train: ciag treningowy Nx1
    :param M: stopien wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzedu M
    '''
    matrix = np.zeros(shape=(len(x_train), M+1))
    length = len(x_train)
    for i in range(length):
        for j in range(M+1):
            matrix[i][j] = x_train[i]**j

    return matrix


def least_squares(x_train, y_train, M):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu, a err blad sredniokwadratowy
    dopasowania
    '''

    matrix = design_matrix(x_train, M)
    w = np.linalg.inv(matrix.transpose() @ matrix) @ matrix.transpose() @ y_train

    return (w, mean_squared_error(x_train, y_train, w))


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu zgodnie z kryterium z regularyzacja l2,
    a err blad sredniokwadratowy dopasowania
    '''
    matrix = design_matrix(x_train, M)
    w = np.linalg.inv(
        matrix.transpose() @ matrix + regularization_lambda * np.eye(M + 1)) @ matrix.transpose() @ y_train

    return (w, mean_squared_error(x_train, y_train, w))


def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, ktore maja byc sprawdzone
    :return: funkcja zwraca krotke (w,train_err,val_err), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym, train_err i val_err to bledy na sredniokwadratowe na ciagach treningowym
    i walidacyjnym
    '''

    train_err, w = 0, 0
    val_err = np.inf

    for i in M_values:
        w_temp, train_err_temp = least_squares(x_train, y_train, i)
        val_err_temp = mean_squared_error(x_val, y_val, w_temp)
        if (val_err > val_err_temp):
            val_err = val_err_temp
            w = w_temp
            train_err = train_err_temp

    return w, train_err, val_err

    pass


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M: stopien wielomianu
    :param lambda_values: lista ze wartosciami roznych parametrow regularyzacji
    :return: funkcja zwraca krotke (w,train_err,val_err,regularization_lambda), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym. Wielomian dopasowany jest wg kryterium z regularyzacja. train_err i val_err to
    bledy na sredniokwadratowe na ciagach treningowym i walidacyjnym. regularization_lambda to najlepsza wartosc parametru regularyzacji
    '''

    w, train_err, regularization_lamba = 0, 0, 0
    val_err = np.inf

    for l in lambda_values:
        w_temp, train_err_temp = regularized_least_squares(x_train, y_train, M, l)
        val_err_temp = mean_squared_error(x_val, y_val, w_temp)
        if val_err > val_err_temp:
            val_err = val_err_temp
            w = w_temp
            train_err = train_err_temp
            regularization_lamba = l

    return w, train_err, val_err, regularization_lamba
    pass
