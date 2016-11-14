import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import math
import scipy.stats as st

###########################
def WritingInFile(names, sequences, fileName):
    with open(fileName, 'w') as f:
        for i in range(len(names)):
            f.write(names[i] + ': ')
        f.write('\n')
        for j in range(len(sequences[0])):
            for i in range(len(names)):
                f.write(str(sequences[i][j]) + ' ')
            f.write('\n')

def get_xy(fname):
    str_file = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    y = [] 
    with open(fname, 'r') as f:
        for line in f:
            str_file.append(line)
    for i in range(len(str_file)):
        s = str_file[i].expandtabs(1).rstrip()
        x1_el, x2_el, x3_el, x4_el, x5_el, y_el = s.split(' ')
        x1.append(float(x1_el))
        x2.append(float(x2_el))
        x3.append(float(x3_el))
        x4.append(float(x4_el))
        x5.append(float(x5_el))
        y.append(float(y_el))
    return x1, x2, x3, x4, x5, y
#################################
def create_X_matr(x1, x2, x3, x4, x5):
    X = [[1., el1, el2, el3, el4, el5] for el1, el2, el3, el4, el5 in zip(x1, x2, x3, x4, x5)]
    return np.array(X, dtype=float)
################################
def parameter_estimation_tetta(matr_X, Y):
    XtX = np.matmul(matr_X.T, matr_X)
    XtX_1 = np.linalg.inv(XtX)
    XtX_1_Xt = np.matmul(XtX_1, matr_X.T)
    est_tetta = np.matmul(XtX_1_Xt, Y)
    return est_tetta
####################################
def elimination_algorithm(matr_X, y, RSS, N, m, p):
        matr_x = [np.delete(matr_X, np.s_[i], 1) for i in range(p)]
        est_theta = [parameter_estimation_tetta(el, y) for el in matr_x]
        est_y = [np.matmul(el1, el2) for el1, el2 in zip(matr_x, est_theta)]
        RSS_1 = [np.matmul((y - el).T, (y - el)) for el in est_y]
        F = [(N - m) * (el - RSS) / el for el in RSS_1] 
        return F
    ####################
def quality_criterion(sigm_2, matr_X, y, p, Ys, N):
        est_theta_1 = parameter_estimation_tetta(matr_X, y)
        est_y_1 = np.matmul(matr_X, est_theta_1)
        y_est_y_1 = y - est_y_1
        RSS1 = np.matmul((y - est_y_1).T, (y - est_y_1))
        C_p1 = RSS1 / sigm_2 + 2 * p - N
        est_y_mean = np.mean(est_y_1)
        est_Y_s1 = np.sum(np.array([(el - est_y_mean)** 2 for el in est_y_1]))
        R = est_Y_s1 / Ys
        E = (RSS1 / (N * (N - p))) * (1 + N + ((p * (N + 1)) / (N - p - 2)))
        AEV = (p * RSS1) / (N *(N - p))
        return C_p1, R, E, AEV, est_theta_1, est_y_1, y_est_y_1
###################################
def Graph(x, y):
    p1 = plt.plot(x, y, 'r')
    plt.show()
##################################
def model_base(y, matr_X, N, m):
    p = m
    est_theta = parameter_estimation_tetta(matr_X, y)
    RSS = np.matmul((y - np.matmul(matr_X, est_theta)).T, (y - np.matmul(matr_X, est_theta)))
    sigm_2 = RSS / (N - m)
    y_mean = np.mean(y)
    Ys = np.sum(np.array([(el - y_mean) ** 2 for el in y]))
    Cp, R, E, AEV, est_theta, est_y, y_est_y = quality_criterion(sigm_2, matr_X, y, p, Ys, N)
    f1 = elimination_algorithm(matr_X, y, RSS, N, m, p)
    #####исключаем последний регрессор
    matr_X1 = matr_X[:,:5]
    p = m - 1
    Cp1, R1, E1, AEV1, est_theta_1, est_y_1, y_est_y_1 = quality_criterion(sigm_2, matr_X1, y, p, Ys, N)
    f2 = elimination_algorithm(matr_X1, y, RSS, N, m, p)
    ##########################
    matr_X2 = np.delete(matr_X1, np.s_[3], 1)
    p = m - 2
    Cp2, R2, E2, AEV2, est_theta_2, est_y_2, y_est_y_2 = quality_criterion(sigm_2, matr_X2, y, p, Ys, N)
    f3 = elimination_algorithm(matr_X2, y, RSS, N, m, p)
    ##############################
    matr_X3 = np.delete(matr_X2, np.s_[1], 1)
    p = m - 3
    Cp3, R3, E3, AEV3, est_theta_3, est_y_3, y_est_y_3 = quality_criterion(sigm_2, matr_X3, y, p, Ys, N)
    f4 = elimination_algorithm(matr_X3, y, RSS, N, m, p)
    ##############
    matr_X4 = np.delete(matr_X3, np.s_[2], 1)
    p = m - 4
    Cp4, R4, E4, AEV4, est_theta_4, est_y_4, y_est_y_4 = quality_criterion(sigm_2, matr_X4, y, p, Ys, N)
    f5 = elimination_algorithm(matr_X4, y, RSS, N, m, p)
    ############
    matr_X5 = np.delete(matr_X4, np.s_[1], 1)
    p = m - 5
    Cp5, R5, E5, AEV5, est_theta_5, est_y_5, y_est_y_5 = quality_criterion(sigm_2, matr_X5, y, p, Ys, N)
    f6 = elimination_algorithm(matr_X5, y, RSS, N, m, p)
    C = [Cp, Cp1, Cp2, Cp3, Cp4, Cp5]
    R_ = [R, R1, R2, R3, R4, R5]
    E_ = [E, E1, E2, E3, E4, E5]
    AEV_ = [AEV, AEV1, AEV2, AEV3, AEV4, AEV5]
    return C, R_, E_, AEV_