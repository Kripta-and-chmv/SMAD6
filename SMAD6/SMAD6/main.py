import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import func as f

N = 17
m = 6
x = [6, 5, 4, 3, 2, 1]
x1, x2, x3, x4, x5, y = f.get_xy('x.txt')
#f.Graph(x1, y)
#f.Graph(x2, y)
#f.Graph(x3, y)
#f.Graph(x4, y)
#f.Graph(x5, y)
matr_X = f.create_X_matr(x1, x2, x3, x4, x5)
C, R, E, AEV = f.model_base(y, matr_X, N, m)
f.Graph(x, C)
f.Graph(x, R)
f.Graph(x, E)
f.Graph(x, AEV)