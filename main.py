import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f_x(x):
    x2 = x[0]
    x3 = x[1]
    f = 5 * x2 ** 2 + 12 * x2 * x3 - 8 * x2 + 10 * x3 ** 2 - 14 * x3 - 5
    return f
def g_x(x):
    x2 = x[0]
    x3 = x[1]
    g1 = 10 * x2 + 12 * x3 - 8
    g2 = 12 * x2 + 20 * x3 - 14
    g = np.array([g1, g2])
    return g
def H_x(x):
    H = np.array([[10, 12], [12, 20]])
    return H
def phi_alpha(alpha, t, f, g):
    phi = f - t*g.transpose() @ g*alpha
    return phi
def Inexact_Line_Search(XN, t, alpha, max_iter):
    iter = 0
    f = f_x(XN)
    g = g_x(XN)
    f_x_ag = f_x(XN - alpha * g)
    phi = phi_alpha(alpha, t, f, g)
    while f_x_ag > phi and iter < max_iter:
        f_x_ag = f_x(XN - alpha * g)
        phi = phi_alpha(alpha, t, f, g)
        alpha = 0.5 * alpha
        iter += 1
    return alpha
def Gradient_Descent(X0, t, alpha0, eps, max_iter):
    results = []
    XN = X0
    f = f_x(XN)
    g = g_x(XN)
    iter = 0
    results.append([iter, f, g[0], g[1], alpha0, np.linalg.norm(g)])
    while np.linalg.norm(g) > eps and iter < max_iter:
        alpha = Inexact_Line_Search(XN, t, alpha0, max_iter)
        XN = XN - alpha*g
        f = f_x(XN)
        g = g_x(XN)
        iter += 1
        results.append([iter, f, g[0], g[1], alpha, np.linalg.norm(g)])
    return XN, results
def Convergence(f_list, f_star, X0, method):
    X2_0 = int(X0[0])
    X3_0 = int(X0[1])
    error = np.zeros(len(f_list))
    for i in range(0, len(f_list)):
        error[i] = abs(f_list[i] - f_star)
    plt.figure()
    plt.ylabel(r'|$f_k$-$f^{*}$|')
    plt.yscale('log')
    plt.xlabel('Iteration #, k')
    if method == 'GD':
        plt.plot(error)
        plt.ylim([1e-13, 10])
        plt.xlim([0, 90])
        plt.plot([0, 1], [4.92857,1e-13])
        plt.ylim([1e-13, 10])
        plt.xlim([0, 1])

def Newton(X0, eps, max_iter):
    results = []
    XN = X0
    # f = f_x(XN)
    # g = g_x(XN)
    # H = H_x(XN)
    iter = 0
    diff = 1
    while diff > eps and iter < max_iter:
        f = f_x(XN)
        g = g_x(XN)
        H = H_x(XN)
        XNp1 = XN - np.linalg.inv(H) @ g
        diff = abs(np.linalg.norm(XN) - np.linalg.norm(XNp1))
        results.append([iter, f, diff])
        iter += 1
        XN = XNp1
    return XN, results
t = 0.5
alpha = 1
eps = 1e-6
max_iter = 1000
x2_0 = 0
x3_0 = 0
X0 = np.array([[x2_0], [x3_0]])
XN, results = Gradient_Descent(X0, t, alpha, eps, max_iter)
Results = pd.DataFrame(results, columns=['iter', 'f', 'g[0]', 'g[1]', 'alpha', 'norm(g)'])
f_list = np.array(Results['f'])
x_star = np.array([[-1/7], [11/14]])
f_star = f_x(x_star)
Convergence(f_list, f_star, X0, 'GD')
Total_Iterations = len(f_list)-1
print(XN)
print('\n'+str(Total_Iterations))
XN, Results_NM = Newton(X0, eps, max_iter)
Results_NM = pd.DataFrame(Results_NM, columns=['iter', 'f', 'diff'])
f_list = np.array(Results_NM['f'])
Convergence(f_list, f_star, X0, 'NM')
Iterations = len(Results_NM)-1
print('\n')
print (XN)
print('\n'+str(Iterations))