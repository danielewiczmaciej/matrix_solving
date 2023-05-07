import numpy as np
import math
import time
import matplotlib.pyplot as plt

def make_matrix(n=955, w=5, a1=10, a2=-1, a3=-1):
    # Initialize matrix with zeros
    matrix = [[0 for j in range(n)] for i in range(n)]

    for i in range(n):
        matrix[i][i] = a1

    for i in range(n - w):
        matrix[i + 1][i] = a2
        matrix[i][i + 1] = a2

    for i in range(n - 2 * w):
        matrix[i + 2][i] = a3
        matrix[i][i + 2] = a3

    for i in range(n - w):
        matrix[i][i + w] = a2
        matrix[i + w][i] = a2

    for i in range(n - 2 * w):
        matrix[i][i + 2 * w] = a3
        matrix[i + 2 * w][i] = a3

    return matrix


def make_b(n=955):
    return np.array([[math.sin(i*9)] for i in range(n)])

def LUDecompDoolittle(A):
    n = len(A)
    LU = np.array(A, dtype=float)

    # decomposition of matrix, Doolittle's Method
    for i in range(n):
        for j in range(i):
            LU[i, j] = (LU[i, j] - LU[i, :j] @ LU[:j, j]) / LU[j, j]

        j = slice(i, n)
        LU[i, j] = LU[i, j] - LU[i, :i] @ LU[:i, j]

    return LU


def SolveLinearSystem_LU(LU, B):
    n = len(LU)
    y = np.zeros_like(B, dtype=float)

    # find solution of Ly = B
    for i in range(n):
        y[i, :] = B[i, :] - LU[i, :i] @ y[:i, :]

    # find solution of Ux = y
    x = np.zeros_like(B, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i, :] = (y[i, :] - LU[i, i + 1:] @ x[i + 1:, :]) / LU[i, i]

    return x


def jacobi(A, b, tol=1e-10, max_iter=10000000):
    n = len(A)
    x = np.zeros_like(b, dtype=float)
    x_new = np.zeros_like(b, dtype=float)
    A = np.array(A)
    it = 0
    start = time.time()
    for k in range(max_iter):
        it = k
        for i in range(n):
            x_new[i] = (b[i] - np.dot(A[i,:], x) + A[i,i]*x[i])/A[i,i]
        if np.linalg.norm(x_new - x) < tol:
            end = time.time()
            return x_new, it, (end - start)
        x = x_new
    raise Exception("Jacobi method did not converge")


def gauss_seidel(A, b, tol=1e-10, max_iter=10000000):
    n = len(A)
    x = np.zeros((n, 1), dtype=float)
    A = np.array(A)
    it = 0
    start = time.time()
    for k in range(max_iter):
        it = k
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

        if np.linalg.norm(A @ x - b) < tol:
            end = time.time()
            return x, it, (end - start)
    raise Exception("Gauss-Seidel method did not converge")

def calculate_residue_norm(A, x, b):
    return np.linalg.norm(np.dot(A, x) - b)

def task_a(n=955, a1=10):
    A = make_matrix(n=n, a1=a1)
    B = make_b(n=n)

    return A, B


def task_b():
    A, B = task_a()

    x_Jacobi, iter_jacobi, time_jacobi = jacobi(A, B)

    print(f"Jacobi Residue norm: {calculate_residue_norm(A, x_Jacobi, B)}")
    print(f"Jacobi iterations: {iter_jacobi}")
    print(f"Jacobi time: {time_jacobi}")

    x_gauss, iter_gauss, time_gauss = gauss_seidel(A, B)

    print(f"Gauss-Seidel Residue norm: {calculate_residue_norm(A, x_gauss, B)}")
    print(f"Gauss-Seidel iterations: {iter_gauss}")
    print(f"Gauss-Seidel time: {time_gauss}")

def task_c():
    A, B = task_a(3)

    x_Jacobi, iter_jacobi, time_jacobi = jacobi(A, B)

    print(f"Jacobi Residue norm: {calculate_residue_norm(A, x_Jacobi, B)}")
    print(f"Jacobi iterations: {iter_jacobi}")
    print(f"Jacobi time: {time_jacobi}")

    x_gauss, iter_gauss, time_gauss = gauss_seidel(A, B)

    print(f"Gauss-Seidel Residue norm: {calculate_residue_norm(A, x_gauss, B)}")
    print(f"Gauss-Seidel iterations: {iter_gauss}")
    print(f"Gauss-Seidel time: {time_gauss}")

def task_d():
    A = make_matrix()
    B = make_b()
    start = time.time()
    LU = LUDecompDoolittle(A)
    x_LU = SolveLinearSystem_LU(LU, B)
    end = time.time()
    elapsed_time = end - start
    print(f"LU Decomposition residue norm: {calculate_residue_norm(A, x_LU, B)}")
    return elapsed_time

def task_e():

    N = [100, 500, 1000, 2000, 3000]
    gauss_times = []
    jacobi_times = []
    LU_times = []
    for amount in N:
        A, B = task_a(n=amount)
        _, _, time_jacobi = jacobi(A, B)
        _, _, time_gauss = gauss_seidel(A, B)
        time_LU = task_d()
        gauss_times.append(time_gauss)
        jacobi_times.append(time_jacobi)
        LU_times.append(time_LU)

    fig, ax = plt.subplots()
    ax.plot(N, gauss_times, 'blue')
    ax.plot(N, jacobi_times, 'green')
    ax.plot(N, LU_times, 'yellow')


    ax.grid()

    plt.show()


if __name__ == '__main__':
    task_e()

