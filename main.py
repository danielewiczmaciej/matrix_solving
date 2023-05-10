import numpy as np
import math
import time
import matplotlib.pyplot as plt


def make_matrix(n=955, w=5, a1=10, a2=-1, a3=-1):
    # Initialize matrix with zeros
    matrix = [[0.0 for j in range(n)] for i in range(n)]

    for i in range(n):
        matrix[i][i] = a1

    for i in range(n - w):
        matrix[i + 1][i] = a2
        matrix[i][i + 1] = a2

    for i in range(n - 2 * w):
        matrix[i + 2][i] = a3
        matrix[i][i + 2] = a3

    matrix = np.array(matrix)

    return matrix


def make_b(n=955):
    return np.array([[math.sin(i * 9)] for i in range(n)])


def matrix_mult(A, B):
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrices are not compatible for multiplication")
    result = np.zeros((A.shape[0], 1))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i][j] += A[i][k] * B[k][j]
    return result


def norm_residual(A, b, x):
    Ax = matrix_mult(A, x)
    res = np.empty(len(b))
    for i, (item1, item2) in enumerate(zip(b, Ax)):
        res[i] = item1 - item2
    norm_res = 0
    for i in range(len(res)):
        norm_res += (res[i] ** 2)
    return math.sqrt(norm_res)


def lu_factorization_solve(A, b):
    n = A.shape[0]
    U = np.copy(A)
    L = np.eye(n, dtype=np.double)
    for i in range(n):
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, np.newaxis] * U[i]
    # Solve Ly = b
    y = np.zeros(n)
    for i in range(n):
        s = 0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = b[i] - s
    # Solve Ux = y
    x = np.zeros((n, 1))
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s += U[i, j] * x[j]
        x[i] = (y[i] - s) / U[i, i]
    x = np.array(x)
    return x


def jacobi(A, b, tol=1e-9, max_iter=1000):
    n = len(A)
    x = np.zeros_like(b, dtype=float)
    x_old = np.copy(x)
    it = 0
    start = time.time()
    while norm_residual(A, b, x) >= tol:
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x_old[j]
            x[i] = (b[i] - sigma) / A[i][i]
        for i in range(n):
            x_old[i] = x[i]
        it += 1

    end = time.time()
    return x, it, (end - start)


def gauss_seidel(A, b, tol=1e-9, max_iter=1000):
    n = len(A)
    x = np.zeros((n, 1), dtype=float)
    start = time.time()
    it = 0
    while norm_residual(A, b, x) >= tol:
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]
        it += 1
    end = time.time()
    return x, it, (end - start)


def task_a(n=955, a1=10):
    A = make_matrix(n=n, a1=a1)
    B = make_b(n=n)

    return A, B


def task_b():
    A, B = task_a()

    x_Jacobi, iter_jacobi, time_jacobi = jacobi(A, B)

    print(f"Jacobi Residue norm: {norm_residual(A, B, x_Jacobi)}")
    print(f"Jacobi iterations: {iter_jacobi}")
    print(f"Jacobi time: {time_jacobi}")

    x_gauss, iter_gauss, time_gauss = gauss_seidel(A, B)

    print(f"Gauss-Seidel Residue norm: {norm_residual(A, B, x_gauss)}")
    print(f"Gauss-Seidel iterations: {iter_gauss}")
    print(f"Gauss-Seidel time: {time_gauss}")


def task_c():
    A, B = task_a(a1=3)

    x_Jacobi, iter_jacobi, time_jacobi = jacobi(A, B)

    print(f"Jacobi Residue norm: {norm_residual(A, B, x_Jacobi)}")
    print(f"Jacobi iterations: {iter_jacobi}")
    print(f"Jacobi time: {time_jacobi}")

    x_gauss, iter_gauss, time_gauss = gauss_seidel(A, B)

    print(f"Gauss-Seidel Residue norm: {norm_residual(A, B, x_gauss)}")
    print(f"Gauss-Seidel iterations: {iter_gauss}")
    print(f"Gauss-Seidel time: {time_gauss}")


def task_d(n=955, a1=10):
    A = make_matrix(n, a1=a1)
    B = make_b(n)
    start = time.time()
    x_LU = lu_factorization_solve(A, B)
    end = time.time()
    elapsed_time = end - start
    print(f"LU Factorization residual norm: {norm_residual(A, B, x_LU)}")
    return elapsed_time


def task_e():
    N = [100, 500, 1000, 2000, 3000]
    gauss_times = []
    jacobi_times = []
    LU_times = []
    for amount in N:
        A, B = task_a(n=amount)
        x_jacobi, it_jacobi, time_jacobi = jacobi(A, B)
        x_gauss, it_gauss, time_gauss = gauss_seidel(A, B)
        time_LU = task_d(n=amount)
        gauss_times.append(time_gauss)
        jacobi_times.append(time_jacobi)
        LU_times.append(time_LU)
        print(f"Jacobi iterations: {it_jacobi}")
        print(f"Gauss iterations: {it_gauss}")

    fig, ax = plt.subplots()
    ax.plot(N, gauss_times, 'blue')
    ax.plot(N, jacobi_times, 'green')
    ax.plot(N, LU_times, 'yellow')

    ax.grid()

    plt.show()


if __name__ == '__main__':
    task_d(a1=3)
