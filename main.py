import numpy as np
import math
import time
import matplotlib.pyplot as plt


def make_matrix(n=955, w=5, a1=10, a2=-1, a3=-1):
    # Initialize matrix with zeros
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]

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
    result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            result[i][j] = np.sum(np.multiply(A[i, :], B[:, j]))
    return result


def norm_residual(A, b, x):
    Ax = matrix_mult(A, x)
    res = b - Ax
    norm_res = 0
    for i in range(len(res)):
        norm_res += (res[i] ** 2)
    return math.sqrt(norm_res)


def lu_factorization_solve(A, b):
    n = A.shape[0]
    U = np.copy(A)
    L = np.eye(n, dtype=np.double)
    for i in range(n):
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])

            U[i][k] = A[i][k] - sum

        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])
                L[k][i] = (A[k][i] - sum) / U[i][i]
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


def jacobi(A, b, tol=1e-9, max_iter=30):
    n = len(A)
    x = np.zeros_like(b, dtype=float)
    x_old = np.copy(x)
    it = 0
    start = time.time()
    norm = norm_residual(A, b, x)
    norms = []
    while norm >= tol:
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x_old[j]
            x[i] = (b[i] - sigma) / A[i][i]
        for i in range(n):
            x_old[i] = x[i]
        if it > max_iter:
            break
        norm = norm_residual(A, b, x)
        norms.append(norm)
        it += 1

    end = time.time()
    return x, it, (end - start), norms


def gauss_seidel(A, b, tol=1e-9, max_iter=30):
    n = len(A)
    x = np.zeros((n, 1), dtype=float)
    start = time.time()
    it = 0
    norm = norm_residual(A, b, x)
    norms = []
    while norm >= tol:
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]
        if it > max_iter:
            break
        norm = norm_residual(A, b, x)
        norms.append(norm)
        it += 1
    end = time.time()
    return x, it, (end - start), norms


def task_a(n=955, a1=10):
    A = make_matrix(n=n, a1=a1)
    B = make_b(n=n)

    return A, B


def task_b():
    A, B = task_a()

    x_Jacobi, iter_jacobi, time_jacobi, norms_jacobi = jacobi(A, B)

    print(f"Jacobi Residue norm: {norm_residual(A, B, x_Jacobi)}")
    print(f"Jacobi iterations: {iter_jacobi}")
    print(f"Jacobi time: {time_jacobi}")

    x_gauss, iter_gauss, time_gauss, norms_gauss = gauss_seidel(A, B)

    print(f"Gauss-Seidel Residue norm: {norm_residual(A, B, x_gauss)}")
    print(f"Gauss-Seidel iterations: {iter_gauss}")
    print(f"Gauss-Seidel time: {time_gauss}")

    fig, ax = plt.subplots()
    # ax.plot(range(iter_gauss), norms_gauss, 'blue', label="Metoda Gauss'a-Seidel'a")
    ax.plot(range(iter_jacobi), norms_jacobi, 'green', label="Metoda Jacobi")

    ax.set_title("Norma residuum w kolejnych iteracjach")
    ax.set_xlabel("Iteracja [n]")
    ax.set_ylabel("Norma residuum")

    ax.legend()
    ax.grid()
    plt.yscale("log")
    plt.show()


def task_c():
    A, B = task_a(a1=3)

    x_Jacobi, iter_jacobi, time_jacobi, norms_jacobi = jacobi(A, B)

    print(f"Jacobi Residue norm: {norm_residual(A, B, x_Jacobi)}")
    print(f"Jacobi iterations: {iter_jacobi}")
    print(f"Jacobi time: {time_jacobi}")

    x_gauss, iter_gauss, time_gauss, norms_gauss = gauss_seidel(A, B)

    print(f"Gauss-Seidel Residue norm: {norm_residual(A, B, x_gauss)}")
    print(f"Gauss-Seidel iterations: {iter_gauss}")
    print(f"Gauss-Seidel time: {time_gauss}")

    fig, ax = plt.subplots()
    ax.plot(range(iter_gauss), norms_gauss, 'blue', label="Metoda Gauss'a-Seidel'a")
    ax.plot(range(iter_jacobi), norms_jacobi, 'green', label="Metoda Jacobi")

    ax.set_title("Norma residuum w kolejnych iteracjach")
    ax.set_xlabel("Iteracja [n]")
    ax.set_ylabel("Norma residuum")

    ax.legend()
    ax.grid()
    plt.yscale("log")
    plt.show()


def task_d(n=955, a1=10):
    start = time.time()
    A = make_matrix(n, a1=a1)
    B = make_b(n)
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
        print(f"LU Time: {time_LU}")
        print(f"Jacobi iterations: {it_jacobi}")
        print(f"Jacobi time: {time_jacobi}")
        print(f"Gauss iterations: {it_gauss}")
        print(f"Gauss time: {time_gauss}")

    fig, ax = plt.subplots()
    ax.plot(N, gauss_times, 'blue', label="Metoda Gauss'a-Seidel'a")
    ax.plot(N, jacobi_times, 'green', label="Metoda Jacobi")
    ax.plot(N, LU_times, 'yellow', label="Faktoryzacja LU")

    ax.set_title("Porównanie czasu rozwiązania")
    ax.set_xlabel("Rozmiar macierzy [N]")
    ax.set_ylabel("Czas [s]")

    ax.legend()
    ax.grid()

    plt.show()


if __name__ == '__main__':
    task_c()
