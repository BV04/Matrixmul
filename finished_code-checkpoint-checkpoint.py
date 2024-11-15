import time
import numpy as np
import matplotlib.pyplot as plt

#Matrix multiplication
def matrix_multiplication(matrix_1, matrix_2):
    n = len(matrix_1)
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0)
        result.append(row)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += matrix_1[i][k] * matrix_2[k][j]
    return result

# Zeitmessung für die naive Matrixmultiplikation
def measure_multiplication(matrix_size):
    matrix_1 = []
    matrix_2 = []
    for i in range(matrix_size):
        row_1 = []
        row_2 = []
        for j in range(matrix_size):
            row_1.append(np.random.randint(1, 11))
            row_2.append(np.random.randint(1, 11))
        matrix_1.append(row_1)
        matrix_2.append(row_2)
    
    start_time = time.time()
    matrix_multiplication(matrix_1, matrix_2)
    end_time = time.time()
    
    return end_time - start_time

# Laufzeitmessungen für Matrixgrößen von 2x2 bis 1000x1000 in 2er-Schritten
naive_times = []
matrix_sizes = []
for size in range(2, 1002, 2):
    matrix_sizes.append(size)
    naive_times.append(measure_multiplication(size))

# Zeitmessung für die `NumPy`-Matrixmultiplikation
def measure_numpy_time(matrix_size):
    matrix_1 = np.random.randint(1, 11, size=(matrix_size, matrix_size))
    matrix_2 = np.random.randint(1, 11, size=(matrix_size, matrix_size))
    
    start_time = time.time()
    np.dot(matrix_1, matrix_2)
    end_time = time.time()
    
    return end_time - start_time

# Laufzeitmessungen für `NumPy`
numpy_times = []
for size in matrix_sizes:
    numpy_times.append(measure_numpy_time(size))

# Plotten der Ergebnisse
plt.plot(matrix_sizes, naive_times, label='Naive Multiplikation')
plt.plot(matrix_sizes, numpy_times, label='NumPy Multiplikation')
plt.xlabel('Matrixgröße (n x n)')
plt.ylabel('Zeit (Sekunden)')
plt.legend()
plt.title('Vergleich der Laufzeiten von Naiver und NumPy Matrixmultiplikation')
plt.show()

# Addition und Subtraktion für Matrizen
def matrix_addition(matrix_1, matrix_2):
    n = len(matrix_1)
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(matrix_1[i][j] + matrix_2[i][j])
        result.append(row)
    return result

def matrix_subtraction(matrix_1, matrix_2):
    n = len(matrix_1)
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(matrix_1[i][j] - matrix_2[i][j])
        result.append(row)
    return result

# Strassen-Algorithmus
def strassen(matrix_1, matrix_2):
    if len(matrix_1) == 1:
        return [[matrix_1[0][0] * matrix_2[0][0]]]

    mid = len(matrix_1) // 2
    A11, A12, A21, A22 = [], [], [], []
    B11, B12, B21, B22 = [], [], [], []
    
    for i in range(mid):
        A11.append(matrix_1[i][:mid])
        A12.append(matrix_1[i][mid:])
        A21.append(matrix_1[i + mid][:mid])
        A22.append(matrix_1[i + mid][mid:])
        B11.append(matrix_2[i][:mid])
        B12.append(matrix_2[i][mid:])
        B21.append(matrix_2[i + mid][:mid])
        B22.append(matrix_2[i + mid][mid:])
    
    P1 = strassen(matrix_addition(A11, A22), matrix_addition(B11, B22))
    P2 = strassen(matrix_addition(A21, A22), B11)
    P3 = strassen(A11, matrix_subtraction(B12, B22))
    P4 = strassen(A22, matrix_subtraction(B21, B11))
    P5 = strassen(matrix_addition(A11, A12), B22)
    P6 = strassen(matrix_subtraction(A21, A11), matrix_addition(B11, B12))
    P7 = strassen(matrix_subtraction(A12, A22), matrix_addition(B21, B22))

    C11, C12, C21, C22 = [], [], [], []
    for i in range(mid):
        row1, row2, row3, row4 = [], [], [], []
        for j in range(mid):
            row1.append(P1[i][j] + P4[i][j] - P5[i][j] + P7[i][j])
            row2.append(P3[i][j] + P5[i][j])
            row3.append(P2[i][j] + P4[i][j])
            row4.append(P1[i][j] - P2[i][j] + P3[i][j] + P6[i][j])
        C11.append(row1)
        C12.append(row2)
        C21.append(row3)
        C22.append(row4)

    top, bottom = [], []
    for i in range(mid):
        top.append(C11[i] + C12[i])
        bottom.append(C21[i] + C22[i])
    
    return top + bottom

# Test der Strassen-Implementierung und Zeitmessung
def test_strassen(matrix_size):
    matrix_1 = []
    matrix_2 = []
    for i in range(matrix_size):
        row_1 = []
        row_2 = []
        for j in range(matrix_size):
            row_1.append(np.random.randint(1, 11))
            row_2.append(np.random.randint(1, 11))
        matrix_1.append(row_1)
        matrix_2.append(row_2)
    
    start_time = time.time()
    strassen(matrix_1, matrix_2)
    end_time = time.time()
    
    return end_time - start_time

# Vergleich der Strassen-Laufzeit mit Naiv und NumPy
strassen_times = []
strassen_sizes = []
for i in range(1, 10):
    size = 2 ** i
    strassen_sizes.append(size)
    strassen_times.append(test_strassen(size))

# Plot der Laufzeiten
plt.plot(matrix_sizes, naive_times, label='Naive Multiplikation')
plt.plot(matrix_sizes, numpy_times, label='NumPy Multiplikation')
plt.plot(strassen_sizes, strassen_times, label='Strassen Multiplikation')
plt.xlabel('Matrixgröße (n x n)')
plt.ylabel('Zeit (Sekunden)')
plt.legend()
plt.title('Vergleich der Laufzeiten von Naiver, NumPy und Strassen Matrixmultiplikation')
plt.show()
