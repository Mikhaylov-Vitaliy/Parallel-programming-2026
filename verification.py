import numpy as np

def load_two_matrices(filename):
    with open(filename, 'r') as file:
        n = int(file.readline())
        
        matrix1 = []
        for _ in range(n):
            row = list(map(float, file.readline().split()))
            matrix1.append(row)
  
        matrix2 = []
        for _ in range(n):
            row = list(map(float, file.readline().split()))
            matrix2.append(row)
            
    return np.array(matrix1), np.array(matrix2)

print("------ВЕРИФИКАЦИЯ---")


try:
    A, B = load_two_matrices("lab_1/matrix_generated.txt")
 
    my_result = load_matrix("lab_1/result.txt")
    
    numpy_result = np.matmul(A, B)

    max_error = np.max(np.abs(numpy_result - my_result))
    print(f"Максимальная ошибка: {max_error:.10f}")
    
    if max_error < 1e-6:
        print("ВЕРИФИКАЦИЯ ПРОЙДЕНА!")
    else:
        print("ВЕРИФИКАЦИЯ НЕ ПРОЙДЕНА!")
        print(f"Ошибка слишком большая: {max_error:.10f}")

    print("\n--- Матрица A ---")
    print(A)
    print("\n--- Матрица B ---")
    print(B)
    print("\n--- Результат из NumPy ---")
    print(numpy_result)
    print("\n--- Результат (ваша программа) ---")
    print(my_result)
        
except FileNotFoundError as e:
    print(f" Ошибка: файл не найден!")
    print(f"  {e}")
  
except Exception as e:
    print(f" Ошибка: {e}")
