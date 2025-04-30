def matrix_multiplication(A, B):
    '''
    Perform matrix multiplication on two square matrices.
    
    INPUT: 
        A: List of lists of integers (n x n matrix)
        B: List of lists of integers (n x n matrix)
        
    OUTPUT: 
        List of lists of integers (n x n matrix) - product of A and B
    '''
    # Get the size of the matrices
    n = len(A)
    
    # Initialize the result matrix with zeros
    result = [[0 for _ in range(n)] for _ in range(n)]
    
    # Perform matrix multiplication
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

# Example usage
if __name__ == "__main__":
    # Example 2x2 matrices
    A = [[1, 2],
         [3, 4]]
    
    B = [[5, 6],
         [7, 8]]
    
    # Perform multiplication
    result = matrix_multiplication(A, B)
    
    # Print the result
    print("Matrix A:")
    for row in A:
        print(row)
    
    print("\nMatrix B:")
    for row in B:
        print(row)
    
    print("\nResult of A × B:")
    for row in result:
        print(row) 