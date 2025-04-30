import numpy as np

def create_and_sum_matrix(vectorLower, vectorUpper):
    """
    Create a vector from vectorLower to vectorUpper-1, reshape it into a matrix,
    and return the row sums.
    
    Args:
        vectorLower (int): Starting value of the vector
        vectorUpper (int): Ending value (exclusive) of the vector
        
    Returns:
        list: Sums of each row in the matrix
    """
    # Create the vector
    vector = np.arange(vectorLower, vectorUpper)
    
    # Reshape into 10x15 matrix
    matrix = vector.reshape(10, 15)
    
    # Calculate row sums
    row_sums = np.sum(matrix, axis=1)
    
    return row_sums.tolist()

# Example usage
if __name__ == "__main__":
    vectorLower = 1
    vectorUpper = 151
    
    row_sums = create_and_sum_matrix(vectorLower, vectorUpper)
    print("Row sums:")
    for i, sum_val in enumerate(row_sums, 1):
        print(f"Row {i}: {sum_val}") 