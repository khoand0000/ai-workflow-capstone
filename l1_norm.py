import numpy as np

def calculate_l1_norm(v):
    """
    Calculate the L1 norm (Manhattan norm) of a vector.
    
    Args:
        v (numpy.ndarray): Input vector
        
    Returns:
        float: L1 norm of the vector
    """
    return np.sum(np.abs(v))

# Example usage
if __name__ == "__main__":
    v = np.array([2.0, -3.5, 5.1])
    l1_norm = calculate_l1_norm(v)
    print(f"L1 norm of vector {v} is: {l1_norm}") 