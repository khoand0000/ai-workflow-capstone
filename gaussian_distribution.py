import scipy.stats as stats

def gaussian_distribution(loc_val, scale_val, cdf_val):
    '''
    Calculate the probability of observing a score greater than or equal to cdf_val
    using the Normal distribution.
    
    INPUT: 
        loc_val (float): mean of the distribution
        scale_val (float): standard deviation of the distribution
        cdf_val (float): value to calculate probability for
        
    OUTPUT: 
        float: probability of observing a score >= cdf_val
    '''
    # Calculate P(X >= cdf_val) = 1 - P(X < cdf_val)
    return 1 - stats.norm.cdf(cdf_val, loc=loc_val, scale=scale_val)

# Example usage
if __name__ == "__main__":
    loc_val = 50.0    # Mean score
    scale_val = 20.0  # Standard deviation
    cdf_val = 80.0    # Score we want to find probability for
    
    probability = gaussian_distribution(loc_val, scale_val, cdf_val)
    print(f"The probability of observing a score >= {cdf_val} is: {probability:.6f}")
    print(f"This means there's a {probability*100:.4f}% chance of scoring {cdf_val} or higher")
    print(f"on the assessment.") 