import scipy.stats as stats

def geometric_distribution(p, k):
    '''
    Calculate the probability of having k-1 failures before the first success
    using the geometric distribution.
    
    INPUT: 
        p (float): probability of success
        k (int): number of trials until first success
        
    OUTPUT: 
        float: probability of having k-1 failures before first success
    '''
    # The geometric distribution in scipy.stats uses k-1 failures
    # So we use k-1 for the calculation
    return stats.geom.pmf(k, p)

# Example usage
if __name__ == "__main__":
    p = 1/10  # Probability of success (1 out of 10 people buy)
    k = 20    # Number of trials until first success
    
    probability = geometric_distribution(p, k)
    print(f"The probability of waiting until the {k}th person is: {probability:.6f}")
    print(f"This means there's a {probability*100:.4f}% chance the vendor will have to wait")
    print(f"until {k} people walk by before someone buys a taco.") 