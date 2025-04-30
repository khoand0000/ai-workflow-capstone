import scipy.stats as stats

def poisson_distribution(mu, k):
    '''
    Calculate the probability of more than k accidents using the Poisson distribution.
    
    INPUT: 
        mu (float): expected number of events (mean of the distribution)
        k (int): number of events to calculate probability for
        
    OUTPUT: 
        float: probability of more than k events occurring
    '''
    # Calculate P(X > k) = 1 - P(X ≤ k)
    return 1 - stats.poisson.cdf(k, mu)

# Example usage
if __name__ == "__main__":
    mu = 4  # Expected number of accidents per month
    k = 7   # We want to know probability of more than 7 accidents
    
    probability = poisson_distribution(mu, k)
    print(f"The probability of more than {k} accidents next month is: {probability:.6f}")
    print(f"This means there's a {probability*100:.4f}% chance of having more than {k} accidents")
    print(f"at the intersection next month.") 