import numpy as np
def beta2rms(beta2a, beta2b):
    return -np.sqrt(beta2a**2 + beta2b**2)
  
def beta2rms_complementary(beta2rms, beta2a):
    tmp = 2*beta2rms**2-beta2a**2
    assert(tmp>0)
    return -np.sqrt(tmp)
    
def beta2avg(beta2a, beta2b):
    return (beta2a + beta2b) / 2
  
def beta2avg_complementary(beta2avg, beta2a):
    return beta2avg-beta2a