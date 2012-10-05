# -*- coding: utf-8 -*-

#! /usr/bin/env python
from random import gammavariate
from random import random

"""
Samples from a Dirichlet distribution with parameter @alpha using a Gamma distribution
Reference: 
http://en.wikipedia.org/wiki/Dirichlet_distribution
http://stackoverflow.com/questions/3028571/non-uniform-distributed-random-array
"""
def Dirichlet(alpha):
    sample = [gammavariate(a,1) for a in alpha]
    sample = [v/sum(sample) for v in sample]
    return sample
    
"""
Normalize a vector to be a probablistic representation
"""
def normalize(vec):
    s = sum(vec)
    assert(abs(s) > 1e-6) # the sum must not be 0
    for i in range(len(vec)):
        assert(vec[i] >= 0) # element must be >= 0
        vec[i] = vec[i] * 1.0 / s
        
"""
Choose a element in @vec according to a specified distribution @pr
Reference:
http://stackoverflow.com/questions/4437250/choose-list-variable-given-probability-of-each-variable
"""
def choose(vec, pr):
    assert(len(vec) == len(pr))
    # normalize the distributions
    normalize(pr)
    r = random()
    index = -1
    while (r > 0):
        r = r - pr[index]
        index = index + 1
    return vec[index]
    
if __name__ == "__main__":
    # This is a test
    print Dirichlet([1,1,1]);
