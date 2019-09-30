import numpy as np

"""
a = 1+2j
b = 1-2j

print(a*b)
print(np.sqrt(-1+0j))
biases = [(np.random.randn(y, 1) + 1j*np.random.randn(y, 1)) for y in [1,3]]
print(biases)
"""

biases = [np.random.randn(y,) for y in [1,3]]
print(biases)
