import numpy as np

NUM_FEATURES = 5
NUM_SAMPLES = 1000

data = np.zeros((NUM_SAMPLES, NUM_FEATURES))

data[:,0] = np.random.random_integers(0,3,NUM_SAMPLES)
data[:,1] = np.random.random_integers(0,4,NUM_SAMPLES)
data[:,2] = np.random.random_integers(7,21,NUM_SAMPLES)
print(data)