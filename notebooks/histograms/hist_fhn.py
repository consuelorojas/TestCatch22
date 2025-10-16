import numpy as np
import matplotlib.pyplot as plt

x0 = [0,0]
samples = 100000
x = np.zeros((samples, len(x0)))
for i in range(samples):
    x1 = np.random.normal(0.0, 1.5)
    x2 = np.random.normal(0.0, 0.91)
    x[i] = [x1, x2]
counts, bins = np.histogram(x[:,0], bins=50)
#print(np.sum(counts))
plt.stairs(counts, bins, label='v0', fill=True)
plt.title('Histogram of initial condition v0')
plt.ylabel('count')
plt.xlabel('value')
plt.show()

counts, bins = np.histogram(x[:,1], bins=50)
#print(np.sum(counts))
plt.stairs(counts, bins, label='w0', fill=True)
plt.title('Histogram of initial condition w0')
plt.ylabel('count')
plt.xlabel('value')
plt.show()