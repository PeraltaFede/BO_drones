import numpy as np
import matplotlib.pyplot as plt



u = 10
v = 0.1


x = (np.arange(0, 20, 0.1)-u)/v
y = np.exp(-(x ** 2)/2)/(v*np.sqrt(2*np.pi))

print(np.sum(y))

plt.plot(x, y)
plt.show()
