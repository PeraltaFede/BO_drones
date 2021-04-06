import matplotlib.pyplot as plt
import numpy as np

length_scale = 1
# lam = np.array([0.67])

d = np.arange(-2, 2.01, 0.0001)
# y = (np.sqrt(np.pi) * length_scale) * np.exp(-(d ** 2) / (4 * length_scale ** 2))
# y = np.exp(-(d ** 2) / (2 * length_scale ** 2))
y = (2 * np.pi * length_scale ** 2) ** (1 / 2) * np.exp(-2 * (np.pi * length_scale) ** 2 * d ** 2)

# print(np.max(y)*0.5)

halfpower = np.max(y) / 2
for yid, yi in enumerate(y):
    if yi > halfpower:
        break
print('tmuestreo', -1 / (2 * np.pi * d[yid]) / 2)
print('tcorte', -1 / (2 * np.pi * d[yid]))

# print(-d[yid]/length_scale)
# area = np.trapz(y, dx=0.01)
# print("area =", area)

plt.plot(d, y, '-')
# plt.plot(d, y2)
plt.xlabel("$|| (x - x') ||$")
plt.xlabel("s(x)")
# plt.ylabel("$exp{\\frac{-|| (x - x')^2 ||}{2*1^2}}$")
plt.ylabel("S(s)")
plt.show(block=False)
plt.vlines([-(d[yid]) / 2 / length_scale, (d[yid]) / 2 / length_scale], 0, 2 * halfpower, color='red')
plt.vlines([-(d[yid]) / length_scale, (d[yid]) / length_scale], 0, 2 * halfpower, color='blue')
plt.hlines(halfpower, -2, 2, color='red')
plt.show(block=True)
