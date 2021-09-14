import matplotlib.pyplot as plt
import numpy as np

length_scale = 1
# lam = np.array([0.67])

# d = np.arange(-5, 5.01, 0.0001)

# d = np.arange(-2.5, 2.51, 0.0001)
# d_paint = np.arange(-1.96, 1.97, 0.0001)
# y = (np.sqrt(np.pi) * length_scale) * np.exp(-(d ** 2) / (2 * length_scale ** 2))
# y_paint = (np.sqrt(np.pi) * length_scale) * np.exp(-(d_paint ** 2) / (2 * length_scale ** 2))
# y = np.exp(-(d ** 2) / (2 * length_scale ** 2))
# y = (2 * np.pi * length_scale ** 2) ** (1 / 2) * np.exp(-2 * (np.pi * length_scale) ** 2 * d ** 2)

# print(np.max(y)*0.5)

# halfpower = np.max(y) / 2
# for yid, yi in enumerate(y):
#     if yi > halfpower:
#         break
# print('tmuestreo', -1 / (2 * np.pi * d[yid]) / 2)
# print('tcorte', -1 / (2 * np.pi * d[yid]))

# print(-d[yid]/length_scale)
# area = np.trapz(y, dx=0.01)
# print("area =", area)

# plt.plot(d, y, '-')
# plt.plot(d, y2)
# plt.xlabel("")
plt.xlabel("$x$")
# plt.ylabel("$exp{\\frac{-|| (x - x')^2 ||}{2*1^2}}$")
plt.ylabel("$\\mu(x)$")
plt.show(block=False)

plt.fill_between(np.arange(-2, 2.01, 0.01), -1.96, 1.96, alpha=0.4)
# plt.fill_between(d_paint, 0, y_paint, alpha=0.4)
plt.vlines(0, -2.1, 2.1, color='red', linestyle='dotted')
# plt.vlines(-1.96, 0, 2 * halfpower, color='blue')
# plt.vlines(0, -0.1, 2.5 * halfpower, color='black')
# plt.vlines([-(d[yid]) / 2 / length_scale, (d[yid]) / 2 / length_scale], 0, 2 * halfpower, color='red')
# plt.vlines([-(d[yid]) / length_scale, (d[yid]) / length_scale], 0, 2 * halfpower, color='blue')
plt.hlines(0, -2.5, 2.5, color='black')
plt.hlines(-1.96, -2, 2, color='blue')
plt.hlines(1.96, -2, 2, color='blue')
plt.show(block=True)
