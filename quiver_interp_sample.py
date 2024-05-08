import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

x = [0, 0, 1, 1, 2, 2, 0, 1, 2]
y = [1, 2, 1, 2, 1, 2, 1.5, 1.5, 1.5]
u = np.array([0.5, -1, 0, 0, 0.25, 1, 0, 0, 0.75])
v = [1, 1, 1, 1, 1, 1, 1, 1, 1]

plt.figure(1)
plt.quiver(x, y, u, v)

xx = np.linspace(0, 2, 10)
yy = np.linspace(1, 2, 10)
xx, yy = np.meshgrid(xx, yy)

points = np.transpose(np.vstack((x, y)))
print(u.shape)
print(xx.shape)
print(yy.shape)
print(points.shape)
u_interp = interpolate.griddata(points, u, (xx, yy), method='cubic')
v_interp = interpolate.griddata(points, v, (xx, yy), method='cubic')

plt.figure(2)
plt.quiver(xx, yy, u_interp, v_interp)
# plt.show()