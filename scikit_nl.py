import numpy as np
from sko.PSO import PSO
from sample_scalar_field_1 import plume


def demo_func(x):
    x1, x2 = x
    demo_env = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 10 + np.e
    if demo_env.size > 1: 
        print(demo_func)
        # for i in range(demo_env.size):print('concentration field: ', demo_env[i])
    return demo_env


constraint_ueq = (
    lambda x: (x[0] - 1) ** 2 + (x[1] - 0) ** 2 - 0.5 ** 2
    ,
)

sample_plume = plume()
max_iter = 50
# print(demo_func)
pso = PSO(field=sample_plume, n_dim=2, pop=41, max_iter=max_iter, lb=[0,0], ub=[119,119])
pso.record_mode = True
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

# %% Now Plot the animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

record_value = pso.record_value
X_list, V_list = record_value['X'], record_value['V']

fig, ax = plt.subplots(1, 1)
ax.set_title('title', loc='center')
line = ax.plot([], [], 'b.')

X_grid, Y_grid = np.meshgrid(np.linspace(0, 120, 120), np.linspace(0, 120, 120))
# Z_grid = demo_func((X_grid, Y_grid))
# ax.contour(X_grid, Y_grid, Z_grid, 30)

ax.contour(X_grid, Y_grid, sample_plume, 100)

ax.set_xlim(0, 120)
ax.set_ylim(0, 120)

# t = np.linspace(0, 2 * np.pi, 40)
# ax.plot(0.5 * np.cos(t) + 1, 0.5 * np.sin(t), color='r')

plt.ion()
p = plt.show()


def update_scatter(frame):
    i, j = frame // 10, frame % 10
    ax.set_title('iter = ' + str(i))
    X_tmp = X_list[i] + V_list[i] * j / 10.0
    plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1])
    return line


ani = FuncAnimation(fig, update_scatter, blit=True, interval=25, frames=max_iter * 10)
ani.save('pso.gif', writer='pillow')
plt.show()