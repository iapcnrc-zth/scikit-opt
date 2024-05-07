import numpy as np
from sko.PSO import PSO
from sample_scalar_field_1 import plume


sample_plume = plume()
max_iteration = 60
agent_num = 12
speed_lim = .01
env_dim = 2
s_lim_vector = []
for i in range(env_dim): s_lim_vector.append(speed_lim)

constraint_ueq = (
    lambda x: (x[0] - 1) ** 2 + (x[1] - 0) ** 2 - 0.5 ** 2
    ,
)

def demo_func(x):
    x1, x2 = x
    demo_env = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 10 + np.e
    if demo_env.size > 1: 
        print(demo_func)
        # for i in range(demo_env.size):print('concentration field: ', demo_env[i])
    return demo_env

def line_point_gen(first_point, dir, ax, point_num, interval):
    re = []
    if ax == 'y':
        for i in range(int(point_num)):re.append(np.array([int(first_point[0]) , int(first_point[1]) + (dir * int(interval) * i)]))
    else:
        for i in range(int(point_num)):re.append(np.array([int(first_point[0]) + (dir * int(interval) * i), int(first_point[1]) ]))
    return re

def rectangle_point_gen(field_dimension, point_num):
    init_list = []
    init_list.append(line_point_gen([5, field_dimension[1] / (point_num/4)], 1, 'y', point_num/4, field_dimension[1] / (point_num/4)))
    init_list.append(line_point_gen([field_dimension[0] / (point_num/4), field_dimension[1] - 5], 1, 'x', point_num/4, field_dimension[0] / (point_num/4)))
    init_list.append(line_point_gen([field_dimension[0] - 5, field_dimension[1] / (point_num/4)], 1, 'y', point_num/4, field_dimension[1] / (point_num/4)))
    init_list.append(line_point_gen([field_dimension[0] / (point_num/4), 5], 1, 'x', point_num/4, field_dimension[0] / (point_num/4)))
    re = []
    for i in range(len(init_list)): 
        for j in range(len(init_list[i])): re.append(init_list[i][j])
    return re

# initial_pos = line_point_gen([5, 10], 1, 'y', agent_num, 10)
initial_pos = rectangle_point_gen([119, 119], agent_num)

# performance test section
# success_num = 0
# for n in range(1000):
#     pso = PSO(sample_plume, initial_pos, s_lim_vector, env_dim, pop=agent_num, max_iter=max_iteration, lb=[0,0], ub=[119,119])
#     pso.record_mode = True
#     pso.run()
#     # print('s lim: ', speed_lim)
#     # print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
#     if np.linalg.norm(np.array([60,60]) - pso.gbest_x) < 10: success_num += 1
# print('success rate: ', success_num/n)
# print('agent number: ', agent_num)
# print('speed: ', speed_lim*120, 'm/s')

# %% Now Plot the animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

pso = PSO(sample_plume, initial_pos, s_lim_vector, env_dim, pop=agent_num, max_iter=max_iteration, lb=[0,0], ub=[120,120])
pso.record_mode = True
pso.run()
print('s lim: ', s_lim_vector)
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

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
    '''position update function for each animation frame'''
    i, j = frame // 10, frame % 10
    ax.set_title('iter = ' + str(i))
    X_tmp = X_list[i] + V_list[i] * j / 10.0
    plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1])
    return line


ani = FuncAnimation(fig, update_scatter, blit=True, interval=25, frames=max_iteration * 10)
ani.save('pso_surround_init.gif', writer='pillow')
plt.show()