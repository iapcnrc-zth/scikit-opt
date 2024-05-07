#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987

import numpy as np
from sko.tools import func_transformer
from .base import SkoBase
from .operators import crossover, mutation, ranking, selection
from .operators import mutation

class PSO_TSP(SkoBase):
    def __init__(self, func, n_dim, size_pop=50, max_iter=200, w=0.8, c1=0.1, c2=0.1):
        self.field = func_transformer(func)
        self.field_raw = func
        self.n_dim = n_dim
        self.size_pop = size_pop
        self.max_iter = max_iter

        self.w = w
        self.cp = c1
        self.cg = c2

        self.X = self.crt_X()
        self.Y = self.cal_y()
        self.pbest_x = self.X.copy()
        self.pbest_y = np.array([[np.inf]] * self.size_pop)

        self.gbest_x = self.pbest_x[0, :]
        self.gbest_y = np.inf
        self.gbest_y_hist = []
        self.update_gbest()
        self.update_pbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.verbose = False

    def crt_X(self):
        tmp = np.random.rand(self.size_pop, self.n_dim)
        return tmp.argsort(axis=1)

    def pso_add(self, c, x1, x2):
        x1, x2 = x1.tolist(), x2.tolist()
        ind1, ind2 = np.random.randint(0, self.n_dim - 1, 2)
        if ind1 >= ind2:
            ind1, ind2 = ind2, ind1 + 1

        part1 = x2[ind1:ind2]
        part2 = [i for i in x1 if i not in part1]  # this is very slow

        return np.array(part1 + part2)

    def update_X(self):
        for i in range(self.size_pop):
            x = self.X[i, :]
            x = self.pso_add(self.cp, x, self.pbest_x[i])
            self.X[i, :] = x

        self.cal_y()
        self.update_pbest()
        self.update_gbest()

        for i in range(self.size_pop):
            x = self.X[i, :]
            x = self.pso_add(self.cg, x, self.gbest_x)
            self.X[i, :] = x

        self.cal_y()
        self.update_pbest()
        self.update_gbest()

        for i in range(self.size_pop):
            x = self.X[i, :]
            new_x_strategy = np.random.randint(3)
            if new_x_strategy == 0:
                x = mutation.swap(x)
            elif new_x_strategy == 1:
                x = mutation.reverse(x)
            elif new_x_strategy == 2:
                x = mutation.transpose(x)

            self.X[i, :] = x

        self.cal_y()
        self.update_pbest()
        self.update_gbest()

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.field(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            # self.update_V()
            self.recorder()
            self.update_X()
            # self.cal_y()
            # self.update_pbest()
            # self.update_gbest()

            if self.verbose:
                print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))

            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y
