#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/9/17
# @Author  : github.com/guofei9987


import numpy as np
from tqdm import tqdm
from sko.tools import func_transformer
from scipy import spatial


# class ASFA_raw:
#     # 这里统一做成极小值求解的问题
#     # 参考某 Matlab 教课书上的代码，很多地方的设定无力吐槽，基本没有泛用性
#     def __init__(self, func):
#         self.func = func
#         self.size_pop = 50
#         self.max_iter = 300
#         self.n_dim = 2
#         self.max_try_num = 100  # 最大尝试次数
#         self.step = 0.1  # 每一步的最大位移
#         self.visual = 0.5  # 鱼的最大感知范围
#         self.delta = 1.3  # 拥挤度阈值
#
#         self.X = np.random.rand(self.size_pop, self.n_dim)
#         self.Y = np.array([self.func(x) for x in self.X])
#
#         best_idx = self.Y.argmin()
#         self.best_Y = self.Y[best_idx]
#         self.best_X = self.X[best_idx, :]
#
#     def move_to_target(self, index_individual, x_target):
#         # 向目标移动，prey(), swarm(), follow() 三个算子中的移动都用这个
#         x = self.X[index_individual, :]
#         x_new = x + self.step * np.random.rand() * (x_target - x) / np.linalg.norm(x_target - x)
#         self.X[index_individual, :] = x_new
#         self.Y[index_individual] = self.func(x_new)
#         if self.Y[index_individual] < self.best_Y:
#             self.best_X = self.X[index_individual, :]
#
#     def move(self, index_individual):
#         r = 2 * np.random.rand(self.n_dim) - 1
#         x_new = self.X[index_individual, :] + self.visual * r
#         self.X[index_individual, :] = x_new
#         self.Y[index_individual] = self.func(x_new)
#         if self.Y[index_individual] < self.best_Y:
#             self.best_X = self.X[index_individual, :]
#
#     def prey(self, index_individual):
#         for try_num in range(self.max_try_num):
#             r = 2 * np.random.rand(self.n_dim) - 1
#             x_target = self.X[index_individual, :] + self.visual * r
#             if self.func(x_target) < self.Y[index_individual]:  # 捕食成功
#                 print('prey',index_individual)
#                 self.move_to_target(index_individual, x_target)
#                 return None
#         # 捕食 max_try_num 次后仍不成功，就调用 move 算子
#         self.move(index_individual)
#
#     def find_individual_in_vision(self, index_individual):
#         # 找出 index_individual 这个个体视线范围内的所有鱼
#         distances = spatial.distance.cdist(self.X[[index_individual], :], self.X, metric='euclidean').reshape(-1)
#         index_individual_in_vision = np.argwhere((distances > 0) & (distances < self.visual))[:, 0]
#         return index_individual_in_vision
#
#     def swarm(self, index_individual):
#         index_individual_in_vision = self.find_individual_in_vision(index_individual)
#         num_index_individual_in_vision = len(index_individual_in_vision)
#         if num_index_individual_in_vision > 0:
#             individual_in_vision = self.X[index_individual_in_vision, :]
#             center_individual_in_vision = individual_in_vision.mean(axis=0)
#             center_y_in_vision = self.func(center_individual_in_vision)
#             if center_y_in_vision * num_index_individual_in_vision < self.delta * self.Y[index_individual]:
#                 self.move_to_target(index_individual, center_individual_in_vision)
#                 return None
#         self.prey(index_individual)
#
#     def follow(self, index_individual):
#         index_individual_in_vision = self.find_individual_in_vision(index_individual)
#         num_index_individual_in_vision = len(index_individual_in_vision)
#         if num_index_individual_in_vision > 0:
#             individual_in_vision = self.X[index_individual_in_vision, :]
#             y_in_vision = np.array([self.func(x) for x in individual_in_vision])
#             index_target = y_in_vision.argmax()
#             x_target = individual_in_vision[index_target]
#             y_target = y_in_vision[index_target]
#             if y_target * num_index_individual_in_vision < self.delta * self.Y[index_individual]:
#                 self.move_to_target(index_individual, x_target)
#                 return None
#         self.prey(index_individual)
#
#     def fit(self):
#         for epoch in range(self.max_iter):
#             for index_individual in range(self.size_pop):
#                 self.swarm(index_individual)
#                 self.follow(index_individual)
#         return self.best_X, self.best_Y

# %%
class AFSA:
    def __init__(self, func, n_dim, size_pop=50, max_iter=300,
                 max_try_num=100, lb=-1e5, ub=1e5, step=0.5, visual=0.3,
                 q=0.98, delta=0.5, verbose=False, n_processes=0):
        self.func = func
        self.trans_func = func_transformer(func, n_processes)
        self.progress_bar = tqdm(total=max_iter)
        self.verbose=verbose
        
        self.n_dim = n_dim
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.max_try_num = max_try_num  # 最大尝试捕食次数
        self.step = step  # 每一步的最大位移比例
        self.visual = visual  # 鱼的最大感知范围
        self.q = q  # 鱼的感知范围衰减系数
        self.delta = delta  # 拥挤度阈值，越大越容易聚群和追尾
        
        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))
        self.Y = self.cal_y()

        best_idx = self.Y.argmin()
        self.best_x, self.best_y = self.X[best_idx, :], self.Y[best_idx]
        self.best_X, self.best_Y = self.X.copy(), np.array([[np.inf]] * self.size_pop)
        self.best_y_hist = []
        self.best_y_avg_hist = []
        self.best_y_std_hist = []

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.trans_func(self.X).reshape(-1, 1)
        return self.Y
    
    def move_to_target(self, idx_individual, x_target):
        '''
        move to target
        called by prey(), swarm(), follow()

        :param idx_individual:
        :param x_target:
        :return:
        '''
        x = self.X[idx_individual, :]
        x_new = x + self.step * np.random.rand() * (x_target - x)
        # x_new = x_target
        self.X[idx_individual, :] = x_new
        # self.Y = self.cal_y()
        # if self.Y[idx_individual] < self.best_Y:
        #     self.best_x = self.X[idx_individual, :].copy()
        #     self.best_y = self.Y[idx_individual].copy()

    def move(self, idx_individual):
        '''
        randomly move to a point

        :param idx_individual:
        :return:
        '''
        r = 2 * np.random.rand(self.n_dim) - 1
        x_new = self.X[idx_individual, :] + self.visual * r
        self.X[idx_individual, :] = x_new
        # self.Y = self.cal_y()
        # if self.Y[idx_individual] < self.best_Y:
        #     self.best_X = self.X[idx_individual, :].copy()
        #     self.best_Y = self.Y[idx_individual].copy()

    def prey(self, idx_individual):
        '''
        prey
        :param idx_individual:
        :return:
        '''
        for try_num in range(self.max_try_num):
            r = 2 * np.random.rand(self.n_dim) - 1
            x_target = self.X[idx_individual, :] + self.visual * r
            if self.func(x_target) < self.Y[idx_individual]:  # 捕食成功
                self.move_to_target(idx_individual, x_target)
                return None
        # 捕食 max_try_num 次后仍不成功，就调用 move 算子
        self.move(idx_individual)

    def find_individual_in_vision(self, idx_individual):
        # 找出 idx_individual 这条鱼视线范围内的所有鱼
        distances = spatial.distance.cdist(self.X[[idx_individual], :], self.X, metric='euclidean').reshape(-1)
        idx_individual_in_vision = np.argwhere((distances > 0) & (distances < self.visual))[:, 0]
        return idx_individual_in_vision

    def swarm(self, idx_individual):
        # 聚群行为
        idx_individual_in_vision = self.find_individual_in_vision(idx_individual)
        num_idx_individual_in_vision = len(idx_individual_in_vision)
        if num_idx_individual_in_vision > 0:
            individual_in_vision = self.X[idx_individual_in_vision, :]
            center_individual_in_vision = individual_in_vision.mean(axis=0)
            center_y_in_vision = self.func(center_individual_in_vision)
            if center_y_in_vision * num_idx_individual_in_vision < self.delta * self.Y[idx_individual]:
                self.move_to_target(idx_individual, center_individual_in_vision)
                return None
        self.prey(idx_individual)

    def follow(self, idx_individual):
        # 追尾行为
        idx_individual_in_vision = self.find_individual_in_vision(idx_individual)
        num_idx_individual_in_vision = len(idx_individual_in_vision)
        if num_idx_individual_in_vision > 0:
            individual_in_vision = self.X[idx_individual_in_vision, :]
            y_in_vision = np.array([self.func(x) for x in individual_in_vision])
            idx_target = y_in_vision.argmin()
            x_target = individual_in_vision[idx_target]
            y_target = y_in_vision[idx_target]
            if y_target * num_idx_individual_in_vision < self.delta * self.Y[idx_individual]:
                self.move_to_target(idx_individual, x_target)
                return None
        self.prey(idx_individual)

    def run(self, max_iter=None):
        print("Fish are swarming...")
        self.max_iter = max_iter or self.max_iter
        # progress bar
        for epoch in range(self.max_iter):
            for idx_individual in range(self.size_pop):
                self.swarm(idx_individual)
                self.X = np.clip(self.X, self.lb, self.ub)
                self.Y = self.cal_y()
                self.follow(idx_individual)
                self.X = np.clip(self.X, self.lb, self.ub)
                self.Y = self.cal_y()
                # find best individual
                best_idx = self.Y.argmin()
                if self.Y[best_idx] < self.best_y:
                    self.best_x = self.X[best_idx, :].copy()
                    self.best_y = self.Y[best_idx].copy()
                # for every individual find best position and fitness
                if self.Y[idx_individual] < self.best_Y[idx_individual]:
                    self.best_X[idx_individual, :] = self.X[idx_individual, :].copy()
                    self.best_Y[idx_individual] = self.Y[idx_individual].copy()
            
            self.visual *= self.q
            self.best_x = np.clip(self.best_x, self.lb, self.ub)
            self.best_y_hist.append(self.best_y)
            self.best_y_avg_hist.append(np.average(self.best_Y))
            self.best_y_std_hist.append(np.std(self.best_Y))
            # progress_bar update
            self.progress_bar.update(1)
            self.progress_bar.set_description(f"Epoch: {epoch+1} | Best fit: {self.best_y} at {self.best_x} | Average fit: {np.average(self.best_Y)}")
            self.progress_bar.refresh()

        return self.best_x, self.best_y
