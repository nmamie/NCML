import sys
import numpy as np
from tqdm import tqdm
from sko.tools import func_transformer

class APSO:

    def __init__(self, particles, velocities, fitness_function,
                 w=0.8, c_1=1, c_2=1, max_iter=100, auto_coef=True,
                 verbose=False, n_processes=0):
        self.particles = particles
        self.velocities = velocities
        # self.fitness_function = fitness_function
        self.func = func_transformer(fitness_function, n_processes)
        self.progress_bar = tqdm(total=max_iter)

        self.N = len(self.particles)
        self.D = len(self.particles[0])
        self.w = w
        self.c_1 = c_1
        self.c_2 = c_2
        self.auto_coef = auto_coef
        self.max_iter = max_iter


        self.p_bests = self.particles
        self.p_bests_values = self.cal_y()
        self.g_best = self.p_bests[0]
        self.g_best_value = self.p_bests_values[0]
        self.g_best_history = []
        self.update_bests()

        self.iter = 0
        self.is_running = True
        self.update_coef()

    def __str__(self):
        return f'[{self.iter}/{self.max_iter}] $w$:{self.w:.3f} - $c_1$:{self.c_1:.3f} - $c_2$:{self.c_2:.3f} - $f$:{-self.g_best_value*100:.3f}'

    def next(self):
        if self.iter > 0:
            self.move_particles()
            self.update_bests()
            self.update_coef()

        self.progress_bar.update(1)
        self.progress_bar.set_description(str(self))
        self.progress_bar.refresh()
        self.iter += 1
        self.is_running = self.is_running and self.iter < self.max_iter
        return self.is_running

    def update_coef(self):
        if self.auto_coef:
            t = self.iter
            n = self.max_iter
            self.w = (0.4/n**2) * (t - n) ** 2 + 0.4
            self.c_1 = -3 * t / n + 3.5
            self.c_2 =  3 * t / n + 0.5
            
    def cal_y(self):
        # calculate y for every x in X
        best_y = self.func(self.particles)
        return best_y

    def move_particles(self):

        # add inertia
        new_velocities = self.w * self.velocities
        # add cognitive component
        r_1 = np.random.random(self.N)
        r_1 = np.tile(r_1[:, None], (1, self.D))
        new_velocities += (self.c_1 * r_1) * (self.p_bests - self.particles)
        # add social component
        r_2 = np.random.random(self.N)
        r_2 = np.tile(r_2[:, None], (1, self.D))
        new_velocities += (self.c_2 * r_2) * (self.g_best - self.particles)

        self.is_running = np.sum(self.velocities - new_velocities) != 0

        # update positions and velocities
        self.velocities = new_velocities
        self.particles = self.particles + new_velocities


    def update_bests(self):
        fits = self.func(self.particles)

        for i in range(len(self.particles)):
            # update best personnal value (cognitive)
            if fits[i] < self.p_bests_values[i]:
                self.p_bests_values[i] = fits[i]
                self.p_bests[i] = self.particles[i]
                # update best global value (social)
                if fits[i] < self.g_best_value:
                    self.g_best_value = fits[i]
                    self.g_best_history.append(self.g_best_value)
                    self.g_best = self.particles[i]
