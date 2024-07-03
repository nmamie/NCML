import numpy as np
from tqdm import tqdm
from sko.tools import func_transformer

class BinaryPSO:
    def __init__(self, fitness_func, dim, n_particles, n_iterations, inertia_weight=0.5, cognitive=1.5, social=1.5, n_processes=1):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.dim = dim
        self.fitness_func = func_transformer(fitness_func, n_processes)
        self.inertia_weight = inertia_weight
        self.cognitive = cognitive
        self.social = social
        
        # Initialize particles' positions and velocities
        self.particles = np.random.randint(2, size=(n_particles, dim))  # Initialize with 0 or 1
        self.velocities = np.random.uniform(-1, 1, size=(n_particles, dim))  # Initialize with random values between -1 and 1
        
        # Initialize personal best positions and global best position
        self.pbest_positions = self.particles.copy()
        self.pbest_scores = np.full(n_particles, np.inf)
        self.gbest_position = np.zeros(dim)
        self.gbest_score = np.inf
        self.gbest_y_hist = []
        self.p_best_y_hist = []
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def cal_fitness(self, x):
        self.Y = self.fitness_func(x)
        return self.Y
    
    def update_velocity(self):
        inertia = self.inertia_weight * self.velocities
        cognitive_component = self.cognitive * np.random.rand(self.n_particles, self.dim) * (self.pbest_positions - self.particles)
        social_component = self.social * np.random.rand(self.n_particles, self.dim) * (self.gbest_position - self.particles)
        
        self.velocities = inertia + cognitive_component + social_component
        
        # Velocity clamping to ensure velocities are within a specific range for binary PSO
        vmax = np.abs(self.sigmoid(-self.social) - self.sigmoid(self.social))
        self.velocities = np.clip(self.velocities, -vmax, vmax)
    
    def update_position(self):
        sigmoid_v = self.sigmoid(self.velocities)
        self.particles = np.random.binomial(1, sigmoid_v, size=(self.n_particles, self.dim))
        
    def optimize(self):
        print("Swarming...")
        for iteration in tqdm(range(self.n_iterations)):
            self.update_velocity()
            self.update_position()
            
            # Calculate fitness
            current_fitness = self.cal_fitness(self.particles)
            
            # Update personal best
            for i in range(self.n_particles):
                if current_fitness[i] < self.pbest_scores[i]:
                    self.pbest_scores[i] = current_fitness[i]
                    self.pbest_positions[i] = self.particles[i]
                    
            # Update global best
            if current_fitness.min() < self.gbest_score:
                self.gbest_score = current_fitness.min()
                self.gbest_position = self.particles[current_fitness.argmin()]

            # Update history
            self.gbest_y_hist.append(self.gbest_score)
            self.p_best_y_hist.append(self.pbest_scores.mean())
            
            print(f"Iteration {iteration+1}/{self.n_iterations}, Best Fitness: {self.gbest_score}, Best position: {self.gbest_position}, Average Fitness: {self.pbest_scores.mean()}")
        
        return self.gbest_position, self.gbest_score