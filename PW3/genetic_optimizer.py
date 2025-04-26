import random
import math
import matplotlib.pyplot as plt
import numpy as np

class GeneticOptimizer:

    def __init__(self, func, x_min, x_max, pop_size=9000, mutation_rate=0.2, generations=100, mutation_amount=0.1):
        self.func = func
        self.x_min = x_min
        self.x_max = x_max
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = self.init_population()
        self.mutation_amount = mutation_amount

    def init_population(self):
        random.seed(random.random())
        return [random.uniform(self.x_min, self.x_max) for _ in range(self.pop_size)]

    def fitness(self, x, mode='max'):
        y = self.func(x)
        return y if mode == 'max' else -y  # Мінімум беремо -(Y(x))

    def selection(self, mode='max'):
        selected = []
        for _ in range(len(self.population)):
            i, j = random.sample(range(len(self.population)), 2)
            best = self.population[i] if self.fitness(self.population[i], mode) > self.fitness(self.population[j], mode) else self.population[j]
            selected.append(best)
        self.population = selected

    def crossover(self):
        random.shuffle(self.population)
        offspring = []
        for i in range(0, len(self.population), 2):
            if i + 1 < len(self.population):
                alpha = random.random()
                child1 = alpha * self.population[i] + (1 - alpha) * self.population[i + 1]
                child2 = alpha * self.population[i + 1] + (1 - alpha) * self.population[i]
                offspring.extend([child1, child2])
        self.population = offspring

    def mutate(self):
        for i in range(len(self.population)):
            if random.random() < self.mutation_rate:
                self.population[i] += random.uniform(-self.mutation_amount, self.mutation_amount)
                self.population[i] = max(self.x_min, min(self.x_max, self.population[i]))

    def optimize(self, mode='max'):
        for _ in range(self.generations):
            self.selection(mode)
            self.crossover()
            self.mutate()
        best_x = max(self.population, key=lambda x: self.fitness(x, mode))
        best_y = self.func(best_x)
        self.init_population()
        return best_x, best_y


def func(x):
    return x**3 + math.cos(15*x)


optimizer_max = GeneticOptimizer(func, x_min=-2, x_max=2)
optimizer_min = GeneticOptimizer(func, x_min=-2, x_max=2)

# Пошук максимуму і мінімуму
max_x, max_y = optimizer_max.optimize(mode='max')
min_x, min_y = optimizer_min.optimize(mode='min')

print("\n=== \tРезультати \t===")
print(f"🔴 Максимум функції: Y({max_x:.5f}) = {max_y:.5f}")
print(f"🟢 Мінімум функції: Y({min_x:.5f}) = {min_y:.5f}")
print("===========================")

# Візуалізація
x_vals = np.linspace(-2, 2, 400)
y_vals = [func(x) for x in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="Функція Y(x)", color="blue")
plt.scatter(max_x, max_y, color="red", marker="o", label=f"Максимум ({max_x:.3f}, {max_y:.3f})", s=100)
plt.scatter(min_x, min_y, color="green", marker="o", label=f"Мінімум ({min_x:.3f}, {min_y:.3f})", s=100)

plt.xlabel("x")
plt.ylabel("Y(x)")
plt.title("Оптимізація функції генетичним алгоритмом")
plt.legend()
plt.grid()
plt.show()
