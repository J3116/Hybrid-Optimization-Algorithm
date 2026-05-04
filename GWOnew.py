import numpy as np
import matplotlib.pyplot as plt

# Ackley Function
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    return term1 + term2 + a + np.e

#  Initialize wolves
def initialize_wolves(n_wolves, dim, bounds):
    return np.random.uniform(bounds[:, 0], bounds[:, 1], (n_wolves, dim))

#  Fitness evaluation
def evaluate_fitness(wolves, objective_func):
    return np.array([objective_func(w) for w in wolves])

# Identify alpha, beta, delta
def rank_wolves(wolves, fitness):
    sorted_indices = np.argsort(fitness)
    return wolves[sorted_indices[:3]], fitness[sorted_indices[:3]]

#  Smart position update using A
def position_step(wolf, leader, a):
    r1, r2 = np.random.rand(), np.random.rand()
    A = 2 * a * r1 - a
    C = 2 * r2
    D = abs(C * leader - wolf)
    return leader - A * D

# Update all wolves
def update_positions(wolves, leaders, a, bounds):
    alpha, beta, delta = leaders
    new_wolves = []
    for wolf in wolves:
        X1 = position_step(wolf, alpha, a)
        X2 = position_step(wolf, beta, a)
        X3 = position_step(wolf, delta, a)
        new_wolf = (X1 + X2 + X3) / 3
        new_wolf = np.clip(new_wolf, bounds[:, 0], bounds[:, 1])
        new_wolves.append(new_wolf)
    return np.array(new_wolves)

#  Main GWO Function
def gwo(objective_func, bounds, n_wolves=30, max_iter=100):
    dim = bounds.shape[0]
    wolves = initialize_wolves(n_wolves, dim, bounds)
    for t in range(max_iter):
        a = 2 - 2 * (t / max_iter)
        fitness = evaluate_fitness(wolves, objective_func)
        leaders, _ = rank_wolves(wolves, fitness)
        wolves = update_positions(wolves, leaders, a, bounds)
    final_fitness = evaluate_fitness(wolves, objective_func)
    best_idx = np.argmin(final_fitness)
    return final_fitness[best_idx], wolves[best_idx]

#  Run GWO 30 Times
def run_multiple_trials(trials=30):
    dim = 30
    bounds = np.array([[-32.768, 32.768]] * dim)
    results = []

    for i in range(trials):
        np.random.seed(i)  # For consistent, fair results
        best_fit, _ = gwo(ackley, bounds)
        results.append(best_fit)
        print(f"Run {i+1:02d}: Best Fitness = {best_fit:.8f}")

    results = np.array(results)
    mean = np.mean(results)
    std_dev = np.std(results)
    perfect_runs = np.sum(results < 1e-6)

    print("\n========= Summary Over 30 Runs =========")
    print(f"Mean Best Fitness     : {mean:.8f}")
    print(f"Standard Deviation    : {std_dev:.8f}")
    print(f"Perfect Runs (<1e-6)  : {perfect_runs} / 30")
    print("========================================")

    # Optional: Plot histogram
    plt.hist(results, bins=10, color='skyblue', edgecolor='black')
    plt.title("Distribution of Best Fitness (30 GWO Runs)")
    plt.xlabel("Best Fitness")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

#  Launch test
run_multiple_trials()
