import numpy as np
import matplotlib.pyplot as plt

# 🎯 Ackley Function
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

# Initialize population
def initialize_population(pop_size, dim, bounds):
    return np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(pop_size, dim)
    )

# Chase phase (exploration)
def chasePhase(xi, xs, a):  
    r1 = np.random.rand(len(xi))
    r2 = np.random.rand(len(xi))
    A = 2 * a * r1 - a
    C = 2 * r2
    D = np.abs(C * xs - xi)
    new_position = xs - A * D
    return new_position

# Attack phase (exploitation)
def attackPhase(xi, xs):
    D = np.abs(xs - xi)
    b = 1
    l = np.random.uniform(-1, 1)
    spiral = D * np.exp(b * l) * np.cos(2 * np.pi * l)
    new_position = spiral + xs
    return new_position

# Dynamic switch based on stagnation
def dynamicSwitching(fitness_i, new_fitness, stagnation_counter, threshold=5):
    if new_fitness < fitness_i:
        return True, stagnation_counter  # continue exploring
    else:
        stagnation_counter += 1
        return stagnation_counter < threshold, stagnation_counter  # switch to exploitation

# Bound checking
def clip_to_bounds(position, bounds):
    return np.clip(position, [b[0] for b in bounds], [b[1] for b in bounds])

# 🐋 Main OPA function
def run_opa(pop_size, dim, bounds, fitness_func, max_iter):
    population = initialize_population(pop_size, dim, bounds)
    fitness = np.array([fitness_func(ind) for ind in population])
    stagnation_counter = [0] * pop_size
    xs = population[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    convergence_curve = []

    for t in range(max_iter):
        a = 2 - 2 * (t / max_iter)
        for i in range(pop_size):
            should_explore, stagnation_counter[i] = dynamicSwitching(
                fitness[i], fitness[i], stagnation_counter[i]
            )

            if should_explore:
                new_pos = chasePhase(population[i], xs, a)
            else:
                new_pos = attackPhase(population[i], xs)

            new_pos = clip_to_bounds(new_pos, bounds)
            new_fit = fitness_func(new_pos)

            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_fitness:
                    xs = new_pos
                    best_fitness = new_fit

        convergence_curve.append(best_fitness)

    return xs, best_fitness, convergence_curve

# 🔁 Run OPA 30 times and summarize
def run_multiple_opa_trials(trials=30):
    dim = 30
    bounds = [(-32.768, 32.768)] * dim
    results = []

    for i in range(trials):
        np.random.seed(i)  # For reproducibility
        _, best_fit, _ = run_opa(
            pop_size=50,
            dim=dim,
            bounds=bounds,
            fitness_func=ackley,
            max_iter=100
        )
        results.append(best_fit)
        print(f"Run {i+1:02d}: Best Fitness = {best_fit:.8f}")

    results = np.array(results)
    mean = np.mean(results)
    std_dev = np.std(results)
    success_count = np.sum(results < 1e-6)

    print("\n========= Summary Over 30 Runs =========")
    print(f"Mean Best Fitness     : {mean:.8f}")
    print(f"Standard Deviation    : {std_dev:.8f}")
    print(f"Perfect Runs (<1e-6)  : {success_count} / {trials}")
    print("========================================")

    # 📊 Histogram plot
    plt.hist(results, bins=10, color='lightcoral', edgecolor='black')
    plt.title("OPA: Best Fitness Distribution over 30 Runs (Ackley)")
    plt.xlabel("Best Fitness")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# 🔥 Launch Benchmark
run_multiple_opa_trials()
