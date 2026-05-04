import numpy as np
import matplotlib.pyplot as plt

# ===================== Benchmark Functions =====================
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

def levy(x):
    w = 1 + (x - 1)/4
    return np.sin(np.pi*w[0])**2 + np.sum((w[:-1]-1)**2 * (1+10*np.sin(np.pi*w[:-1]+1)**2)) + (w[-1]-1)**2 * (1+np.sin(2*np.pi*w[-1])**2)

def michalewicz(x):
    return -np.sum(np.sin(x) * np.sin(((np.arange(1,len(x)+1)) * x**2)/np.pi)**20)

def dixon_price(x):
    return (x[0]-1)**2 + np.sum([(i+1)*(2*x[i]**2 - x[i-1])**2 for i in range(1, len(x))])

benchmark_functions = {
    "Sphere": lambda x: np.sum(x**2),
    "Rastrigin": lambda x: 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x)),
    "Ackley": ackley,
    "Rosenbrock": lambda x: np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2),
    "Griewank": lambda x: 1 + np.sum(x**2)/4000 - np.prod(np.cos(x / np.sqrt(np.arange(1,len(x)+1)))),
    "Schwefel": lambda x: 418.9829*len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))),
    "Zakharov": lambda x: np.sum(x**2) + (np.sum(0.5*np.arange(1,len(x)+1)*x))**2 + (np.sum(0.5*np.arange(1,len(x)+1)*x))**4,
    "DixonPrice": dixon_price,
    "Levy": levy,
    "Michalewicz": michalewicz,
}

# ===================== OPA Components =====================
def initialize_population(pop_size, dim, bounds):
    return np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(pop_size, dim)
    )

def chasePhase(xi, xs, a):
    r1 = np.random.rand(len(xi))
    r2 = np.random.rand(len(xi))
    A = 2 * a * r1 - a
    C = 2 * r2
    D = np.abs(C * xs - xi)
    new_position = xs - A * D
    return new_position

def attackPhase(xi, xs):
    D = np.abs(xs - xi)
    b = 1
    l = np.random.uniform(-1, 1)
    spiral = D * np.exp(b * l) * np.cos(2 * np.pi * l)
    new_position = spiral + xs
    return new_position

def dynamicSwitching(fitness_i, new_fitness, stagnation_counter, threshold=5):
    if new_fitness < fitness_i:
        return True, stagnation_counter
    else:
        stagnation_counter += 1
        return stagnation_counter < threshold, stagnation_counter

def clip_to_bounds(position, bounds):
    return np.clip(position, [b[0] for b in bounds], [b[1] for b in bounds])

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

# ===================== Runner for All Functions =====================
def run_all_opa_trials(trials=30):
    dim = 30
    for name, func in benchmark_functions.items():
        print(f"\n=== {name} Function ===")
        bounds = [(-100, 100)] * dim
        if name == "Schwefel":
            bounds = [(-500, 500)] * dim
        elif name == "Ackley":
            bounds = [(-32.768, 32.768)] * dim

        results = []

        for i in range(trials):
            np.random.seed(i)
            _, best_fit, _ = run_opa(
                pop_size=50,
                dim=dim,
                bounds=bounds,
                fitness_func=func,
                max_iter=100
            )
            results.append(best_fit)

        results = np.array(results)
        print(f"Mean: {results.mean():.5e} | Std: {results.std():.5e} | Min: {results.min():.5e} | Max: {results.max():.5e}")

        plt.hist(results, bins=10, alpha=0.7, edgecolor='black')
        plt.title(f"OPA: Best Fitness Distribution ({name})")
        plt.xlabel("Best Fitness")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# 🔥 Launch
run_all_opa_trials()
