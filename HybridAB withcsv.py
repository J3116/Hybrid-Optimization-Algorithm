import numpy as np
import matplotlib.pyplot as plt
import csv

# ========== Benchmark Functions ==========
def sphere(x): return np.sum(x**2)

def rastrigin(x): return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))

def ackley(x):
    a, b, c = 20, 0.2, 2*np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c*x))
    return -a*np.exp(-b*np.sqrt(sum1/d)) - np.exp(sum2/d) + a + np.e

def rosenbrock(x): return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def griewank(x): return 1 + np.sum(x**2)/4000 - np.prod(np.cos(x / np.sqrt(np.arange(1,len(x)+1))))

def schwefel(x): return 418.9829*len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def zakharov(x):
    return np.sum(x**2) + (np.sum(0.5*np.arange(1,len(x)+1)*x))**2 + (np.sum(0.5*np.arange(1,len(x)+1)*x))**4

def dixon_price(x):
    return (x[0]-1)**2 + np.sum([(i+2)*(2*x[i+1]**2 - x[i])**2 for i in range(len(x)-1)])

def levy(x):
    w = 1 + (x - 1)/4
    return np.sin(np.pi*w[0])**2 + np.sum((w[:-1]-1)**2 * (1+10*np.sin(np.pi*w[:-1]+1)**2)) + (w[-1]-1)**2 * (1+np.sin(2*np.pi*w[-1])**2)

def michalewicz(x):
    return -np.sum(np.sin(x) * np.sin(((np.arange(1,len(x)+1)) * x**2)/np.pi)**20)

benchmark_functions = {
    "Sphere": sphere,
    "Rastrigin": rastrigin,
    "Ackley": ackley,
    "Rosenbrock": rosenbrock,
    "Griewank": griewank,
    "Schwefel": schwefel,
    "Zakharov": zakharov,
    "DixonPrice": dixon_price,
    "Levy": levy,
    "Michalewicz": michalewicz,
}

# ========== Phases ==========
def explore_gwo(xi, xs, a):
    r1 = np.random.rand(len(xi))
    r2 = np.random.rand(len(xi))
    A = 2 * a * r1 - a
    C = 2 * r2
    D = np.abs(C * xs - xi)
    return xs - A * D

def exploit_opa(xi, xs):
    D = np.abs(xs - xi)
    l = np.random.uniform(-1, 1)
    spiral = D * np.exp(l) * np.cos(2 * np.pi * l)
    return xs + spiral

def clip_to_bounds(position, bounds):
    return np.clip(position, [b[0] for b in bounds], [b[1] for b in bounds])

# ========== Classic bounds helper (adjust as you like) ==========
def get_bounds(name, dim):
    if name == "Schwefel":
        lower, upper = -500, 500
    elif name == "Michalewicz":
        lower, upper = 0.0, np.pi
    elif name == "Rastrigin":
        lower, upper = -5.12, 5.12
    elif name == "Ackley":
        lower, upper = -32.768, 32.768
    elif name == "Griewank":
        lower, upper = -600, 600
    elif name == "Rosenbrock":
        lower, upper = -2.048, 2.048
    else:
        lower, upper = -100, 100
    return np.array([[lower, upper]] * dim)

# ========== Main Hybrid AB ==========
def run_hybrid_ab(pop_size, dim, bounds, fitness_func, max_iter, stagnation_threshold=3):
    population = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(pop_size, dim)
    )
    fitness = np.array([fitness_func(ind) for ind in population])
    stagnation = np.zeros(pop_size, dtype=int)

    best_idx = np.argmin(fitness)
    xs = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    convergence_curve = []

    for t in range(max_iter):
        a = 2 - 2 * (t / max_iter)
        for i in range(pop_size):
            if stagnation[i] < stagnation_threshold:
                new_pos = explore_gwo(population[i], xs, a)
            else:
                new_pos = exploit_opa(population[i], xs)

            new_pos = clip_to_bounds(new_pos, bounds)
            new_fit = fitness_func(new_pos)

            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                stagnation[i] = 0
                if new_fit < best_fitness:
                    best_fitness = new_fit
                    xs = new_pos.copy()
            else:
                stagnation[i] += 1

        convergence_curve.append(best_fitness)

    return best_fitness, convergence_curve

# ========== Runner with Stats + (optional) Convergence Plot + CSV ==========
def run_all(trials=30, dim=30, max_iter=100, pop_size=30,
            show_std_band=True, show_plots=False, save_csv=True, csv_path="hybrid_ab_summary.csv"):
    summary_rows = []

    for name, func in benchmark_functions.items():
        bounds = get_bounds(name, dim)

        best_vals = []
        curves = []

        for run in range(trials):
            np.random.seed(run)
            best, curve = run_hybrid_ab(
                pop_size=pop_size,
                dim=dim,
                bounds=bounds,
                fitness_func=func,
                max_iter=max_iter
            )
            best_vals.append(best)
            curves.append(curve)

        best_vals = np.array(best_vals)
        curves = np.array(curves)

        # Per-function quick stats (ORDER: Best, Worst, Mean, Std)
        best_v  = best_vals.min()
        worst_v = best_vals.max()
        mean_v  = best_vals.mean()
        std_v   = best_vals.std()

        print(f"\nFunction: {name}")
        print(f"{'Best':>12} {'Worst':>12} {'Mean':>12} {'Std':>12}")
        print(f"{best_v:12.5e} {worst_v:12.5e} {mean_v:12.5e} {std_v:12.5e}")

        # Collect for final table/CSV in the same order
        summary_rows.append([name, best_v, worst_v, mean_v, std_v])

        # Convergence plot (optional)
        if show_plots:
            avg_curve = curves.mean(axis=0)
            plt.figure(figsize=(7,5))
            plt.plot(avg_curve, label="Hybrid AB (avg)")
            if show_std_band and curves.shape[0] > 1:
                std_curve = curves.std(axis=0)
                plt.fill_between(np.arange(len(avg_curve)),
                                 avg_curve - std_curve,
                                 avg_curve + std_curve,
                                 alpha=0.15, label="±1 std")
            plt.title(f"Average Convergence: {name}")
            plt.xlabel("Iteration")
            plt.ylabel("Best Fitness")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    # ---- One final summary table (ORDER: Best, Worst, Mean, Std) ----
    header = ["Function", "Best", "Worst", "Mean", "Std"]
    col_w = [max(len(header[0]), 12), 14, 14, 14, 14]  # uses header[0], no 'h'
    print("\n" + "="*(sum(col_w)+4))
    print(f"{header[0]:<{col_w[0]}} {header[1]:>{col_w[1]}} {header[2]:>{col_w[2]}} {header[3]:>{col_w[3]}} {header[4]:>{col_w[4]}}")
    print("-"*(sum(col_w)+4))
    for func_name, best_v, worst_v, mean_v, std_v in summary_rows:
        print(f"{func_name:<{col_w[0]}} {best_v:>{col_w[1]}.5e} {worst_v:>{col_w[2]}.5e} {mean_v:>{col_w[3]}.5e} {std_v:>{col_w[4]}.5e}")
    print("="*(sum(col_w)+4))

    # ---- Save CSV (same order) ----
    if save_csv:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in summary_rows:
                writer.writerow([r[0],
                                 f"{r[1]:.10e}",  # Best
                                 f"{r[2]:.10e}",  # Worst
                                 f"{r[3]:.10e}",  # Mean
                                 f"{r[4]:.10e}"]) # Std
        print(f"\nSaved summary to: {csv_path}")

# ========== Go ==========
run_all(trials=30, dim=30, max_iter=100, pop_size=30,
        show_std_band=True, show_plots=False,
        save_csv=True, csv_path="hybrid_ab_summary.csv")
