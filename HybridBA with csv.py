import numpy as np
import matplotlib.pyplot as plt
import csv

# ===================== Benchmark Functions =====================
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
def zakharov(x): return np.sum(x**2) + (np.sum(0.5*np.arange(1,len(x)+1)*x))**2 + (np.sum(0.5*np.arange(1,len(x)+1)*x))**4
def dixon_price(x): return (x[0]-1)**2 + np.sum([(i+2)*(2*x[i+1]**2 - x[i])**2 for i in range(len(x)-1)])
def levy(x):
    w = 1 + (x - 1)/4
    return np.sin(np.pi*w[0])**2 + np.sum((w[:-1]-1)**2 * (1+10*np.sin(np.pi*w[:-1]+1)**2)) + (w[-1]-1)**2 * (1+np.sin(2*np.pi*w[-1])**2)
def michalewicz(x): return -np.sum(np.sin(x) * np.sin(((np.arange(1,len(x)+1)) * x**2)/np.pi)**20)

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

# ===================== Bounds helper (classic ranges) =====================
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

# ===================== Utility =====================
def initialize_population(pop_size, dim, bounds):
    return np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, dim))

def clip_to_bounds(position, bounds):
    return np.clip(position, [b[0] for b in bounds], [b[1] for b in bounds])

# ===================== Hybrid BA (GWO-style with OPA exploration) =====================
def explore_opa(xi, xs, a):  # OPA-style exploration (kept per your code)
    r1, r2 = np.random.rand(len(xi)), np.random.rand(len(xi))
    A = 2 * a * r1 - a
    C = 2 * r2
    D = np.abs(C * xs - xi)
    return xs - A * D

def exploit_gwo(xi, alpha, beta, delta, a):  # GWO-style exploitation
    new = 0
    for leader in [alpha, beta, delta]:
        r1, r2 = np.random.rand(), np.random.rand()
        A = 2 * a * r1 - a
        C = 2 * r2
        D = np.abs(C * leader - xi)
        new += leader - A * D
    return new / 3

def hybrid_ba(objective_func, bounds, pop_size=30, max_iter=100):
    dim = bounds.shape[0]
    wolves = initialize_population(pop_size, dim, bounds)
    convergence = []

    for t in range(max_iter):
        a = 2 - 2 * (t / max_iter)
        fitness = np.array([objective_func(w) for w in wolves])
        sorted_idx = np.argsort(fitness)
        alpha, beta, delta = wolves[sorted_idx[:3]]

        new_wolves = []
        for w in wolves:
            if np.random.rand() < 0.5:
                new = explore_opa(w, alpha, a)              # Exploration
            else:
                new = exploit_gwo(w, alpha, beta, delta, a)  # Exploitation

            new = clip_to_bounds(new, bounds)
            new_wolves.append(new)

        wolves = np.array(new_wolves)
        # best fitness this iteration:
        convergence.append(np.min([objective_func(w) for w in wolves]))

    return np.min([objective_func(w) for w in wolves]), convergence

# ===================== Plot helper =====================
def plot_convergence(name, curves, show_std_band=True):
    curves = np.array(curves)
    avg_curve = curves.mean(axis=0)

    plt.figure(figsize=(7,5))
    plt.plot(avg_curve, label='Hybrid BA (avg)')

    if show_std_band and curves.shape[0] > 1:
        std_curve = curves.std(axis=0)
        iters = np.arange(len(avg_curve))
        plt.fill_between(iters, avg_curve - std_curve, avg_curve + std_curve, alpha=0.15, label='±1 std')

    plt.title(f"Average Convergence Curve - {name}")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ===================== Execution (all functions) =====================
def run_hybrid_ba_all(trials=30, dim=30, max_iter=100, pop_size=30,
                      show_plots=False, show_std_band=True,
                      save_csv=True, csv_path="hybrid_ba_summary.csv",
                      print_per_function=True):
    summary_rows = []  # [name, Best, Worst, Mean, Std]

    for name, func in benchmark_functions.items():
        bounds = get_bounds(name, dim)
        stats = []
        all_curves = []

        for run in range(trials):
            np.random.seed(run)  # reproducible runs
            best, curve = hybrid_ba(func, bounds, pop_size=pop_size, max_iter=max_iter)
            stats.append(best)
            all_curves.append(curve)

        stats = np.array(stats)
        best_v  = stats.min()
        worst_v = stats.max()
        mean_v  = stats.mean()
        std_v   = stats.std()
        summary_rows.append([name, best_v, worst_v, mean_v, std_v])

        if print_per_function:
            print(f"\nFunction: {name}")
            print(f"{'Best':>12} {'Worst':>12} {'Mean':>12} {'Std':>12}")
            print(f"{best_v:12.5e} {worst_v:12.5e} {mean_v:12.5e} {std_v:12.5e}")

        if show_plots:
            plot_convergence(name, all_curves, show_std_band=show_std_band)

    # ---- Final summary table (Best, Worst, Mean, Std) ----
    header = ["Function", "Best", "Worst", "Mean", "Std"]
    col_w = [max(len(header[0]), 12), 14, 14, 14, 14]
    print("\n" + "="*(sum(col_w)+4))
    print(f"{header[0]:<{col_w[0]}} {header[1]:>{col_w[1]}} {header[2]:>{col_w[2]}} {header[3]:>{col_w[3]}} {header[4]:>{col_w[4]}}")
    print("-"*(sum(col_w)+4))
    for name, best_v, worst_v, mean_v, std_v in summary_rows:
        print(f"{name:<{col_w[0]}} {best_v:>{col_w[1]}.5e} {worst_v:>{col_w[2]}.5e} {mean_v:>{col_w[3]}.5e} {std_v:>{col_w[4]}.5e}")
    print("="*(sum(col_w)+4))

    # ---- CSV (same order) ----
    if save_csv:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in summary_rows:
                writer.writerow([r[0], f"{r[1]:.10e}", f"{r[2]:.10e}", f"{r[3]:.10e}", f"{r[4]:.10e}"])
        print(f"\nSaved summary to: {csv_path}")

    return summary_rows

# ===================== Run All =====================
if __name__ == "__main__":
    run_hybrid_ba_all(
        trials=30, dim=30, max_iter=100, pop_size=30,
        show_plots=False, show_std_band=True,
        save_csv=True, csv_path="hybrid_ba_summary.csv",
        print_per_function=True
    )
