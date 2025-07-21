import os
import time
from datetime import datetime
import json

import numpy as np

# Run evaluations in parallel
from concurrent.futures import ProcessPoolExecutor

# Import CEM algorithm
from cem.algo import CemParams, CrossEntropyMethodMixed


# Function to evaluate the CEM-MD population
def eval_pop(input):
    xd = input[0]
    xc = input[1]

    xd[0] += 1

    n_jumps = xd[0] + 1
    ids = []
    for i in range(n_jumps):
        ids += [xd[1 + i]]

    print(n_jumps, ids)

    fitness = 0.0
    ## Run trajectory optimisation here ##
    p0 = P0_INIT
    for i in range(n_jumps):
        res = eng.optimize_cpp_mex(
            matlab.double(p0), matlab.double(pf), Fleg_max, Fr_max, Fr_min, mu, params
        )
        fitness += calc_fitness(res)
        p0 = res["pf"]
        pass

    ## Calculate fitness (by default the algorithm maximises) ##
    # if xd[0] == 0:
    #     fitness += 50
    # if xd[1] == 1:
    #     fitness += 100
    # for num in xc:
    #     fitness += -((num - 1.5) ** 2)
    return fitness


# Set up parameters
p = CemParams()
p.seed = int(time.time())
p.n_threads = 16
# General CEM-MD Parameters
p.cem_iters = 15
p.pop_size = 100
p.n_elites = int(p.pop_size * 0.8)
p.decrease_pop_factor = 1.0
p.fraction_elites_reused = 0.0
# Discrete
p.dim_discrete = 5
p.n_values = [3] + [19 for _ in range(4)]
p.init_probs = [
    [1.0 / p.n_values[i] for _ in range(p.n_values[i])] for i in range(p.dim_discrete)
]
p.min_prob = 0.05
# Continuous
MAX_N_PATCHES = 5
p.dim_continuous = 2 * MAX_N_PATCHES
p.max_value_continuous = np.full(p.dim_continuous, 1.0)
p.min_value_continuous = np.full(p.dim_continuous, -1.0)
p.init_mu_continuous = np.full(p.dim_continuous, 1.0)
p.init_std_continuous = np.full(p.dim_continuous, 1.0)
p.min_std_continuous = np.full(p.dim_continuous, 1e-3)

algo = CrossEntropyMethodMixed(p)

# Main loop
cost_hist = np.zeros(p.cem_iters)
start = time.time()
for k in range(p.cem_iters):
    # Generate population by sampling the distributions
    algo.generate_population_discrete()
    algo.generate_population_continuous()
    xd = algo.population_discrete  # shape: dim_discrete x pop_size
    xc = algo.population_continuous  # shape: dim_continuous x pop_size

    # Organise inputs to pass to process pool
    inputs = [[xd[:, i].tolist(), xc[:, i].tolist(), p] for i in range(p.pop_size)]

    # Evaluate population in parallel
    with ProcessPoolExecutor(max_workers=p.n_threads) as executor:
        fitness = list(executor.map(eval_pop, inputs))

    # Evaluate population and update distributions
    algo.evaluate_population(fitness)
    algo.update_distributions()

    cost_hist[k] = algo.log.best_value

    ## Early exit

    # Print intermediate stats
    # print(algo.log.iterations, "(", algo.log.func_evals, "): ", algo.log.best_value)
    # print("discrete probabilities: \n", algo.probs)
    # print("mean: \n", algo.mu.T)
    # print("sigma: \n", algo.std_devs.T)
    # print(" ")

# Save wall-time
end = time.time()
wall_time = end - start

xd = algo.best_discrete
xc = algo.best_continuous

# Generate and save report json
report = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "iterations": k + 1,
    },
    "solution": {
        "elite_cost_history": cost_hist.tolist(),
        "best_discrete": xd.tolist(),
        "best_continuous": xc.tolist(),
        "wall_time_sec": wall_time,
    },
}

# Save to file
filename = f"cem_solution.json"
save_path = os.path.join(os.path.abspath(os.getcwd()), filename)

with open(save_path, "w") as f:
    json.dump(report, f, indent=2)
