import pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.util.plotting import plot as plt
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.petal import Petal
from pymoo.visualization.radviz import Radviz
from pymoo.visualization.pcp import PCP
import numpy as np
from deap.tools import sortNondominated
import pandas as pd

i = 1
columns = ["Iteration", "Algorithm", "Hash", "Final Population Size", "Execution Time"]
results_df = pd.DataFrame(columns=columns)


def print_res_to_excel(algorithm, result, t=None):
    if t is not None:
        name = algorithm.__class__.__name__ + f"T(n_neighbors) = {t}"
    else:
        name = algorithm.__class__.__name__
    data = {
        "Iteration": i,
        "Algorithm": name,
        "Hash": result.F.sum(),
        "Final Population Size": len(result.pop),
        "Execution Time": round(result.exec_time, 2)
    }
    global results_df
    results_df = pd.concat([results_df, pd.DataFrame([data])], ignore_index=True)


def print_res(algorithm, result, t=None):
    print(f"\tAlgorithm: {algorithm.__class__.__name__}")
    if t is not None:
        print(f"T(n_neighbors) = {t}")
    print("Hash: ", result.F.sum())
    print(f"Final Population Size: {len(result.pop)}")
    print(f"Execution Time: {round(result.exec_time, 2)} seconds\n")


def plot_scatter(result, name):
    plot = Scatter(title=name + '. Scatter Plot')
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(result.F, facecolor="none", edgecolor="red")
    plot.show()


def Petal_Diagram(result, name):
    plot = Petal(bounds=[0, 1], reverse=True, title=name + '. Petal Diagram')
    plot.add(result.F[-2])
    plot.show()


def PCP_Graphic(result, name):
    plot = PCP(title=name + '. Parallel Coordinate Plots')
    plot.add(result.F)
    plot.add(result.F[-2], linewidth=3, color="green")
    plot.show()


problem = get_problem("zdt6")


def nsga2():
    algorithm = NSGA2(pop_size=100)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 450),
                   seed=1,
                   verbose=False)

    print_res(algorithm, res)
    print_res_to_excel(algorithm, res)
    plot_scatter(res, algorithm.__class__.__name__)
    Petal_Diagram(res, algorithm.__class__.__name__)
    PCP_Graphic(res, algorithm.__class__.__name__)


def nsga3():
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=100)
    algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs)
    res = minimize(problem,
                   algorithm,
                   seed=1,
                   termination=('n_gen', 450))
    print_res(algorithm, res)
    print_res_to_excel(algorithm, res)
    plot_scatter(res, algorithm.__class__.__name__)
    Petal_Diagram(res, algorithm.__class__.__name__)
    PCP_Graphic(res, algorithm.__class__.__name__)


def moead():
    n_neighbors_values = [5, 15, 40, 200]
    for n_neighbors in n_neighbors_values:
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=99)
        algorithm = MOEAD(ref_dirs, n_neighbors=n_neighbors, prob_neighbor_mating=0.5)

        res = minimize(problem,
                       algorithm,
                       ('n_gen', 450),
                       seed=1,
                       verbose=False)

        print_res(algorithm, res, n_neighbors)
        print_res_to_excel(algorithm, res, n_neighbors)
        plot_scatter(res, algorithm.__class__.__name__ + f". T(n_neighbors) = {n_neighbors}")
        Petal_Diagram(res, algorithm.__class__.__name__ + f". T(n_neighbors) = {n_neighbors}")
        PCP_Graphic(res, algorithm.__class__.__name__ + f". T(n_neighbors) = {n_neighbors}")


def spea():
    algorithm = SPEA2(pop_size=100)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 450),
                   seed=1,
                   verbose=False)
    print_res(algorithm, res)
    print_res_to_excel(algorithm, res)
    plot_scatter(res, algorithm.__class__.__name__)
    Petal_Diagram(res, algorithm.__class__.__name__)
    PCP_Graphic(res, algorithm.__class__.__name__)


def main():
    global i
    while i < 10:
        nsga2()
        nsga3()
        moead()
        spea()
        i = i + 1
    results_df.to_excel("results.xlsx", index=False)


if __name__ == "__main__":
    main()
