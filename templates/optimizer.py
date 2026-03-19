"""
NM i AI 2026 — Optimering Template
====================================
Losning av optimeringsproblemer med scipy og OR-Tools.

Slik fungerer optimering:
- Du har en MOLFUNKSJON du vil minimere/maksimere
- Du har BEGRENSNINGER (constraints) som ma oppfylles
- Algoritmen finner den beste losningen innenfor begrensningene

Vanlige oppgavetyper:
- Ruteplanlegging (TSP, Vehicle Routing)
- Scheduling (ressursallokering, timeplanlegging)
- Knapsack (velg items med maks verdi under vektgrense)
- Lineær/ikke-lineær optimering

Bruk:
1. Definer molfunksjon og begrensninger
2. Velg riktig solver
3. Kjor: python optimizer.py

TODO-markerte steder ma tilpasses for den spesifikke oppgaven.
"""

import numpy as np
from scipy.optimize import minimize, linprog, differential_evolution
from scipy.spatial.distance import pdist, squareform

# OR-Tools for kombinatorisk optimering
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


# === METODE 1: SCIPY — Kontinuerlig optimering ===

def scipy_minimize_example():
    """
    Eksempel: Minimer en funksjon med begrensninger.

    Bruk nol oppgaven handler om a finne optimale TALL/PARAMETRE.
    """
    print("=== Scipy Minimize ===")

    # TODO: Definer molfunksjonen
    # Eksempel: Minimer f(x, y) = x^2 + y^2 + 2*x*y
    def objective(params):
        x, y = params
        return x**2 + y**2 + 2*x*y  # TODO: Din molfunksjon

    # TODO: Definer begrensninger
    constraints = [
        {"type": "ineq", "fun": lambda p: p[0] + p[1] - 1},   # x + y >= 1
        {"type": "ineq", "fun": lambda p: 10 - p[0]},          # x <= 10
        {"type": "ineq", "fun": lambda p: 10 - p[1]},          # y <= 10
    ]

    # TODO: Definer grenser for variablene
    bounds = [(-10, 10), (-10, 10)]  # (min, max) for hver variabel

    # Optimer
    result = minimize(
        objective,
        x0=[5.0, 5.0],        # Startpunkt
        method="SLSQP",        # Algoritme (SLSQP for constraints)
        bounds=bounds,
        constraints=constraints,
    )

    print(f"  Optimal losning: {result.x}")
    print(f"  Optimal verdi: {result.fun:.4f}")
    print(f"  Suksess: {result.success}")
    return result


def scipy_global_optimization():
    """
    Global optimering med differential evolution.

    Bruk nol du har mange lokale minima og trenger a finne det GLOBALE minimum.
    Tregere enn lokal optimering, men finner bedre losninger.
    """
    print("\n=== Global Optimering ===")

    # TODO: Din molfunksjon
    def objective(params):
        x, y = params
        # Rastrigin-funksjon (mange lokale minima)
        return 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)

    bounds = [(-5.12, 5.12), (-5.12, 5.12)]

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=1000,
        seed=42,
        tol=1e-8,
    )

    print(f"  Optimal losning: {result.x}")
    print(f"  Optimal verdi: {result.fun:.6f}")
    return result


# === METODE 2: OR-TOOLS — Kombinatorisk optimering ===

def ortools_scheduling_example():
    """
    Eksempel: Job-shop scheduling.

    Bruk nol oppgaven handler om a PLANLEGGE (hvem gjor hva nol).
    """
    print("\n=== OR-Tools Scheduling ===")

    model = cp_model.CpModel()

    # TODO: Definer problemet
    # Eksempel: 3 jobber, hver med start/slutt-tid, med begrensninger
    num_jobs = 3
    max_time = 100

    # Variabler: starttid for hver jobb
    starts = {}
    ends = {}
    intervals = {}

    # TODO: Tilpass jobber og varigheter
    job_durations = [10, 20, 15]  # Varighet per jobb

    for job in range(num_jobs):
        starts[job] = model.NewIntVar(0, max_time, f"start_{job}")
        ends[job] = model.NewIntVar(0, max_time, f"end_{job}")
        intervals[job] = model.NewIntervalVar(
            starts[job], job_durations[job], ends[job], f"interval_{job}"
        )

    # Begrensning: Jobber kan IKKE overlappe (1 ressurs)
    model.AddNoOverlap(list(intervals.values()))

    # TODO: Legg til flere begrensninger
    # Eksempel: Jobb 1 ma starte etter jobb 0
    model.Add(starts[1] >= ends[0])

    # Mal: Minimer total tid (makespan)
    makespan = model.NewIntVar(0, max_time, "makespan")
    model.AddMaxEquality(makespan, list(ends.values()))
    model.Minimize(makespan)

    # Los
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30  # Tidsbegrensning
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print(f"  Status: {'Optimal' if status == cp_model.OPTIMAL else 'Feasible'}")
        print(f"  Makespan: {solver.Value(makespan)}")
        for job in range(num_jobs):
            print(f"  Jobb {job}: start={solver.Value(starts[job])}, slutt={solver.Value(ends[job])}")
    else:
        print("  Ingen losning funnet")

    return solver


def ortools_routing_example():
    """
    Eksempel: Travelling Salesman Problem (TSP).

    Bruk nol oppgaven handler om RUTEPLANLEGGING.
    """
    print("\n=== OR-Tools Routing (TSP) ===")

    # TODO: Definer avstandsmatrisen
    # Eksempel: 5 lokasjoner med tilfeldige avstander
    np.random.seed(42)
    num_locations = 5
    coords = np.random.rand(num_locations, 2) * 100

    # Beregn avstandsmatrise
    dist_matrix = squareform(pdist(coords)).astype(int)

    # Opprett routing-modell
    manager = pywrapcp.RoutingIndexManager(num_locations, 1, 0)  # num_locations, num_vehicles, depot
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Losningsparametre
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = 10

    # Los
    solution = routing.SolveWithParameters(search_params)

    if solution:
        route = []
        index = routing.Start(0)
        total_distance = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            next_index = solution.Value(routing.NextVar(index))
            total_distance += distance_callback(index, next_index)
            index = next_index
        route.append(manager.IndexToNode(index))

        print(f"  Rute: {' -> '.join(map(str, route))}")
        print(f"  Total avstand: {total_distance}")
    else:
        print("  Ingen losning funnet")

    return solution


# === METODE 3: KNAPSACK (Velg beste delmengde) ===

def knapsack_solver(items: list[dict], capacity: float) -> list[dict]:
    """
    Knapsack-problem: Velg items med maks verdi under vektgrense.

    Args:
        items: Liste med {"name": str, "weight": float, "value": float}
        capacity: Maks totalvekt

    Bruk nol oppgaven handler om a VELGE de beste elementene.
    """
    print(f"\n=== Knapsack (kapasitet: {capacity}) ===")

    solver = pywraplp.Solver.CreateSolver("SCIP")

    # Variabler: velg (1) eller ikke (0) for hvert item
    x = {}
    for i, item in enumerate(items):
        x[i] = solver.IntVar(0, 1, f"x_{i}")

    # Begrensning: total vekt <= kapasitet
    solver.Add(
        sum(items[i]["weight"] * x[i] for i in range(len(items))) <= capacity
    )

    # Mal: Maksimer total verdi
    solver.Maximize(
        sum(items[i]["value"] * x[i] for i in range(len(items)))
    )

    status = solver.Solve()

    selected = []
    if status == pywraplp.Solver.OPTIMAL:
        total_value = 0
        total_weight = 0
        for i, item in enumerate(items):
            if x[i].solution_value() > 0.5:
                selected.append(item)
                total_value += item["value"]
                total_weight += item["weight"]
                print(f"  Valgt: {item['name']} (vekt={item['weight']}, verdi={item['value']})")

        print(f"  Total verdi: {total_value}, Total vekt: {total_weight}/{capacity}")

    return selected


# === KJOR ===

if __name__ == "__main__":
    # Kjor alle eksempler

    # 1. Kontinuerlig optimering
    scipy_minimize_example()
    scipy_global_optimization()

    # 2. Scheduling
    ortools_scheduling_example()

    # 3. Ruteplanlegging (TSP)
    ortools_routing_example()

    # 4. Knapsack
    items = [
        {"name": "Item A", "weight": 10, "value": 60},
        {"name": "Item B", "weight": 20, "value": 100},
        {"name": "Item C", "weight": 30, "value": 120},
        {"name": "Item D", "weight": 15, "value": 80},
        {"name": "Item E", "weight": 25, "value": 90},
    ]
    knapsack_solver(items, capacity=50)
