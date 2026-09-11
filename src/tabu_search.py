import math
import random
from typing import Any, Dict, List, Tuple
from src.io_utils import get_customers


def copy_solution_cvrp(solution: List[List[int]]) -> List[List[int]]:
    return [route[:] for route in solution]


def route_load(route: List[int], demands: Dict[int, int]) -> int:
    return sum(demands[node] for node in route)


def is_valid_cvrp(
    solution: List[List[int]],
    capacity: int,
    coords: Dict[int, Tuple[float, float]],
    demands: Dict[int, int],
    depot: int,
    vehicle_count: int,
) -> bool:
    """Validează soluția conform restricțiilor CVRP (vehicule, capacitate, vizită unică)."""
    if len(solution) > vehicle_count:
        return False

    customers = get_customers(coords, depot)
    visited = []

    for route in solution:
        if route_load(route, demands) > capacity:
            return False
        for customer in route:
            if customer == depot or customer not in demands:
                return False
            visited.append(customer)

    return sorted(visited) == sorted(customers)


def generate_initial_solution(
    capacity: int,
    coords: Dict[int, Tuple[float, float]],
    demands: Dict[int, int],
    depot: int,
    vehicle_count: int,
) -> List[List[int]]:
    """Construiește o soluție inițială validă (Best-Fit descrescător după cerere)."""
    customers = get_customers(coords, depot)

    for customer in customers:
        if demands[customer] > capacity:
            raise ValueError("Un client depășește capacitatea maximă a vehiculului.")

    solution = [[] for _ in range(vehicle_count)]
    loads = [0 for _ in range(vehicle_count)]

    customers_order = customers[:]
    random.shuffle(customers_order)
    customers_order.sort(key=lambda c: demands[c], reverse=True)

    for customer in customers_order:
        demand = demands[customer]
        feasible_routes = [
            r for r in range(vehicle_count) if loads[r] + demand <= capacity
        ]

        if not feasible_routes:
            raise ValueError("Nu s-a putut construi o soluție inițială fezabilă.")

        best_route = feasible_routes[0]
        best_remaining = capacity - (loads[best_route] + demand)

        for r in feasible_routes:
            rem = capacity - (loads[r] + demand)
            if rem < best_remaining:
                best_route = r
                best_remaining = rem

        solution[best_route].append(customer)
        loads[best_route] += demand

    for route in solution:
        random.shuffle(route)

    return solution


def fitness_cvrp(
    solution: List[List[int]],
    capacity: int,
    coords: Dict[int, Tuple[float, float]],
    demands: Dict[int, int],
    depot: int,
) -> float:
    """Calculează distanța euclidiană totală; returnează inf dacă soluția încalcă capacitatea."""
    total_distance = 0.0

    for route in solution:
        if sum(demands[c] for c in route) > capacity:
            return float("inf")

        full_route = [depot] + route + [depot]
        for i in range(len(full_route) - 1):
            c1, c2 = coords[full_route[i]], coords[full_route[i + 1]]
            total_distance += math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    return total_distance


def all_neighbors_cvrp(
    solution: List[List[int]],
    capacity: int,
    coords: Dict[int, Tuple[float, float]],
    demands: Dict[int, int],
    depot: int,
    vehicle_count: int,
    neighborhood_type: str = "2-swap",
) -> List[Tuple[List[List[int]], Tuple[Any, ...]]]:
    """Generează toți vecinii fezabili și mutările asociate pentru lista tabu."""
    neighbors = []

    if neighborhood_type == "2-swap":
        positions = [
            (r_idx, c_idx)
            for r_idx, route in enumerate(solution)
            for c_idx in range(len(route))
        ]
        for a in range(len(positions) - 1):
            for b in range(a + 1, len(positions)):
                r1, p1 = positions[a]
                r2, p2 = positions[b]
                c1, c2 = solution[r1][p1], solution[r2][p2]

                cand = copy_solution_cvrp(solution)
                cand[r1][p1], cand[r2][p2] = cand[r2][p2], cand[r1][p1]

                if is_valid_cvrp(cand, capacity, coords, demands, depot, vehicle_count):
                    move = ("2-swap", min(c1, c2), max(c1, c2))
                    neighbors.append((cand, move))

    elif neighborhood_type == "2-opt":
        for r_idx, route in enumerate(solution):
            if len(route) < 2:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    cand = copy_solution_cvrp(solution)
                    c1, c2 = cand[r_idx][i], cand[r_idx][j]
                    cand[r_idx][i : j + 1] = reversed(cand[r_idx][i : j + 1])

                    if is_valid_cvrp(cand, capacity, coords, demands, depot, vehicle_count):
                        move = ("2-opt", r_idx, min(c1, c2), max(c1, c2))
                        neighbors.append((cand, move))
    else:
        raise ValueError("neighborhood_type trebuie să fie '2-swap' sau '2-opt'")

    return neighbors


def tabu_search_cvrp(
    max_iterations: int,
    tabu_tenure: int,
    capacity: int,
    coords: Dict[int, Tuple[float, float]],
    demands: Dict[int, int],
    depot: int,
    vehicle_count: int,
    neighborhood_type: str = "2-swap",
) -> Tuple[List[List[int]], float, List[float], List[float]]:
    """Algoritmul Tabu Search complet pentru CVRP cu criteriu de aspirație."""
    current_sol = generate_initial_solution(capacity, coords, demands, depot, vehicle_count)
    current_fitness = fitness_cvrp(current_sol, capacity, coords, demands, depot)

    best_sol = copy_solution_cvrp(current_sol)
    best_fitness = current_fitness

    tabu: Dict[Tuple[Any, ...], int] = {}
    found_solutions = [current_fitness]
    best_history = [best_fitness]

    for _ in range(max_iterations):
        neighbors = all_neighbors_cvrp(
            current_sol, capacity, coords, demands, depot, vehicle_count, neighborhood_type
        )

        candidate_sol = None
        candidate_move = None
        candidate_fitness = float("inf")

        for x, move in neighbors:
            fx = fitness_cvrp(x, capacity, coords, demands, depot)
            is_tabu = move in tabu and tabu[move] > 0
            aspiration = fx < best_fitness

            if (not is_tabu or aspiration) and fx < candidate_fitness:
                candidate_sol = copy_solution_cvrp(x)
                candidate_move = move
                candidate_fitness = fx

        if candidate_sol is None:
            break

        # Decrementare durată tabu
        for m in list(tabu.keys()):
            tabu[m] -= 1
            if tabu[m] <= 0:
                del tabu[m]

        tabu[candidate_move] = tabu_tenure

        current_sol = copy_solution_cvrp(candidate_sol)
        current_fitness = candidate_fitness
        found_solutions.append(current_fitness)

        if current_fitness < best_fitness:
            best_sol = copy_solution_cvrp(current_sol)
            best_fitness = current_fitness

        best_history.append(best_fitness)

    return best_sol, best_fitness, found_solutions, best_history