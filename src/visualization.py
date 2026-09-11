import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from src.tabu_search import route_load


def plot_solution_cvrp(
    solution: List[List[int]],
    coords: Dict[int, Tuple[float, float]],
    depot: int,
    title: str = "CVRP - Rute Optimizate",
    output_path: str = None,
):
    """Generează reprezentarea 2D a clienților, depozitului și a fiecărei rute."""
    plt.figure(figsize=(8, 8))

    customers = [node for node in coords.keys() if node != depot]
    plt.scatter(
        [coords[n][0] for n in customers],
        [coords[n][1] for n in customers],
        c="blue",
        label="Clienti",
    )
    plt.scatter(
        [coords[depot][0]],
        [coords[depot][1]],
        c="red",
        marker="s",
        s=130,
        label="Depot",
    )

    for idx, route in enumerate(solution):
        if not route:
            continue
        full_route = [depot] + route + [depot]
        plt.plot(
            [coords[n][0] for n in full_route],
            [coords[n][1] for n in full_route],
            marker="o",
            label=f"Ruta {idx + 1}",
        )

    for node, (x, y) in coords.items():
        plt.text(x + 0.5, y + 0.5, str(node), fontsize=8)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()


def plot_history(
    history: List[float],
    title: str,
    ylabel: str,
    output_path: str = None,
):
    plt.figure(figsize=(10, 5))
    plt.plot(history, color="#2b5c8f", lw=2)
    plt.title(title)
    plt.xlabel("Iteratie")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()


def plot_route_loads(
    solution: List[List[int]],
    demands: Dict[int, int],
    capacity: int,
    title: str = "Incarcare rute vs Capacitate",
    output_path: str = None,
):
    loads = [route_load(r, demands) for r in solution]
    labels = [f"R{i + 1}" for i in range(len(loads))]

    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, loads, color="#4a90e2")
    plt.axhline(capacity, color="r", linestyle="--", label=f"Capacitate max ({capacity})")
    plt.title(title)
    plt.xlabel("Ruta")
    plt.ylabel("Incarcare Totala")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()