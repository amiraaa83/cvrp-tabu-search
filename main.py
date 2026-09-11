import argparse
import time
from src.io_utils import read_vrp_instances, get_vehicle_count_from_name
from src.tabu_search import tabu_search_cvrp
from src.visualization import plot_solution_cvrp, plot_history, plot_route_loads


def main():
    parser = argparse.ArgumentParser(
        description="Optimizator Tabu Search pentru Capacitated Vehicle Routing Problem (CVRP)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/A-VRP.zip",
        help="Calea catre arhiva .zip cu instantele VRP",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default="A-n33-k5",
        help="Numele instantei de rezolvat (ex: A-n32-k5, A-n33-k5)",
    )
    parser.add_argument(
        "--neighborhood",
        type=str,
        choices=["2-swap", "2-opt"],
        default="2-swap",
        help="Tipul vecinatatii utilizate",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Numarul maxim de iteratii Tabu Search",
    )
    parser.add_argument(
        "--tenure",
        type=int,
        default=5,
        help="Durata listei tabu (tabu tenure)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Dezactiveaza afisarea interactiva a graficelor",
    )

    args = parser.parse_args()

    print(f"[*] Incarcare instante din {args.data}...")
    instances = read_vrp_instances(args.data)

    if args.instance not in instances:
        raise KeyError(
            f"Instanta '{args.instance}' nu a fost gasita. Instante disponibile: {list(instances.keys())[:5]}..."
        )

    inst = instances[args.instance]
    k_vehicles = get_vehicle_count_from_name(args.instance)

    print(f"[*] Rezolvare {args.instance}:")
    print(f"    - Nr. vehicule (k): {k_vehicles}")
    print(f"    - Capacitate: {inst['capacity']}")
    print(f"    - Vecinatate: {args.neighborhood}")
    print(f"    - Iteratii: {args.iterations} | Tenure: {args.tenure}")

    t_start = time.time()
    best_sol, best_fit, found_hist, best_hist = tabu_search_cvrp(
        max_iterations=args.iterations,
        tabu_tenure=args.tenure,
        capacity=inst["capacity"],
        coords=inst["coords"],
        demands=inst["demands"],
        depot=inst["depot"],
        vehicle_count=k_vehicles,
        neighborhood_type=args.neighborhood,
    )
    elapsed = time.time() - t_start

    print("\n[+] Optimizare incheiata cu succes!")
    print(f"    - Timp de rulare: {elapsed:.4f} secunde")
    print(f"    - Cea mai buna distanta euclidiana: {best_fit:.2f}")
    print(f"    - Rute optime gasite: {best_sol}")

    if not args.no_plot:
        plot_history(
            best_hist,
            f"CVRP - Convergenta celei mai bune distante ({args.instance}, {args.neighborhood})",
            "Best Distance",
        )
        plot_solution_cvrp(
            best_sol,
            inst["coords"],
            inst["depot"],
            f"CVRP - Rute Optimizate ({args.instance})",
        )
        plot_route_loads(
            best_sol,
            inst["demands"],
            inst["capacity"],
            f"Incarcarea Rutelor ({args.instance})",
        )


if __name__ == "__main__":
    main()