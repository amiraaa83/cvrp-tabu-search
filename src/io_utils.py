import math
import re
import zipfile
from typing import Any, Dict, List, Tuple


def read_vrp_instances(zip_path: str) -> Dict[str, Dict[str, Any]]:
    """Parsează o arhivă .zip ce conține instanțe standard CVRP în format TSPLIB."""
    instances = {}

    with zipfile.ZipFile(zip_path, "r") as z:
        for filename in z.namelist():
            if not filename.endswith(".vrp"):
                continue

            with z.open(filename) as f:
                lines = f.read().decode("utf-8").splitlines()

            name = None
            capacity = None
            coords = {}
            demands = {}
            depot = None
            section = None

            for line in lines:
                line = line.strip()
                if line == "" or line == "EOF":
                    continue

                if line.startswith("NAME"):
                    name = line.split(":")[1].strip()
                elif line.startswith("CAPACITY"):
                    capacity = int(line.split(":")[1].strip())
                elif line == "NODE_COORD_SECTION":
                    section = "coords"
                elif line == "DEMAND_SECTION":
                    section = "demands"
                elif line == "DEPOT_SECTION":
                    section = "depot"
                elif section == "coords":
                    parts = line.split()
                    node = int(parts[0])
                    coords[node] = (float(parts[1]), float(parts[2]))
                elif section == "demands":
                    parts = line.split()
                    node = int(parts[0])
                    demands[node] = int(parts[1])
                elif section == "depot":
                    val = int(line)
                    if val != -1:
                        depot = val

            instances[name] = {
                "name": name,
                "capacity": capacity,
                "coords": coords,
                "demands": demands,
                "depot": depot,
            }

    return instances


def get_vehicle_count_from_name(instance_name: str) -> int:
    """Extrage numărul k de vehicule din formatul numelui (ex: A-n32-k5 -> 5)."""
    match = re.search(r"-k(\d+)", instance_name)
    if match is None:
        raise ValueError(
            f"Nu s-a putut deduce numărul de vehicule din instanța: {instance_name}"
        )
    return int(match.group(1))


def get_customers(coords: Dict[int, Tuple[float, float]], depot: int) -> List[int]:
    """Returnează lista nodurilor client, excluzând depozitul."""
    return [node for node in coords.keys() if node != depot]


def compute_distance_matrix(coords: Dict[int, Tuple[float, float]]) -> Dict[int, Dict[int, int]]:
    """Calculează matricea simetrică a distanțelor euclidiene rotunjite."""
    dm = {}
    nodes = list(coords.keys())
    for i in nodes:
        dm[i] = {}
        for j in nodes:
            xd = coords[i][0] - coords[j][0]
            yd = coords[i][1] - coords[j][1]
            dist = round(math.sqrt(xd**2 + yd**2))
            dm[i][j] = dist
            dm[j][i] = dist
    return dm