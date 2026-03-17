import argparse
import time
import numpy as np
import visualizer3d
from numba import njit, prange, set_num_threads, get_num_threads

# Unités:
# - Distance: année-lumière (ly)
# - Masse: masse solaire (M_sun)
# - Vitesse: année-lumière par an (ly/an)
# - Temps: année

# Constante gravitationnelle en unités [ly^3 / (M_sun * an^2)]
G = 1.560339e-13


def generate_star_color(mass: float) -> tuple[int, int, int]:
    if mass > 5.0:
        return (150, 180, 255)
    if mass > 2.0:
        return (255, 255, 255)
    if mass >= 1.0:
        return (255, 255, 200)
    return (255, 150, 100)


@njit(inline="always")
def _to_morse_index(ix: int, iy: int, iz: int, n_cells: np.ndarray) -> int:
    return ix + iy * n_cells[0] + iz * n_cells[0] * n_cells[1]


@njit(inline="always")
def _cell_index_from_position(pos: np.ndarray, grid_min: np.ndarray, cell_size: np.ndarray, n_cells: np.ndarray):
    cell_idx = np.floor((pos - grid_min) / cell_size).astype(np.int64)
    for i in range(3):
        if cell_idx[i] >= n_cells[i]:
            cell_idx[i] = n_cells[i] - 1
        elif cell_idx[i] < 0:
            cell_idx[i] = 0
    return cell_idx


@njit(parallel=True)
def update_stars_in_grid(
    cell_start_indices: np.ndarray,
    body_indices: np.ndarray,
    cell_masses: np.ndarray,
    cell_com_positions: np.ndarray,
    masses: np.ndarray,
    positions: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
):
    n_bodies = positions.shape[0]
    n_total_cells = np.prod(n_cells)
    cell_start_indices.fill(-1)

    cell_counts = np.zeros(shape=(n_total_cells,), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = _cell_index_from_position(positions[ibody], grid_min, cell_size, n_cells)
        morse_idx = _to_morse_index(cell_idx[0], cell_idx[1], cell_idx[2], n_cells)
        cell_counts[morse_idx] += 1

    running_index = 0
    for i in range(n_total_cells):
        cell_start_indices[i] = running_index
        running_index += cell_counts[i]
    cell_start_indices[n_total_cells] = running_index

    current_counts = np.zeros(shape=(n_total_cells,), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = _cell_index_from_position(positions[ibody], grid_min, cell_size, n_cells)
        morse_idx = _to_morse_index(cell_idx[0], cell_idx[1], cell_idx[2], n_cells)
        index_in_cell = cell_start_indices[morse_idx] + current_counts[morse_idx]
        body_indices[index_in_cell] = ibody
        current_counts[morse_idx] += 1

    for i in prange(n_total_cells):
        cell_mass = 0.0
        com_position = np.zeros(3, dtype=np.float32)
        start_idx = cell_start_indices[i]
        end_idx = cell_start_indices[i + 1]
        for j in range(start_idx, end_idx):
            ibody = body_indices[j]
            m = masses[ibody]
            cell_mass += m
            com_position += positions[ibody] * m
        if cell_mass > 0.0:
            com_position /= cell_mass
        cell_masses[i] = cell_mass
        cell_com_positions[i] = com_position


@njit(parallel=True)
def compute_acceleration(
    positions: np.ndarray,
    masses: np.ndarray,
    cell_start_indices: np.ndarray,
    body_indices: np.ndarray,
    cell_masses: np.ndarray,
    cell_com_positions: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
):
    n_bodies = positions.shape[0]
    a = np.zeros_like(positions)
    for ibody in prange(n_bodies):
        pos = positions[ibody]
        cell_idx = _cell_index_from_position(pos, grid_min, cell_size, n_cells)

        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    morse_idx = _to_morse_index(ix, iy, iz, n_cells)
                    if (abs(ix - cell_idx[0]) > 2) or (abs(iy - cell_idx[1]) > 2) or (abs(iz - cell_idx[2]) > 2):
                        cell_com = cell_com_positions[morse_idx]
                        cell_mass = cell_masses[morse_idx]
                        if cell_mass > 0.0:
                            direction = cell_com - pos
                            distance = np.sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
                            if distance > 1.0e-10:
                                inv_dist3 = 1.0 / (distance * distance * distance)
                                a[ibody, :] += G * direction[:] * inv_dist3 * cell_mass
                    else:
                        start_idx = cell_start_indices[morse_idx]
                        end_idx = cell_start_indices[morse_idx + 1]
                        for j in range(start_idx, end_idx):
                            jbody = body_indices[j]
                            if jbody != ibody:
                                direction = positions[jbody] - pos
                                distance = np.sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
                                if distance > 1.0e-10:
                                    inv_dist3 = 1.0 / (distance * distance * distance)
                                    a[ibody, :] += G * direction[:] * inv_dist3 * masses[jbody]
    return a


class SpatialGrid:
    def __init__(self, positions: np.ndarray, nb_cells_per_dim: tuple[int, int, int]):
        self.min_bounds = np.min(positions, axis=0) - 1.0e-6
        self.max_bounds = np.max(positions, axis=0) + 1.0e-6
        self.n_cells = np.array(nb_cells_per_dim, dtype=np.int64)
        self.cell_size = (self.max_bounds - self.min_bounds) / self.n_cells
        n_total_cells = np.prod(self.n_cells)
        self.cell_start_indices = np.full(n_total_cells + 1, -1, dtype=np.int64)
        self.body_indices = np.empty(shape=(positions.shape[0],), dtype=np.int64)
        self.cell_masses = np.zeros(shape=(n_total_cells,), dtype=np.float32)
        self.cell_com_positions = np.zeros(shape=(n_total_cells, 3), dtype=np.float32)

    def update(self, positions: np.ndarray, masses: np.ndarray):
        update_stars_in_grid(
            self.cell_start_indices,
            self.body_indices,
            self.cell_masses,
            self.cell_com_positions,
            masses,
            positions,
            self.min_bounds,
            self.cell_size,
            self.n_cells,
        )


class NBodySystem:
    def __init__(self, filename: str, ncells_per_dir: tuple[int, int, int] = (20, 20, 1)):
        positions = []
        velocities = []
        masses = []

        self.max_mass = 0.0
        self.box = np.array([[-1.0e-6, -1.0e-6, -1.0e-6], [1.0e-6, 1.0e-6, 1.0e-6]], dtype=np.float64)
        with open(filename, "r", encoding="utf-8") as fich:
            line = fich.readline()
            while line:
                data = line.split()
                masses.append(float(data[0]))
                positions.append([float(data[1]), float(data[2]), float(data[3])])
                velocities.append([float(data[4]), float(data[5]), float(data[6])])
                self.max_mass = max(self.max_mass, masses[-1])

                for i in range(3):
                    self.box[0][i] = min(self.box[0][i], positions[-1][i] - 1.0e-6)
                    self.box[1][i] = max(self.box[1][i], positions[-1][i] + 1.0e-6)

                line = fich.readline()

        self.positions = np.array(positions, dtype=np.float32)
        self.velocities = np.array(velocities, dtype=np.float32)
        self.masses = np.array(masses, dtype=np.float32)
        self.colors = [generate_star_color(m) for m in masses]
        self.grid = SpatialGrid(self.positions, ncells_per_dir)
        self.grid.update(self.positions, self.masses)

    def update_positions(self, dt: float):
        a = compute_acceleration(
            self.positions,
            self.masses,
            self.grid.cell_start_indices,
            self.grid.body_indices,
            self.grid.cell_masses,
            self.grid.cell_com_positions,
            self.grid.min_bounds,
            self.grid.cell_size,
            self.grid.n_cells,
        )
        self.positions += self.velocities * dt + 0.5 * a * dt * dt
        self.grid.update(self.positions, self.masses)
        a_new = compute_acceleration(
            self.positions,
            self.masses,
            self.grid.cell_start_indices,
            self.grid.body_indices,
            self.grid.cell_masses,
            self.grid.cell_com_positions,
            self.grid.min_bounds,
            self.grid.cell_size,
            self.grid.n_cells,
        )
        self.velocities += 0.5 * (a + a_new) * dt


system: NBodySystem | None = None


def update_positions(dt: float):
    global system
    system.update_positions(dt)
    return system.positions


def run_visual_simulation(filename, ncells_per_dir=(20, 20, 1), dt=0.001):
    global system
    system = NBodySystem(filename, ncells_per_dir=ncells_per_dir)
    pos = system.positions
    col = system.colors
    intensity = np.clip(system.masses / system.max_mass, 0.5, 1.0)
    visu = visualizer3d.Visualizer3D(
        pos,
        col,
        intensity,
        [[system.box[0][0], system.box[1][0]], [system.box[0][1], system.box[1][1]], [system.box[0][2], system.box[1][2]]],
    )
    visu.run(updater=update_positions, dt=dt)


def run_headless_benchmark(filename, ncells_per_dir=(20, 20, 1), dt=0.001, warmup_steps=2, steps=40):
    nbody = NBodySystem(filename, ncells_per_dir=ncells_per_dir)
    for _ in range(max(0, warmup_steps)):
        nbody.update_positions(dt)

    t0 = time.perf_counter()
    for _ in range(steps):
        nbody.update_positions(dt)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    per_step_ms = (elapsed / steps) * 1000.0
    print(f"Benchmark (numba): {steps} steps en {elapsed:.6f} s")
    print(f"Temps moyen par step: {per_step_ms:.4f} ms | Steps/s: {steps / elapsed:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simulation de galaxie en grille avec Numba")
    parser.add_argument("filename", nargs="?", default="data/galaxy_1000", help="Fichier de données")
    parser.add_argument("dt", nargs="?", type=float, default=0.001, help="Pas de temps")
    parser.add_argument("nx", nargs="?", type=int, default=20, help="Nombre de cellules en x")
    parser.add_argument("ny", nargs="?", type=int, default=20, help="Nombre de cellules en y")
    parser.add_argument("nz", nargs="?", type=int, default=1, help="Nombre de cellules en z")
    parser.add_argument("--threads", type=int, default=None, help="Nombre de threads Numba")
    parser.add_argument("--steps", type=int, default=40, help="Nombre de steps pour benchmark")
    parser.add_argument("--warmup", type=int, default=2, help="Nombre de steps warmup JIT")
    parser.add_argument("--no-display", action="store_true", help="Exécute en mode benchmark sans affichage")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.threads is not None:
        available_threads = get_num_threads()
        target_threads = max(1, min(args.threads, available_threads))
        if target_threads != args.threads:
            print(
                f"Threads demandés={args.threads} > limite numba={available_threads}; "
                f"utilisation de {target_threads}."
            )
        set_num_threads(target_threads)
    ncells = (args.nx, args.ny, args.nz)
    print(f"Simulation de {args.filename} avec dt = {args.dt} et grille {ncells}")
    print(f"Numba threads actifs: {get_num_threads()}")

    if args.no_display:
        run_headless_benchmark(
            filename=args.filename,
            ncells_per_dir=ncells,
            dt=args.dt,
            warmup_steps=args.warmup,
            steps=args.steps,
        )
    else:
        run_visual_simulation(args.filename, ncells_per_dir=ncells, dt=args.dt)


if __name__ == "__main__":
    main()
    
