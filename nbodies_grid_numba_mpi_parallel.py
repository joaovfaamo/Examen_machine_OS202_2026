import argparse
import time
import numpy as np
from mpi4py import MPI
from numba import njit, prange, set_num_threads, get_num_threads

from nbodies_grid_numba import G, update_stars_in_grid

NEAR_CHEB_RADIUS = 2


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


@njit
def _accumulate_local_cell_mass_moment(
    masses: np.ndarray,
    positions: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
):
    n_total_cells = np.prod(n_cells)
    cell_mass = np.zeros(n_total_cells, dtype=np.float32)
    cell_moment = np.zeros((n_total_cells, 3), dtype=np.float32)

    for i in range(positions.shape[0]):
        cell_idx = _cell_index_from_position(positions[i], grid_min, cell_size, n_cells)
        morse = _to_morse_index(cell_idx[0], cell_idx[1], cell_idx[2], n_cells)
        mass = masses[i]
        cell_mass[morse] += mass
        cell_moment[morse, 0] += positions[i, 0] * mass
        cell_moment[morse, 1] += positions[i, 1] * mass
        cell_moment[morse, 2] += positions[i, 2] * mass

    return cell_mass, cell_moment


@njit(parallel=True)
def compute_acceleration_owned_with_ghosts(
    owned_positions: np.ndarray,
    owned_global_ids: np.ndarray,
    available_positions: np.ndarray,
    available_masses: np.ndarray,
    available_global_ids: np.ndarray,
    available_cell_start_indices: np.ndarray,
    available_body_indices: np.ndarray,
    global_cell_masses: np.ndarray,
    global_cell_com_positions: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
):
    n_owned = owned_positions.shape[0]
    accelerations = np.zeros((n_owned, 3), dtype=np.float32)

    for iowned in prange(n_owned):
        pos = owned_positions[iowned]
        gid = owned_global_ids[iowned]
        cell_idx = _cell_index_from_position(pos, grid_min, cell_size, n_cells)

        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    morse = _to_morse_index(ix, iy, iz, n_cells)
                    is_far = (abs(ix - cell_idx[0]) > NEAR_CHEB_RADIUS) or (abs(iy - cell_idx[1]) > NEAR_CHEB_RADIUS) or (abs(iz - cell_idx[2]) > NEAR_CHEB_RADIUS)

                    if is_far:
                        cell_mass = global_cell_masses[morse]
                        if cell_mass > 0.0:
                            direction = global_cell_com_positions[morse] - pos
                            distance = np.sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
                            if distance > 1.0e-10:
                                inv_dist3 = 1.0 / (distance * distance * distance)
                                accelerations[iowned, :] += G * direction[:] * inv_dist3 * cell_mass
                    else:
                        start = available_cell_start_indices[morse]
                        end = available_cell_start_indices[morse + 1]
                        for j in range(start, end):
                            available_index = available_body_indices[j]
                            other_gid = available_global_ids[available_index]
                            if other_gid != gid:
                                direction = available_positions[available_index] - pos
                                distance = np.sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
                                if distance > 1.0e-10:
                                    inv_dist3 = 1.0 / (distance * distance * distance)
                                    accelerations[iowned, :] += G * direction[:] * inv_dist3 * available_masses[available_index]

    return accelerations


def parse_args():
    parser = argparse.ArgumentParser(description="Parallélisation MPI + Numba avec cellules fantômes (voisins uniquement)")
    parser.add_argument("filename", nargs="?", default="data/galaxy_1000", help="Fichier de données")
    parser.add_argument("dt", nargs="?", type=float, default=0.001, help="Pas de temps")
    parser.add_argument("nx", nargs="?", type=int, default=20, help="Nombre de cellules en x")
    parser.add_argument("ny", nargs="?", type=int, default=20, help="Nombre de cellules en y")
    parser.add_argument("nz", nargs="?", type=int, default=1, help="Nombre de cellules en z")
    parser.add_argument("--threads", type=int, default=None, help="Nombre de threads Numba par processus")
    parser.add_argument("--steps", type=int, default=30, help="Nombre de steps benchmark")
    parser.add_argument("--warmup", type=int, default=2, help="Nombre de steps warmup")
    return parser.parse_args()


def load_bodies(filename):
    masses = []
    positions = []
    velocities = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data = line.split()
            masses.append(float(data[0]))
            positions.append([float(data[1]), float(data[2]), float(data[3])])
            velocities.append([float(data[4]), float(data[5]), float(data[6])])
    return (
        np.array(masses, dtype=np.float32),
        np.array(positions, dtype=np.float32),
        np.array(velocities, dtype=np.float32),
    )


def build_slab_bounds(nx: int, n_ranks: int):
    starts = np.empty(n_ranks, dtype=np.int64)
    ends = np.empty(n_ranks, dtype=np.int64)
    owner_of_x = np.empty(nx, dtype=np.int64)

    base = nx // n_ranks
    rem = nx % n_ranks
    cursor = 0
    for rank in range(n_ranks):
        width = base + (1 if rank < rem else 0)
        if width > 0:
            starts[rank] = cursor
            ends[rank] = cursor + width - 1
            owner_of_x[cursor : cursor + width] = rank
        else:
            starts[rank] = -1
            ends[rank] = -2
        cursor += width

    return starts, ends, owner_of_x


def compute_cell_x(positions: np.ndarray, grid_min: np.ndarray, cell_size: np.ndarray, nx: int):
    if positions.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    cell_x = np.floor((positions[:, 0] - grid_min[0]) / cell_size[0]).astype(np.int64)
    return np.clip(cell_x, 0, nx - 1)


def _sendrecv_rows(comm, rank, size, send_left: np.ndarray, send_right: np.ndarray, ncols: int):
    recv_from_left = np.empty((0, ncols), dtype=np.float64)
    recv_from_right = np.empty((0, ncols), dtype=np.float64)

    left = rank - 1
    right = rank + 1
    count_tag = 100
    data_tag = 101

    if left >= 0:
        recv_count = comm.sendrecv(sendobj=send_left.shape[0], dest=left, sendtag=count_tag, source=left, recvtag=count_tag)
        recv_from_left = np.empty((recv_count, ncols), dtype=np.float64)
        comm.Sendrecv(
            sendbuf=send_left.reshape(-1),
            dest=left,
            sendtag=data_tag,
            recvbuf=recv_from_left.reshape(-1),
            source=left,
            recvtag=data_tag,
        )

    if right < size:
        recv_count = comm.sendrecv(sendobj=send_right.shape[0], dest=right, sendtag=count_tag, source=right, recvtag=count_tag)
        recv_from_right = np.empty((recv_count, ncols), dtype=np.float64)
        comm.Sendrecv(
            sendbuf=send_right.reshape(-1),
            dest=right,
            sendtag=data_tag,
            recvbuf=recv_from_right.reshape(-1),
            source=right,
            recvtag=data_tag,
        )

    return recv_from_left, recv_from_right


def migrate_owned_by_neighbors(
    comm,
    rank,
    size,
    local_masses,
    local_positions,
    local_velocities,
    local_global_ids,
    local_a_old,
    x_start,
    x_end,
    grid_min,
    cell_size,
    nx,
):
    if local_positions.shape[0] == 0:
        rows = np.empty((0, 11), dtype=np.float64)
    else:
        rows = np.empty((local_positions.shape[0], 11), dtype=np.float64)
        rows[:, 0] = local_masses.astype(np.float64)
        rows[:, 1:4] = local_positions.astype(np.float64)
        rows[:, 4:7] = local_velocities.astype(np.float64)
        rows[:, 7:10] = local_a_old.astype(np.float64)
        rows[:, 10] = local_global_ids.astype(np.float64)

    while True:
        if rows.shape[0] == 0:
            cell_x = np.empty(0, dtype=np.int64)
        else:
            cell_x = compute_cell_x(rows[:, 1:4].astype(np.float32), grid_min, cell_size, nx)

        left_mask = cell_x < x_start
        right_mask = cell_x > x_end
        stay_mask = ~(left_mask | right_mask)

        send_left = rows[left_mask]
        send_right = rows[right_mask]
        stay_rows = rows[stay_mask]

        local_out = int(send_left.shape[0] + send_right.shape[0])
        global_out = comm.allreduce(local_out, op=MPI.SUM)

        recv_left, recv_right = _sendrecv_rows(comm, rank, size, send_left, send_right, 11)

        if recv_left.shape[0] + recv_right.shape[0] > 0:
            rows = np.vstack([stay_rows, recv_left, recv_right])
        else:
            rows = stay_rows

        if global_out == 0:
            break

    if rows.shape[0] == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float32),
        )

    return (
        rows[:, 0].astype(np.float32),
        rows[:, 1:4].astype(np.float32),
        rows[:, 4:7].astype(np.float32),
        rows[:, 10].astype(np.int64),
        rows[:, 7:10].astype(np.float32),
    )


def exchange_ghost_layers(
    comm,
    rank,
    size,
    local_masses,
    local_positions,
    local_global_ids,
    x_start,
    x_end,
    grid_min,
    cell_size,
    nx,
):
    if local_positions.shape[0] == 0:
        cell_x = np.empty(0, dtype=np.int64)
    else:
        cell_x = compute_cell_x(local_positions, grid_min, cell_size, nx)

    left_boundary = cell_x <= (x_start + NEAR_CHEB_RADIUS - 1)
    right_boundary = cell_x >= (x_end - NEAR_CHEB_RADIUS + 1)

    send_left = np.empty((np.sum(left_boundary), 5), dtype=np.float64)
    send_right = np.empty((np.sum(right_boundary), 5), dtype=np.float64)

    if send_left.shape[0] > 0:
        send_left[:, 0] = local_masses[left_boundary].astype(np.float64)
        send_left[:, 1:4] = local_positions[left_boundary].astype(np.float64)
        send_left[:, 4] = local_global_ids[left_boundary].astype(np.float64)

    if send_right.shape[0] > 0:
        send_right[:, 0] = local_masses[right_boundary].astype(np.float64)
        send_right[:, 1:4] = local_positions[right_boundary].astype(np.float64)
        send_right[:, 4] = local_global_ids[right_boundary].astype(np.float64)

    recv_left, recv_right = _sendrecv_rows(comm, rank, size, send_left, send_right, 5)

    if recv_left.shape[0] + recv_right.shape[0] == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    ghost_rows = np.vstack([recv_left, recv_right]) if recv_left.shape[0] > 0 and recv_right.shape[0] > 0 else (recv_left if recv_right.shape[0] == 0 else recv_right)
    return (
        ghost_rows[:, 0].astype(np.float32),
        ghost_rows[:, 1:4].astype(np.float32),
        ghost_rows[:, 4].astype(np.int64),
    )


def compute_global_cell_mass_and_com(comm, local_masses, local_positions, grid_min, cell_size, n_cells):
    local_cell_mass, local_cell_moment = _accumulate_local_cell_mass_moment(local_masses, local_positions, grid_min, cell_size, n_cells)

    global_cell_mass = np.empty_like(local_cell_mass)
    global_cell_moment = np.empty_like(local_cell_moment)
    comm.Allreduce(local_cell_mass, global_cell_mass, op=MPI.SUM)
    comm.Allreduce(local_cell_moment, global_cell_moment, op=MPI.SUM)

    global_cell_com = np.zeros_like(global_cell_moment)
    non_zero = global_cell_mass > 0.0
    global_cell_com[non_zero, :] = global_cell_moment[non_zero, :] / global_cell_mass[non_zero][:, None]
    return global_cell_mass, global_cell_com


def build_available_cell_index(available_masses, available_positions, grid_min, cell_size, n_cells):
    n_total_cells = int(np.prod(n_cells))
    available_cell_start = np.full(n_total_cells + 1, -1, dtype=np.int64)
    available_body_indices = np.empty(available_positions.shape[0], dtype=np.int64)
    available_cell_masses = np.zeros(n_total_cells, dtype=np.float32)
    available_cell_com = np.zeros((n_total_cells, 3), dtype=np.float32)

    update_stars_in_grid(
        available_cell_start,
        available_body_indices,
        available_cell_masses,
        available_cell_com,
        available_masses,
        available_positions,
        grid_min,
        cell_size,
        n_cells,
    )

    return available_cell_start, available_body_indices


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.threads is not None:
        available_threads = get_num_threads()
        target_threads = max(1, min(args.threads, available_threads))
        if target_threads != args.threads and rank == 0:
            print(
                f"Threads demandés={args.threads} > limite MPI/numba={available_threads}; "
                f"utilisation de {target_threads}."
            )
        set_num_threads(target_threads)

    masses_all = positions_all = velocities_all = None
    if rank == 0:
        masses_all, positions_all, velocities_all = load_bodies(args.filename)

    masses_all = comm.bcast(masses_all, root=0)
    positions_all = comm.bcast(positions_all, root=0)
    velocities_all = comm.bcast(velocities_all, root=0)

    n_cells = np.array([args.nx, args.ny, args.nz], dtype=np.int64)
    grid_min = np.min(positions_all, axis=0) - 1.0e-6
    grid_max = np.max(positions_all, axis=0) + 1.0e-6
    cell_size = (grid_max - grid_min) / n_cells

    n_bodies = positions_all.shape[0]
    starts, ends, owner_of_x = build_slab_bounds(args.nx, size)
    x_start = starts[rank]
    x_end = ends[rank]

    all_gids = np.arange(n_bodies, dtype=np.int64)
    owners = owner_of_x[compute_cell_x(positions_all, grid_min, cell_size, args.nx)]
    local_mask = owners == rank

    local_masses = masses_all[local_mask].copy()
    local_positions = positions_all[local_mask].copy()
    local_velocities = velocities_all[local_mask].copy()
    local_global_ids = all_gids[local_mask].copy()

    if rank == 0:
        print(
            f"MPI ghost cells (voisins): {size} processus, {get_num_threads()} threads/process, "
            f"N={n_bodies}, grille=({args.nx},{args.ny},{args.nz})"
        )

    def one_step(local_masses_, local_positions_, local_velocities_, local_global_ids_):
        global_cell_mass_old, global_cell_com_old = compute_global_cell_mass_and_com(
            comm, local_masses_, local_positions_, grid_min, cell_size, n_cells
        )

        ghost_masses, ghost_positions, ghost_gids = exchange_ghost_layers(
            comm,
            rank,
            size,
            local_masses_,
            local_positions_,
            local_global_ids_,
            x_start,
            x_end,
            grid_min,
            cell_size,
            args.nx,
        )

        if ghost_positions.shape[0] > 0:
            available_masses = np.concatenate([local_masses_, ghost_masses])
            available_positions = np.vstack([local_positions_, ghost_positions])
            available_gids = np.concatenate([local_global_ids_, ghost_gids])
        else:
            available_masses = local_masses_
            available_positions = local_positions_
            available_gids = local_global_ids_

        available_cell_start, available_body_indices = build_available_cell_index(
            available_masses,
            available_positions,
            grid_min,
            cell_size,
            n_cells,
        )

        a_old = compute_acceleration_owned_with_ghosts(
            local_positions_,
            local_global_ids_,
            available_positions,
            available_masses,
            available_gids,
            available_cell_start,
            available_body_indices,
            global_cell_mass_old,
            global_cell_com_old,
            grid_min,
            cell_size,
            n_cells,
        )

        local_positions_new = local_positions_ + local_velocities_ * args.dt + 0.5 * a_old * args.dt * args.dt

        (
            local_masses_migr,
            local_positions_migr,
            local_velocities_migr,
            local_global_ids_migr,
            local_a_old_migr,
        ) = migrate_owned_by_neighbors(
            comm,
            rank,
            size,
            local_masses_,
            local_positions_new,
            local_velocities_,
            local_global_ids_,
            a_old,
            x_start,
            x_end,
            grid_min,
            cell_size,
            args.nx,
        )

        global_cell_mass_new, global_cell_com_new = compute_global_cell_mass_and_com(
            comm, local_masses_migr, local_positions_migr, grid_min, cell_size, n_cells
        )

        ghost_masses_new, ghost_positions_new, ghost_gids_new = exchange_ghost_layers(
            comm,
            rank,
            size,
            local_masses_migr,
            local_positions_migr,
            local_global_ids_migr,
            x_start,
            x_end,
            grid_min,
            cell_size,
            args.nx,
        )

        if ghost_positions_new.shape[0] > 0:
            available_masses_new = np.concatenate([local_masses_migr, ghost_masses_new])
            available_positions_new = np.vstack([local_positions_migr, ghost_positions_new])
            available_gids_new = np.concatenate([local_global_ids_migr, ghost_gids_new])
        else:
            available_masses_new = local_masses_migr
            available_positions_new = local_positions_migr
            available_gids_new = local_global_ids_migr

        available_cell_start_new, available_body_indices_new = build_available_cell_index(
            available_masses_new,
            available_positions_new,
            grid_min,
            cell_size,
            n_cells,
        )

        a_new = compute_acceleration_owned_with_ghosts(
            local_positions_migr,
            local_global_ids_migr,
            available_positions_new,
            available_masses_new,
            available_gids_new,
            available_cell_start_new,
            available_body_indices_new,
            global_cell_mass_new,
            global_cell_com_new,
            grid_min,
            cell_size,
            n_cells,
        )

        local_velocities_new = local_velocities_migr + 0.5 * (local_a_old_migr + a_new) * args.dt

        return local_masses_migr, local_positions_migr, local_velocities_new, local_global_ids_migr

    comm.Barrier()
    for _ in range(max(0, args.warmup)):
        local_masses, local_positions, local_velocities, local_global_ids = one_step(
            local_masses,
            local_positions,
            local_velocities,
            local_global_ids,
        )

    comm.Barrier()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        local_masses, local_positions, local_velocities, local_global_ids = one_step(
            local_masses,
            local_positions,
            local_velocities,
            local_global_ids,
        )
    comm.Barrier()
    t1 = time.perf_counter()

    local_count = np.array([local_positions.shape[0]], dtype=np.int64)
    global_count = np.array([0], dtype=np.int64)
    comm.Allreduce(local_count, global_count, op=MPI.SUM)

    if global_count[0] != n_bodies and rank == 0:
        raise RuntimeError(f"Incohérence du nombre total de corps après migration: {global_count[0]} != {n_bodies}")

    elapsed_local = t1 - t0
    elapsed_max = comm.reduce(elapsed_local, op=MPI.MAX, root=0)

    if rank == 0:
        print(f"Benchmark (MPI calcul ghost): {args.steps} steps en {elapsed_max:.6f} s")
        print(f"Temps moyen par step: {(elapsed_max / args.steps) * 1000.0:.4f} ms | Steps/s: {args.steps / elapsed_max:.3f}")


if __name__ == "__main__":
    main()
