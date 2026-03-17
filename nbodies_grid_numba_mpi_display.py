import argparse
import time
import numpy as np
from mpi4py import MPI
from numba import set_num_threads, get_num_threads

import visualizer3d
from nbodies_grid_numba import NBodySystem

CMD_TAG = 10
POS_TAG = 11
CMD_STEP = 1
CMD_STOP = 0


def parse_args():
    parser = argparse.ArgumentParser(description="MPI: séparation affichage (rank 0) / calcul (rank 1)")
    parser.add_argument("filename", nargs="?", default="data/galaxy_1000", help="Fichier de données")
    parser.add_argument("dt", nargs="?", type=float, default=0.001, help="Pas de temps")
    parser.add_argument("nx", nargs="?", type=int, default=20, help="Nombre de cellules en x")
    parser.add_argument("ny", nargs="?", type=int, default=20, help="Nombre de cellules en y")
    parser.add_argument("nz", nargs="?", type=int, default=1, help="Nombre de cellules en z")
    parser.add_argument("--threads", type=int, default=None, help="Nombre de threads Numba (côté calcul)")
    parser.add_argument("--steps", type=int, default=40, help="Nombre de steps pour benchmark headless")
    parser.add_argument("--warmup", type=int, default=2, help="Nombre de steps warmup")
    parser.add_argument("--no-display", action="store_true", help="Mode benchmark sans GUI")
    return parser.parse_args()


def run_rank1_compute_loop(comm, system, dt):
    cmd = np.empty(1, dtype=np.int32)
    while True:
        comm.Recv(cmd, source=0, tag=CMD_TAG)
        if cmd[0] == CMD_STOP:
            break
        system.update_positions(dt)
        comm.Send(np.ascontiguousarray(system.positions), dest=0, tag=POS_TAG)


def run_rank0_visual(comm, system, dt):
    pos_buffer = np.empty_like(system.positions)

    def updater(_dt):
        cmd = np.array([CMD_STEP], dtype=np.int32)
        comm.Send(cmd, dest=1, tag=CMD_TAG)
        comm.Recv(pos_buffer, source=1, tag=POS_TAG)
        return pos_buffer

    pos = system.positions
    col = system.colors
    intensity = np.clip(system.masses / system.max_mass, 0.5, 1.0)
    visu = visualizer3d.Visualizer3D(
        pos,
        col,
        intensity,
        [[system.box[0][0], system.box[1][0]], [system.box[0][1], system.box[1][1]], [system.box[0][2], system.box[1][2]]],
    )
    visu.run(updater=updater, dt=dt)

    cmd = np.array([CMD_STOP], dtype=np.int32)
    comm.Send(cmd, dest=1, tag=CMD_TAG)


def run_rank0_headless_benchmark(comm, system, dt, warmup, steps):
    pos_buffer = np.empty_like(system.positions)
    cmd_step = np.array([CMD_STEP], dtype=np.int32)
    cmd_stop = np.array([CMD_STOP], dtype=np.int32)

    for _ in range(max(0, warmup)):
        comm.Send(cmd_step, dest=1, tag=CMD_TAG)
        comm.Recv(pos_buffer, source=1, tag=POS_TAG)

    t0 = time.perf_counter()
    for _ in range(steps):
        comm.Send(cmd_step, dest=1, tag=CMD_TAG)
        comm.Recv(pos_buffer, source=1, tag=POS_TAG)
    t1 = time.perf_counter()

    comm.Send(cmd_stop, dest=1, tag=CMD_TAG)

    elapsed = t1 - t0
    print(f"Benchmark (MPI display/calc): {steps} steps en {elapsed:.6f} s")
    print(f"Temps moyen par step: {(elapsed / steps) * 1000.0:.4f} ms | Steps/s: {steps / elapsed:.3f}")


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        if rank == 0:
            print("Erreur: lancer avec au moins 2 processus MPI (rank 0=affichage, rank 1=calcul).")
        return

    if args.threads is not None:
        available_threads = get_num_threads()
        target_threads = max(1, min(args.threads, available_threads))
        if target_threads != args.threads and rank == 0:
            print(
                f"Threads demandés={args.threads} > limite MPI/numba={available_threads}; "
                f"utilisation de {target_threads}."
            )
        set_num_threads(target_threads)

    ncells = (args.nx, args.ny, args.nz)

    if rank in (0, 1):
        system = NBodySystem(args.filename, ncells_per_dir=ncells)
    else:
        system = None

    if rank == 0:
        print(f"MPI séparation affichage/calcul | threads numba (rank 0): {get_num_threads()}")
    if rank == 1:
        print(f"Rank 1 calcul | threads numba actifs: {get_num_threads()}")

    comm.Barrier()

    if rank == 0:
        if args.no_display:
            run_rank0_headless_benchmark(comm, system, args.dt, args.warmup, args.steps)
        else:
            run_rank0_visual(comm, system, args.dt)
    elif rank == 1:
        run_rank1_compute_loop(comm, system, args.dt)

    comm.Barrier()


if __name__ == "__main__":
    main()
