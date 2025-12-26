#!/usr/bin/env python3
"""
Compute 1D RMSD Free Energy Surface from GaMD trajectory
(using cumulant expansion reweighting).

Author: Sanjay Saravanan
"""

import argparse
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt


# -------------------------------
# Argument parser
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute 1D GaMD reweighted RMSD free energy surface"
    )

    parser.add_argument("-t", "--traj", required=True, help="Trajectory file (DCD/XTC)")
    parser.add_argument("-p", "--top", required=True, help="Topology file (PDB/PRMTOP)")
    parser.add_argument("-g", "--gamd-log", required=True, help="GaMD log file")
    parser.add_argument("-o", "--out", default="gamd_1d_fes.png", help="Output plot name")

    parser.add_argument("--start", type=int, default=0,
                        help="Starting frame index for analysis")
    parser.add_argument("--bins", type=int, default=15,
                        help="Number of RMSD bins (recommended 10–15)")
    parser.add_argument("--min-frames", type=int, default=50,
                        help="Minimum frames per bin")

    parser.add_argument("--temperature", type=float, default=300.0,
                        help="Simulation temperature (K)")

    return parser.parse_args()


# -------------------------------
# Main workflow
# -------------------------------
def main():
    args = parse_args()

    # Constants
    kB = 0.001987  # kcal/mol/K
    beta = 1.0 / (kB * args.temperature)

    print("Loading trajectory...")
    traj = md.load(args.traj, top=args.top)
    traj = traj[args.start:]

    traj.image_molecules(inplace=True)

    # CA alignment
    ca_indices = traj.topology.select("name CA")
    traj.superpose(traj, frame=0, atom_indices=ca_indices)

    print("Computing RMSD...")
    rmsd = md.rmsd(traj, traj, frame=0, atom_indices=ca_indices)

    print("Loading GaMD log...")
    gamd = np.loadtxt(args.gamd_log, comments=("#", "@"))
    gamd = gamd[args.start:]

    # Adjust column indices if needed
    del_v_total = gamd[:, 7]
    del_v_dihedral = gamd[:, 8]
    del_v_dual = del_v_total + del_v_dihedral

    # -------------------------------
    # Bin RMSD
    # -------------------------------
    bins = np.linspace(rmsd.min(), rmsd.max(), args.bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    digitized = np.digitize(rmsd, bins) - 1

    P_star = np.zeros(len(bin_centers))
    weights = np.zeros(len(bin_centers))

    print("Reweighting bins...")
    for i in range(len(bin_centers)):
        idx = np.where(digitized == i)[0]

        if len(idx) < args.min_frames:
            P_star[i] = np.nan
            weights[i] = np.nan
            continue

        P_star[i] = len(idx)

        mu = del_v_dual[idx].mean()
        var = del_v_dual[idx].var()

        weights[i] = np.exp(beta * mu + 0.5 * beta**2 * var)

    # Normalize probabilities
    P_star /= np.nansum(P_star)

    P = P_star * weights
    P /= np.nansum(P)

    # -------------------------------
    # Free energies
    # -------------------------------
    F = -kB * args.temperature * np.log(P)
    F -= np.nanmin(F)

    F_biased = -kB * args.temperature * np.log(P_star)
    F_biased -= np.nanmin(F_biased)

    valid = np.isfinite(F)

    # -------------------------------
    # Plot
    # -------------------------------
    print("Saving plot:", args.out)

    plt.figure(figsize=(6, 4))

    plt.plot(
        bin_centers[valid] * 10.0,   # nm → Å
        F[valid],
        lw=2,
        label="GaMD reweighted"
    )

    plt.plot(
        bin_centers[valid] * 10.0,
        F_biased[valid],
        lw=2,
        linestyle="--",
        label="Biased"
    )

    plt.xlabel("RMSD (Å)", fontsize=12, fontweight="bold")
    plt.ylabel("Free Energy (kcal/mol)", fontsize=12, fontweight="bold")
    plt.title("1D Free Energy Surface (RMSD)", fontsize=14, fontweight="bold")

    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()