import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_gene_names() -> Tuple[List[str], Dict[str, List[int]]]:
    names = []
    groups: Dict[str, List[int]] = {}

    def add_group(prefix: str, count: int) -> List[int]:
        start = len(names)
        idxs = []
        for i in range(1, count + 1):
            names.append(f"{prefix}{i}")
            idxs.append(start + i - 1)
        groups[prefix.rstrip("_").lower()] = idxs
        return idxs

    groups["prog"] = add_group("TF_PROG_", 4)
    groups["lineage"] = add_group("TF_LIN", 12)
    groups["ligand"] = add_group("LIG_", 20)
    groups["receptor"] = add_group("REC_", 20)
    groups["target"] = add_group("TARG_", 80)
    groups["housekeeping"] = add_group("HK_", 50)
    groups["other"] = add_group("OTHER_", 14)

    return names, groups


def build_grn(
    rng: np.random.Generator,
    gene_names: List[str],
    groups: Dict[str, List[int]],
    n_lineages: int,
    lineage_tfs_per: int,
) -> Dict[str, np.ndarray]:
    n_genes = len(gene_names)
    rows = []
    cols = []
    data = []
    edge_list = []

    def add_edge(src: int, dst: int, weight: float, label: str) -> None:
        rows.append(dst)
        cols.append(src)
        data.append(weight)
        edge_list.append(
            {
                "source": gene_names[src],
                "target": gene_names[dst],
                "weight": float(weight),
                "type": label,
            }
        )

    prog = groups["prog"]
    lineage = groups["lineage"]
    ligands = groups["ligand"]
    receptors = groups["receptor"]
    targets = groups["target"]

    for i in prog:
        add_edge(i, i, 0.2, "prog_self")
        for j in prog:
            if i == j:
                continue
            add_edge(j, i, 0.35 + 0.1 * rng.random(), "prog_mutual")

    for lin_idx in lineage:
        for prog_idx in prog:
            add_edge(prog_idx, lin_idx, 0.15, "prog_to_lineage")

    for lin_block in range(n_lineages):
        tf_indices = lineage[
            lin_block * lineage_tfs_per : (lin_block + 1) * lineage_tfs_per
        ]
        for tf in tf_indices:
            add_edge(tf, tf, 0.6, "lineage_self")
        if len(tf_indices) == 2:
            add_edge(tf_indices[0], tf_indices[1], 0.4, "lineage_partner")
            add_edge(tf_indices[1], tf_indices[0], 0.4, "lineage_partner")

        for tf in tf_indices:
            for other_block in range(n_lineages):
                if other_block == lin_block:
                    continue
                other_tfs = lineage[
                    other_block * lineage_tfs_per : (other_block + 1) * lineage_tfs_per
                ]
                for other_tf in other_tfs:
                    add_edge(other_tf, tf, -0.5, "lineage_inhibit")

        for tf in tf_indices:
            for prog_idx in prog:
                add_edge(tf, prog_idx, -0.3, "lineage_repress_prog")

    target_splits = np.array_split(targets, n_lineages)
    for lin_block, target_subset in enumerate(target_splits):
        tf_indices = lineage[
            lin_block * lineage_tfs_per : (lin_block + 1) * lineage_tfs_per
        ]
        for tf in tf_indices:
            for tgt in target_subset:
                add_edge(tf, tgt, 0.7, "lineage_to_target")

    ligand_splits = np.array_split(ligands, n_lineages)
    for lin_block, ligand_subset in enumerate(ligand_splits):
        tf_indices = lineage[
            lin_block * lineage_tfs_per : (lin_block + 1) * lineage_tfs_per
        ]
        for tf in tf_indices:
            for lig in ligand_subset:
                add_edge(tf, lig, 0.6, "lineage_to_ligand")

    for prog_idx in prog:
        for rec in receptors:
            add_edge(prog_idx, rec, 0.2, "prog_to_receptor")

    for _ in range(200):
        src = rng.integers(0, n_genes)
        dst = rng.integers(0, n_genes)
        if src == dst:
            continue
        add_edge(src, dst, rng.uniform(-0.05, 0.05), "random")

    W = sp.csr_matrix((data, (rows, cols)), shape=(n_genes, n_genes))

    bias = np.zeros(n_genes, dtype=np.float32)
    decay = np.zeros(n_genes, dtype=np.float32)
    noise = np.zeros(n_genes, dtype=np.float32)

    bias[groups["housekeeping"]] = 0.6
    bias[groups["ligand"]] = 0.2
    bias[groups["receptor"]] = 0.2
    bias[groups["target"]] = 0.05
    bias[groups["prog"]] = 0.05
    bias[groups["lineage"]] = 0.05
    bias[groups["other"]] = 0.1

    decay[groups["housekeeping"]] = 0.1
    decay[groups["ligand"]] = 0.2
    decay[groups["receptor"]] = 0.2
    decay[groups["target"]] = 0.2
    decay[groups["prog"]] = 0.3
    decay[groups["lineage"]] = 0.3
    decay[groups["other"]] = 0.2

    noise[groups["housekeeping"]] = 0.05
    noise[groups["ligand"]] = 0.06
    noise[groups["receptor"]] = 0.06
    noise[groups["target"]] = 0.08
    noise[groups["prog"]] = 0.1
    noise[groups["lineage"]] = 0.1
    noise[groups["other"]] = 0.08

    return {
        "W": W,
        "bias": bias,
        "decay": decay,
        "noise": noise,
        "edges": edge_list,
    }


def build_ligand_receptor_effects(
    gene_names: List[str],
    groups: Dict[str, List[int]],
    n_lineages: int,
    lineage_tfs_per: int,
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    ligands = groups["ligand"]
    receptors = groups["receptor"]
    lineage = groups["lineage"]
    n_pairs = min(len(ligands), len(receptors))

    pair_map = []
    for idx in range(n_pairs):
        pair_map.append(
            {
                "pair_id": idx,
                "ligand": ligands[idx],
                "receptor": receptors[idx],
            }
        )

    pair_to_lineage = np.arange(n_pairs) % n_lineages
    B = np.zeros((len(gene_names), n_pairs), dtype=np.float32)
    for pair_id, lin_idx in enumerate(pair_to_lineage):
        tf_indices = lineage[
            lin_idx * lineage_tfs_per : (lin_idx + 1) * lineage_tfs_per
        ]
        for tf in tf_indices:
            B[tf, pair_id] = 0.6

    return B, pair_map


def hill(x: np.ndarray, k: float, n: float) -> np.ndarray:
    x_pos = np.maximum(x, 0.0)
    x_pow = np.power(x_pos, n)
    k_pow = k**n
    return x_pow / (k_pow + x_pow + 1e-8)


def simulate_sample(
    rng: np.random.Generator,
    n_cells: int,
    timepoints: np.ndarray,
    dt: float,
    steps_per_tp: int,
    grn: Dict[str, np.ndarray],
    ligand_effects: np.ndarray,
    groups: Dict[str, List[int]],
    morphogen: np.ndarray,
    x_max: float,
    hill_k: float,
    hill_n: float,
    neighbor_mode: str,
    neighbor_k: int,
    neighbor_mix: float,
) -> np.ndarray:
    n_genes = len(grn["bias"])
    W = grn["W"].transpose().toarray()
    bias = grn["bias"]
    decay = grn["decay"]
    noise = grn["noise"]

    x = np.zeros((n_cells, n_genes), dtype=np.float32)
    x[:, groups["prog"]] = rng.normal(1.5, 0.1, size=(n_cells, len(groups["prog"])))
    x[:, groups["lineage"]] = rng.normal(
        0.1, 0.05, size=(n_cells, len(groups["lineage"]))
    )
    x[:, groups["ligand"]] = rng.normal(
        0.2, 0.05, size=(n_cells, len(groups["ligand"]))
    )
    x[:, groups["receptor"]] = rng.normal(
        0.2, 0.05, size=(n_cells, len(groups["receptor"]))
    )
    x[:, groups["target"]] = rng.normal(
        0.1, 0.05, size=(n_cells, len(groups["target"]))
    )
    x[:, groups["housekeeping"]] = rng.normal(
        1.0, 0.1, size=(n_cells, len(groups["housekeeping"]))
    )
    x[:, groups["other"]] = rng.normal(
        0.1, 0.05, size=(n_cells, len(groups["other"]))
    )
    x = np.clip(x, 0.0, x_max)

    n_timepoints = len(timepoints)
    snapshots = np.zeros((n_cells, n_timepoints, n_genes), dtype=np.float32)
    snapshots[:, 0, :] = x

    total_steps = (n_timepoints - 1) * steps_per_tp
    snapshot_idx = 1
    lig_idx = groups["ligand"]
    rec_idx = groups["receptor"]

    neighbor_idx = None
    if neighbor_mode == "random":
        if neighbor_k <= 0 or n_cells <= 1:
            neighbor_idx = None
        else:
            k = min(neighbor_k, n_cells - 1)
            neighbor_idx = np.empty((n_cells, k), dtype=np.int64)
            for i in range(n_cells):
                choices = rng.choice(n_cells - 1, size=k, replace=False)
                choices = np.where(choices >= i, choices + 1, choices)
                neighbor_idx[i] = choices

    for step in range(total_steps):
        h = hill(x, hill_k, hill_n)
        f = h @ W + bias - decay * x

        if neighbor_idx is None:
            ligand_mean = np.maximum(x[:, lig_idx], 0.0).mean(axis=0)
        else:
            neighbor_lig = np.maximum(x[neighbor_idx][:, :, lig_idx], 0.0).mean(axis=1)
            global_lig = np.maximum(x[:, lig_idx], 0.0).mean(axis=0)
            ligand_mean = neighbor_mix * neighbor_lig + (1.0 - neighbor_mix) * global_lig
        receptor_act = hill(x[:, rec_idx], hill_k, hill_n)
        signal = receptor_act * ligand_mean
        g_signal = signal.dot(ligand_effects.T)

        deriv = f + g_signal + morphogen
        noise_term = rng.normal(0.0, noise, size=x.shape) * np.sqrt(dt)
        x = x + dt * deriv + noise_term
        x = np.clip(x, 0.0, x_max)

        if (step + 1) % steps_per_tp == 0:
            snapshots[:, snapshot_idx, :] = x
            snapshot_idx += 1

    return snapshots


def latent_to_counts(
    rng: np.random.Generator,
    x: np.ndarray,
    dispersion: np.ndarray,
    dropout_mid: float,
    dropout_scale: float,
) -> np.ndarray:
    mu = np.log1p(np.exp(x))
    shape = dispersion
    scale = mu / dispersion
    lam = rng.gamma(shape, scale)
    counts = rng.poisson(lam).astype(np.int32)

    drop_prob = 1.0 / (1.0 + np.exp((mu - dropout_mid) / dropout_scale))
    drop_mask = rng.random(mu.shape) < drop_prob
    counts[drop_mask] = 0
    return counts


def assign_lineage(
    x: np.ndarray,
    lineage_indices: List[int],
    n_lineages: int,
    lineage_tfs_per: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.zeros((x.shape[0], n_lineages), dtype=np.float32)
    for lin_idx in range(n_lineages):
        tf_indices = lineage_indices[
            lin_idx * lineage_tfs_per : (lin_idx + 1) * lineage_tfs_per
        ]
        scores[:, lin_idx] = np.mean(x[:, tf_indices], axis=1)

    max_scores = scores.max(axis=1)
    lineage_id = np.argmax(scores, axis=1) + 1
    lineage_id[max_scores < threshold] = 0

    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    return lineage_id.astype(np.int32), probs.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate GRN time-series dataset.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-cells", type=int, default=500000)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--n-timepoints", type=int, default=12)
    parser.add_argument("--hours-per-timepoint", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--branch-timepoint-index", type=int, default=4)
    parser.add_argument("--holdout-lineage", type=int, default=6)
    parser.add_argument("--lineage-threshold", type=float, default=1.2)
    parser.add_argument("--dropout-mid", type=float, default=1.0)
    parser.add_argument("--dropout-scale", type=float, default=0.6)
    parser.add_argument("--x-max", type=float, default=6.0)
    parser.add_argument("--hill-k", type=float, default=1.0)
    parser.add_argument("--hill-n", type=float, default=2.0)
    parser.add_argument("--morphogen-scale", type=float, default=0.6)
    parser.add_argument("--neighbor-mode", choices=["none", "random"], default="none")
    parser.add_argument("--neighbor-k", type=int, default=10)
    parser.add_argument("--neighbor-mix", type=float, default=1.0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    gene_names, groups = build_gene_names()
    n_genes = len(gene_names)
    n_lineages = 6
    lineage_tfs_per = 2

    grn = build_grn(rng, gene_names, groups, n_lineages, lineage_tfs_per)
    ligand_effects, pair_map = build_ligand_receptor_effects(
        gene_names, groups, n_lineages, lineage_tfs_per
    )

    total_trajectories = args.total_cells // args.n_timepoints
    base_traj = total_trajectories // args.n_samples
    remainder = total_trajectories % args.n_samples
    traj_per_sample = [
        base_traj + (1 if i < remainder else 0) for i in range(args.n_samples)
    ]
    actual_total_cells = sum(traj_per_sample) * args.n_timepoints

    timepoints = np.arange(
        0,
        args.n_timepoints * args.hours_per_timepoint,
        args.hours_per_timepoint,
    )
    steps_per_tp = int(args.hours_per_timepoint / args.dt)
    if abs(steps_per_tp * args.dt - args.hours_per_timepoint) > 1e-6:
        raise ValueError("hours-per-timepoint must be divisible by dt.")

    dispersion = rng.lognormal(mean=1.0, sigma=0.3, size=n_genes).astype(np.float32)

    X_blocks = []
    obs_records = []
    fate_blocks = []

    src_indices = []
    tgt_indices = []
    weights = []

    offset = 0
    for sample_idx in range(args.n_samples):
        n_cells = traj_per_sample[sample_idx]
        morphogen = np.zeros(n_genes, dtype=np.float32)
        lineage_bias = rng.dirichlet(alpha=np.ones(n_lineages))
        for lin_idx in range(n_lineages):
            tf_indices = groups["lineage"][
                lin_idx * lineage_tfs_per : (lin_idx + 1) * lineage_tfs_per
            ]
            morphogen[tf_indices] = args.morphogen_scale * lineage_bias[lin_idx]

        snapshots = simulate_sample(
            rng=rng,
            n_cells=n_cells,
            timepoints=timepoints,
            dt=args.dt,
            steps_per_tp=steps_per_tp,
            grn=grn,
            ligand_effects=ligand_effects,
            groups=groups,
            morphogen=morphogen,
            x_max=args.x_max,
            hill_k=args.hill_k,
            hill_n=args.hill_n,
            neighbor_mode=args.neighbor_mode,
            neighbor_k=args.neighbor_k,
            neighbor_mix=args.neighbor_mix,
        )

        snapshot_flat = snapshots.reshape(n_cells * args.n_timepoints, n_genes)
        counts = latent_to_counts(
            rng=rng,
            x=snapshot_flat,
            dispersion=dispersion,
            dropout_mid=args.dropout_mid,
            dropout_scale=args.dropout_scale,
        )
        X_blocks.append(sp.csr_matrix(counts))

        lineage_id, fate_probs = assign_lineage(
            snapshot_flat,
            groups["lineage"],
            n_lineages,
            lineage_tfs_per,
            args.lineage_threshold,
        )
        fate_blocks.append(fate_probs)

        traj_ids = np.repeat(np.arange(n_cells), args.n_timepoints)
        time_idx = np.tile(np.arange(args.n_timepoints), n_cells)
        time_values = timepoints[time_idx]

        cell_types = np.where(
            lineage_id == 0,
            "progenitor",
            np.array([f"lin{idx}" for idx in lineage_id]),
        )

        obs_records.append(
            pd.DataFrame(
                {
                    "sample": np.full_like(time_idx, f"sample_{sample_idx}", dtype=object),
                    "timepoint": time_values,
                    "timepoint_idx": time_idx,
                    "pseudotime": time_values.astype(np.float32),
                    "lineage_id": lineage_id,
                    "cell_type": cell_types,
                }
            )
        )

        cell_count = n_cells * args.n_timepoints
        src_mask = time_idx < (args.n_timepoints - 1)
        src = offset + np.where(src_mask)[0]
        tgt = src + 1
        src_indices.append(src.astype(np.int64))
        tgt_indices.append(tgt.astype(np.int64))
        weights.append(np.ones_like(src, dtype=np.float32))

        offset += cell_count

    X = sp.vstack(X_blocks, format="csr")
    obs = pd.concat(obs_records, ignore_index=True)
    obs_names = [
        f"cell_{idx:07d}" for idx in range(obs.shape[0])
    ]
    obs.index = obs_names

    fate_all = np.vstack(fate_blocks).astype(np.float32)

    total_cells = X.shape[0]
    src_indices = np.concatenate(src_indices)
    tgt_indices = np.concatenate(tgt_indices)
    weights = np.concatenate(weights)
    transition = sp.csr_matrix(
        (weights, (src_indices, tgt_indices)),
        shape=(total_cells, total_cells),
    )

    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "anndata is required to write .h5ad files. Install with `pip install anndata`."
        ) from exc

    var = pd.DataFrame(index=gene_names)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsp["transition_matrix"] = transition
    adata.obsp["T_fwd"] = transition.copy()
    adata.obsm["fate_probabilities"] = fate_all
    adata.uns["simulation_info"] = {
        "n_genes": n_genes,
        "n_samples": args.n_samples,
        "n_timepoints": args.n_timepoints,
        "timepoints": timepoints.tolist(),
        "total_cells_requested": int(args.total_cells),
        "total_cells_generated": int(actual_total_cells),
        "neighbor_mode": args.neighbor_mode,
        "neighbor_k": int(args.neighbor_k),
        "neighbor_mix": float(args.neighbor_mix),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    full_path = os.path.join(args.output_dir, "synthetic_full.h5ad")
    adata.write_h5ad(full_path)

    holdout_lineage = args.holdout_lineage
    branch_idx = args.branch_timepoint_index
    mask = ~(
        (adata.obs["lineage_id"] == holdout_lineage)
        & (adata.obs["timepoint_idx"] >= branch_idx)
    )
    adata_train = adata[mask].copy()
    if "transition_matrix" in adata_train.obsp:
        tm = adata_train.obsp["transition_matrix"]
        adata_train.obsp["transition_matrix"] = tm
        adata_train.obsp["T_fwd"] = tm.copy()

    train_path = os.path.join(args.output_dir, "synthetic_train.h5ad")
    adata_train.write_h5ad(train_path)

    grn_path = os.path.join(args.output_dir, "synthetic_grn.json")
    with open(grn_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "gene_names": gene_names,
                "edges": grn["edges"],
                "ligand_receptor_pairs": pair_map,
            },
            handle,
            indent=2,
        )

    print(f"Wrote {full_path}")
    print(f"Wrote {train_path}")
    print(f"Wrote {grn_path}")
    print(f"Total cells generated: {actual_total_cells}")


if __name__ == "__main__":
    main()
