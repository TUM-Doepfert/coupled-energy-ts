"""Stream the Germany H/C tree into per-archetype, per-dwelling mean series.

Why this exists
---------------
``demand_comparison.aggregate_pipeline_heating`` collapses the whole tree
into ONE series by summing over every (location, archetype, profile)
combination with equal weight. That is fine for a peak-normalised shape
comparison, but it is not a stock estimate, so no absolute-magnitude
claim can be made from it. Two biases are baked in:

  * archetype mix - all 11 TABULA-DE vintages contribute equally, while
    the real stock is dominated by the 1958-1978 classes.
  * spatial mix - all 4 045 grid cells contribute equally, so an alpine
    cell with a handful of houses counts as much as a Ruhr cell with
    thousands. This systematically over-weights cold, rural locations.

This script factorises the aggregation so weights become a cheap
post-processing step instead of requiring a re-stream of the ~85 GB tree
every time a weighting assumption changes.

The algebra
-----------
Let ``q[l,a,p](t)`` be simulated power for one dwelling of archetype ``a``
at location ``l`` driven by electricity profile ``p``.

Per-dwelling expectation over household behaviour::

    qbar[l,a](t) = mean_p q[l,a,p](t)

National demand under separable weights ``N[l,a] = N_total * f_a * g_l``
(with ``sum_a f_a = 1`` and ``sum_l g_l = 1``)::

    Q(t) = N_total * sum_a f_a * ( sum_l g_l * qbar[l,a](t) )
                                  \_______________________/
                                           A_a(t)

This script emits ``A_a(t)``: the spatially averaged demand of a single
dwelling of archetype ``a``, in W. Applying ``f_a`` and ``N_total``
afterwards is arithmetic on 11 numbers.

Both the uniform weighting (``g_l = 1/4045``, reproduces the current
behaviour) and a supplied spatial weighting are accumulated in the SAME
pass, so the two are directly comparable without paying for the stream
twice.

Outputs
-------
``archetype_series.parquet``
    index ``timestamp`` (UTC), columns ``arch01_uniform`` ... plus
    ``arch01_weighted`` ... when ``--location-weights`` is given.
    Units W per dwelling.

``annual_energy_by_loc_arch.parquet``
    ``location_id``, ``archetype_id``, ``annual_kwh_per_dwelling``.
    4 045 x 11 scalars. Lets you test alternative SPATIAL weightings on
    annual energy without re-streaming, even though the time series is
    only emitted for the two weightings chosen at run time.

Usage
-----
    uv run python validation/germany/aggregate_by_archetype.py \
        --hc-root output/HC \
        --out-dir validation/germany/data \
        --location-weights validation/germany/data/location_weights.csv

``location_weights.csv`` needs columns ``location_id,weight``. Weights are
renormalised to sum to 1, so raw building counts can be passed directly.
Missing locations are treated as weight 0 and reported.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VALUE_COL_DEFAULT = "q_heat_w"


def load_location_weights(path: Path | None, location_ids: list[int]) -> pd.Series | None:
    """Return a weight per location_id, renormalised to sum to 1."""
    if path is None:
        return None
    w = pd.read_csv(path)
    missing_cols = {"location_id", "weight"} - set(w.columns)
    if missing_cols:
        raise ValueError(f"{path} is missing columns {sorted(missing_cols)}")
    s = w.set_index("location_id")["weight"].astype(float)
    if (s < 0).any():
        raise ValueError(f"{path} contains negative weights")
    s = s.reindex(location_ids)
    n_missing = int(s.isna().sum())
    if n_missing:
        print(f"  [warn] {n_missing} of {len(location_ids)} locations have no "
              f"weight and are treated as 0")
        s = s.fillna(0.0)
    total = float(s.sum())
    if total <= 0:
        raise ValueError(f"{path} weights sum to {total}, cannot renormalise")
    return s / total


def _reduce_batch(batch: list[tuple[str, int, int]],
                  value_col: str,
                  weights: dict[int, float] | None):
    """Reduce a batch of files to per-archetype running sums.

    Returns ``(raw, wtd, annual, profile_counts)`` where ``raw[arch]`` is the
    UNWEIGHTED sum of the per-dwelling means ``qbar[l,a]`` over the batch's
    locations. The uniform factor ``g_l = 1/n_loc`` is applied once at the
    end instead of per file, which is algebraically identical and keeps the
    checkpoint independent of how many locations end up being processed.

    Module-level so joblib's loky backend can pickle it.
    """
    raw: dict[int, pd.Series] = {}
    wtd: dict[int, pd.Series] = {}
    annual: list[tuple[int, int, float]] = []
    profs: set[int] = set()
    for path_str, loc, arch in batch:
        df = pd.read_parquet(path_str, columns=["timestamp", "profile_id", value_col])
        profs.add(int(df["profile_id"].nunique()))
        # Mean over profiles = expected demand of ONE dwelling, marginalising
        # over household behaviour. The existing aggregate sums instead, which
        # is why it cannot be read as a per-dwelling quantity.
        qbar = df.groupby("timestamp")[value_col].mean()
        # Annual energy of that mean dwelling, kWh. Hourly steps.
        annual.append((loc, arch, float(qbar.sum()) / 1000.0))
        raw[arch] = qbar if arch not in raw else raw[arch].add(qbar, fill_value=0.0)
        if weights is not None:
            w = weights.get(loc, 0.0)
            if w > 0.0:
                c = qbar * w
                wtd[arch] = c if arch not in wtd else wtd[arch].add(c, fill_value=0.0)
    return raw, wtd, annual, sorted(profs)


def _merge_into(acc: dict[int, pd.Series], part: dict[int, pd.Series]) -> None:
    for arch, s in part.items():
        acc[arch] = s if arch not in acc else acc[arch].add(s, fill_value=0.0)


def _dict_to_frame(d: dict[int, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({str(k): v for k, v in sorted(d.items())})


def _frame_to_dict(df: pd.DataFrame) -> dict[int, pd.Series]:
    return {int(c): df[c] for c in df.columns}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hc-root", type=Path, default=Path("output/HC"))
    p.add_argument("--out-dir", type=Path, default=Path("validation/germany/data"))
    p.add_argument("--location-weights", type=Path, default=None,
                   help="CSV with columns location_id,weight. Raw building "
                        "counts are fine, they get renormalised.")
    p.add_argument("--value-col", type=str, default=VALUE_COL_DEFAULT,
                   choices=["q_heat_w", "q_cool_w",
                            "q_cool_sensible_w", "q_cool_latent_w"])
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N files. Smoke-test only.")
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallel worker processes. The stream is decode- "
                        "bound, so >1 helps up to the physical core count.")
    p.add_argument("--chunk-size", type=int, default=1000,
                   help="Files per checkpointed chunk.")
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Persist per-chunk partial accumulators here so an "
                        "interrupted run resumes instead of restarting. "
                        "Chunks are keyed by position in the sorted file "
                        "list, so the same --hc-root/--limit must be used.")
    args = p.parse_args()

    try:
        from tqdm import tqdm
    except ImportError:                                   # pragma: no cover
        def tqdm(x, **kw):                                # type: ignore
            return x

    files = sorted(args.hc_root.glob("loc*/hc_arch*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquets matching {args.hc_root}/loc*/hc_arch*.parquet")
    if args.limit:
        files = files[:args.limit]
    print(f"[agg] {len(files)} files under {args.hc_root}")

    # location_id / archetype_id live in the path, not in the file.
    def ids(f: Path) -> tuple[int, int]:
        return int(f.parent.name[3:]), int(f.stem[len("hc_arch"):])

    parsed = [(f, *ids(f)) for f in files]
    location_ids = sorted({l for _, l, _ in parsed})
    archetype_ids = sorted({a for _, _, a in parsed})
    print(f"[agg] {len(location_ids)} locations, {len(archetype_ids)} archetypes")

    g = load_location_weights(args.location_weights, location_ids)
    g_uniform = 1.0 / len(location_ids)

    weights = None if g is None else {int(k): float(v) for k, v in g.items()}

    acc_raw: dict[int, pd.Series] = {}          # unweighted sum of qbar
    acc_weighted: dict[int, pd.Series] = {}
    annual_rows: list[tuple[int, int, float]] = []
    n_profiles_seen: set[int] = set()

    ckpt = args.checkpoint_dir
    if ckpt is not None:
        ckpt.mkdir(parents=True, exist_ok=True)

    chunks = [parsed[i:i + args.chunk_size]
              for i in range(0, len(parsed), args.chunk_size)]
    print(f"[agg] {len(chunks)} chunk(s) of <= {args.chunk_size} files, "
          f"jobs={args.jobs}, checkpoint={'off' if ckpt is None else ckpt}")

    if args.jobs > 1:
        from joblib import Parallel, delayed

    bar = tqdm(total=len(parsed), desc="[agg] streaming", unit="file",
               smoothing=0.05)
    for ci, chunk in enumerate(chunks):
        done_marker = None if ckpt is None else ckpt / f"chunk_{ci:05d}.done"
        if done_marker is not None and done_marker.exists():
            # Resume: the marker is written LAST, so its presence means all
            # three payload files for this chunk are complete on disk.
            raw_part = _frame_to_dict(pd.read_parquet(ckpt / f"chunk_{ci:05d}_raw.parquet"))
            wpath = ckpt / f"chunk_{ci:05d}_wtd.parquet"
            wtd_part = (_frame_to_dict(pd.read_parquet(wpath))
                        if wpath.exists() else {})
            ann = pd.read_parquet(ckpt / f"chunk_{ci:05d}_annual.parquet")
            annual_part = list(ann.itertuples(index=False, name=None))
            profs = [int(x) for x in
                     done_marker.read_text().strip().split(",") if x]
            bar.update(len(chunk))
            bar.set_postfix_str(f"chunk {ci} from checkpoint")
        else:
            if args.jobs > 1:
                nb = min(args.jobs, len(chunk))
                batches = [chunk[i::nb] for i in range(nb)]
                results = Parallel(n_jobs=nb, backend="loky")(
                    delayed(_reduce_batch)(
                        [(str(f), l, a) for f, l, a in b],
                        args.value_col, weights)
                    for b in batches)
            else:
                results = [_reduce_batch([(str(f), l, a) for f, l, a in chunk],
                                         args.value_col, weights)]
            raw_part, wtd_part, annual_part, profs = {}, {}, [], []
            for r, w_, ann_, pf in results:
                _merge_into(raw_part, r)
                _merge_into(wtd_part, w_)
                annual_part.extend(ann_)
                profs.extend(pf)
            # Interleaved batching scrambles row order within a chunk; the
            # serial path emits file order, so restore it here.
            annual_part.sort(key=lambda t: (t[0], t[1]))
            bar.update(len(chunk))
            if ckpt is not None:
                _dict_to_frame(raw_part).to_parquet(ckpt / f"chunk_{ci:05d}_raw.parquet")
                if wtd_part:
                    _dict_to_frame(wtd_part).to_parquet(ckpt / f"chunk_{ci:05d}_wtd.parquet")
                pd.DataFrame(annual_part,
                             columns=["location_id", "archetype_id",
                                      "annual_kwh_per_dwelling"]
                             ).to_parquet(ckpt / f"chunk_{ci:05d}_annual.parquet",
                                          index=False)
                done_marker.write_text(",".join(str(x) for x in sorted(set(profs))))

        _merge_into(acc_raw, raw_part)
        _merge_into(acc_weighted, wtd_part)
        annual_rows.extend(annual_part)
        n_profiles_seen.update(int(x) for x in profs)
    bar.close()

    # g_l = 1/n_loc applied once, after the stream.
    acc_uniform = {a: s_ * g_uniform for a, s_ in acc_raw.items()}

    if len(n_profiles_seen) != 1:
        print(f"  [warn] inconsistent profile counts across files: "
              f"{sorted(n_profiles_seen)}. The per-dwelling mean is still "
              f"correct per file, but the sample size varies.")
    else:
        print(f"[agg] {n_profiles_seen.pop()} profiles per (location, archetype)")

    out = {}
    for arch in archetype_ids:
        out[f"arch{arch:02d}_uniform"] = acc_uniform[arch]
        if g is not None and arch in acc_weighted:
            out[f"arch{arch:02d}_weighted"] = acc_weighted[arch]
    series = pd.DataFrame(out)
    series.index = pd.to_datetime(series.index)
    series.index = (series.index.tz_localize("UTC") if series.index.tz is None
                    else series.index.tz_convert("UTC"))
    series = series.sort_index()
    series.index.name = "timestamp"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.value_col == VALUE_COL_DEFAULT else f"_{args.value_col}"
    s_path = args.out_dir / f"archetype_series{suffix}.parquet"
    e_path = args.out_dir / f"annual_energy_by_loc_arch{suffix}.parquet"
    series.to_parquet(s_path)
    pd.DataFrame(annual_rows,
                 columns=["location_id", "archetype_id",
                          "annual_kwh_per_dwelling"]).to_parquet(e_path, index=False)
    print(f"[agg] -> {s_path}  ({series.shape[0]} timestamps, "
          f"{series.shape[1]} columns)")
    print(f"[agg] -> {e_path}  ({len(annual_rows)} rows)")

    # Sanity readout: per-archetype annual energy of one average dwelling.
    print("\nAnnual energy per dwelling, kWh (spatially averaged):")
    for arch in archetype_ids:
        u = acc_uniform[arch].sum() / 1000.0
        line = f"  arch{arch:02d}  uniform {u:9.0f}"
        if g is not None and arch in acc_weighted:
            line += f"   weighted {acc_weighted[arch].sum() / 1000.0:9.0f}"
        print(line)


if __name__ == "__main__":
    main()
