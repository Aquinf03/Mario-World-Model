"""
Stage 3 adapter: train SAEs on JEPA latents with aquin's SAE tooling.

WHY THIS FILE EXISTS
--------------------
aquin's activation store is genuinely dimension-agnostic: a directory of
`chunk_*.pt`, each a (N, d) tensor, optional `norm.pt` with {mean,std}, optional
`manifest.json`. validate_acts_manifest() raises ONLY on a layer mismatch — a
model-id mismatch is a printed warning, and a missing manifest short-circuits
validation entirely. So 192-dim JEPA latents go in cleanly.

BUT the `aquin sae train` CLI cannot be used: cmd_sae_train never passes
d_model down, so the SAE is built at the session model's width (768 for GPT-2)
and silently mismatches our 192. We therefore call the Python entrypoint
directly, where d_model and n_features are real keyword arguments.

Two warts, accepted knowingly rather than papered over:
  * `model_id` must resolve in aquin's catalog (resolve_model_id runs before the
    d_model override), so we pass a placeholder purely for config lookup. The
    trained SAE registers under that placeholder name.
  * `layer` is unchecked when no manifest is written, so we reuse it as a free
    integer slot to key the SIGReg-lambda index.
"""
from __future__ import annotations

from pathlib import Path

from algo.config import CFG


def dump_latents(ckpt: str, out_dir: str | Path, loader=None,
                 chunk_vectors: int | None = None, data_dir=None,
                 max_obs_per_episode: int | None = None) -> Path:
    """
    Encode frames -> 192-dim latents and write them as aquin activation chunks.

    Deliberately writes NO manifest.json: with none present aquin skips model-id
    and layer validation entirely, which is exactly what we want for a latent
    space that has no catalog model behind it.
    """
    import torch

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    n = chunk_vectors or CFG.sae.chunk_vectors

    if loader is None:
        # Wired to the Stage 2 encoder via gates.stream_latents, which already
        # owns shard inflation and episode slicing. Uses the SAME held-out
        # trajectory split as the gates, so a feature found here is a feature
        # about data the encoder was scored on, not data it was fitted to.
        from algo import data as Dt
        from algo.gates import stream_latents
        from algo.model import load_jepa

        import torch as _T
        dev = "cuda" if _T.cuda.is_available() else "cpu"
        model = load_jepa(ckpt, cfg=CFG).to(dev)
        paths = Dt.shard_paths(Path(data_dir or CFG.data.data_dir))
        if not paths:
            raise SystemExit(f"no shards under {data_dir or CFG.data.data_dir}")
        _, val_eps = Dt.split_episodes(paths, CFG.gate.probe_trajectories, seed=0)
        loader = stream_latents(model, val_eps, CFG, device=dev,
                                max_obs_per_episode=max_obs_per_episode)

    buf, idx, total = [], 0, 0
    for z in loader:                      # z: (B, 192) encoder output
        buf.append(z.detach().cpu().float())
        if sum(b.shape[0] for b in buf) >= n:
            t = torch.cat(buf)[:n]
            torch.save(t, out / f"chunk_{idx:05d}.pt")
            total += t.shape[0]
            buf, idx = [torch.cat(buf)[n:]], idx + 1
    if buf and (rest := __import__("torch").cat(buf)).shape[0]:
        torch.save(rest, out / f"chunk_{idx:05d}.pt")
        total += rest.shape[0]

    # norm.pt is optional but saves aquin a full pass to compute mean/std
    import torch as T
    allv = T.cat([T.load(p, map_location="cpu") for p in sorted(out.glob("chunk_*.pt"))])
    T.save({"mean": allv.mean(0), "std": allv.std(0).clamp_min(1e-6)}, out / "norm.pt")
    print(f"[aq_sae] wrote {idx+1} chunks, {total} vectors of dim {allv.shape[1]} -> {out}")
    if total < 10_000:
        print("[aq_sae] warning: aquin warns below 10k vectors; SAE quality will be poor")
    return out


def train_latent_sae(acts_dir: str | Path, tag: str, layer: int = 0,
                     output: str | Path | None = None,
                     d_model: int | None = None,
                     n_features: int | None = None,
                     max_steps: int | None = None) -> Path:
    """
    Call aquin's SAE trainer on our latents with an explicit d_model.
    This is the path the CLI cannot express.
    """
    from aquin.compute.sae_train import train_sae

    out = Path(output) if output else (CFG.run_dir / "sae" / f"{tag}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    return train_sae(
        CFG.sae.placeholder_model_id,       # catalog lookup only; overridden below
        layer,
        out,
        d_model=d_model or CFG.sae.d_model,          # 192, NOT the catalog width
        n_features=n_features or CFG.sae.n_features,  # 2048, not aquin's 32768 default
        max_steps=max_steps,
        activations_dir=str(acts_dir),                # implies no forward passes
    )


def dump_latents_with_labels(ckpt, out_dir, data_dir=None, cfg=CFG,
                             max_obs_per_episode=600, chunk_vectors=None):
    """
    Latents AS aquin activation chunks, PLUS the RAM ground truth aligned to
    them, so a feature can be asked what it is selective for.

    Without the labels an SAE gives you 2048 anonymous directions and a
    reconstruction score, which is not an interpretability result. With them the
    lambda sweep can say the thing it exists to say: *which* features SIGReg
    destroys, in terms a person can check.

    Latents come from the SAME held-out trajectories the gates score, so a
    feature found here is a feature about data the encoder was graded on, not
    fitted to.
    """
    import torch
    from algo import data as Dt
    from algo.gates import encode_episodes
    from algo.model import load_jepa

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    n = chunk_vectors or cfg.sae.chunk_vectors
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_jepa(ckpt, cfg=cfg).to(dev)
    paths = Dt.shard_paths(Path(data_dir or cfg.data.data_dir))
    if not paths:
        raise SystemExit(f"no shards under {data_dir or cfg.data.data_dir}")
    _, val = Dt.split_episodes(paths, cfg.gate.probe_trajectories, seed=0)
    cache = encode_episodes(model, val, cfg, dev, max_obs_per_episode=max_obs_per_episode)

    Z = cache["z"].float()
    for i in range(0, len(Z), n):
        torch.save(Z[i:i + n], out / f"chunk_{i // n:05d}.pt")
    torch.save({"mean": Z.mean(0), "std": Z.std(0).clamp_min(1e-6)}, out / "norm.pt")
    labels = {k: torch.from_numpy(cache[k]).float()
              for k in ("world_x", "world_y", "scroll", "dies_in_5")}
    labels["ep_id"] = torch.from_numpy(cache["ep_id"])
    torch.save(labels, out / "labels.pt")
    print(f"[aq_sae] {len(Z)} latents of dim {Z.shape[1]} + labels -> {out}", flush=True)
    if len(Z) < 10_000:
        print("[aq_sae] warning: aquin warns below 10k vectors; SAE quality will be poor")
    return out


def load_sae_decoder(sae_path, d_model=None):
    """
    (n_features, d_model) decoder from an aquin SAE checkpoint.

    VERIFIED against aquin.compute.sae_train by running it: the file is
    {d_model, n_features, state_dict} and state_dict["W_dec"] is already
    (n_features, d_model). Asserted rather than assumed, because a silently
    transposed dictionary makes every feature below a linear combination of the
    wrong thing and nothing downstream would complain.
    """
    import torch
    blob = torch.load(Path(sae_path), map_location="cpu", weights_only=False)
    sd = blob.get("state_dict", blob)
    W = sd["W_dec"]
    d = d_model or blob.get("d_model") or CFG.sae.d_model
    if W.shape[-1] != d:
        W = W.T
    assert W.shape[-1] == d, f"W_dec {tuple(W.shape)} does not match d_model {d}"
    return W, sd


def feature_activations(sae_path, acts_dir, batch=8192):
    """Run the SAE encoder over the dumped latents -> (N, n_features) codes."""
    import torch
    _, sd = load_sae_decoder(sae_path)
    W_enc, b_enc, b_pre = sd["W_enc"], sd["b_enc"], sd.get("b_pre", 0.0)
    chunks = sorted(Path(acts_dir).glob("chunk_*.pt"))
    outs = []
    for c in chunks:
        z = torch.load(c, map_location="cpu", weights_only=False).float()
        for i in range(0, len(z), batch):
            x = z[i:i + batch] - b_pre
            outs.append(torch.relu(x @ W_enc + b_enc))
    return torch.cat(outs)


def feature_report(sae_path, acts_dir, min_fire_rate=1e-4) -> dict:
    """
    What each SAE feature is selective for, in RAM units.

    For every live feature: how often it fires, and the correlation of its
    activation with world_x / world_y / scroll / dies_in_5. A feature with
    |corr| > 0.3 against world_y is a "height feature", and whether those exist
    is the whole question the lambda sweep asks.
    """
    import torch
    A = feature_activations(sae_path, acts_dir)
    labels = torch.load(Path(acts_dir) / "labels.pt", map_location="cpu", weights_only=False)
    fire = (A > 0).float().mean(0)
    live = torch.nonzero(fire > min_fire_rate).flatten()

    def corr(a, b):
        a = a - a.mean(0, keepdim=True)
        b = b - b.mean()
        na, nb = a.norm(dim=0).clamp_min(1e-8), b.norm().clamp_min(1e-8)
        return (a * b.unsqueeze(1)).sum(0) / (na * nb)

    Al = A[:, live]
    out = {"n_features": int(A.shape[1]), "n_live": int(len(live)),
           "n_latents": int(A.shape[0]),
           "mean_l0": float((A > 0).float().sum(1).mean()),
           "dead_frac": float(1 - len(live) / A.shape[1])}
    per = {}
    for k in ("world_x", "world_y", "scroll", "dies_in_5"):
        c = corr(Al, labels[k])
        per[k] = c
        out[f"n_selective_{k}"] = int((c.abs() > 0.3).sum())
        out[f"max_abs_corr_{k}"] = float(c.abs().max())
    out["live_idx"] = live.tolist()
    out["corr"] = {k: v.tolist() for k, v in per.items()}
    top = per["world_y"].abs().argsort(descending=True)[:10]
    out["top_y_features"] = [{"feature": int(live[i]),
                              "corr_y": float(per["world_y"][i]),
                              "corr_x": float(per["world_x"][i]),
                              "fire_rate": float(fire[live[i]])} for i in top]
    return out


def lambda_diff(sae_paths: dict, acts_dirs: dict, output=None) -> Path:
    """
    THE Stage 3 headline: which features does SIGReg lambda destroy?

    Two halves, both computed here rather than shelled out:

    1. DICTIONARY MATCHING. For every feature of SAE A, the best cosine match
       among B's decoder directions. Features whose best match is weak are
       features that exist at one lambda and not the other. (`aquin sae align`
       compares two SAE files and both are 192-dim here, so it also works — it
       is run alongside when available, but its stdout is not the result.)

    2. WHAT THOSE FEATURES WERE FOR. Matching alone says "37 features changed",
       which is not interpretable. Cross-referencing each feature against the RAM
       ground truth turns it into "the features that vanish are the ones tracking
       Mario's height", which is a claim somebody can disagree with.
    """
    import itertools, json, subprocess
    import torch

    out = Path(output) if output else (CFG.run_dir / "sae" / "lambda_diff.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    reports = {k: feature_report(sae_paths[k], acts_dirs[k]) for k in sae_paths}
    decs = {k: torch.nn.functional.normalize(load_sae_decoder(sae_paths[k])[0], dim=-1)
            for k in sae_paths}

    pairs = []
    for a, b in itertools.combinations(sorted(sae_paths, key=float), 2):
        la = torch.tensor(reports[a]["live_idx"])
        lb = torch.tensor(reports[b]["live_idx"])
        sim = decs[a][la] @ decs[b][lb].T                    # (live_a, live_b)
        best, arg = sim.max(dim=1)
        ya = torch.tensor(reports[a]["corr"]["world_y"]).abs()
        orphan = best < 0.5
        pairs.append({
            "lambda_a": a, "lambda_b": b,
            "live_a": len(la), "live_b": len(lb),
            "median_best_match": float(best.median()),
            "orphaned_a": int(orphan.sum()),
            "orphaned_frac": float(orphan.float().mean()),
            # the claim that matters: are the LOST features the y-selective ones?
            "mean_y_corr_matched": float(ya[~orphan].mean()) if (~orphan).any() else None,
            "mean_y_corr_orphaned": float(ya[orphan].mean()) if orphan.any() else None,
            "y_features_a": reports[a]["n_selective_world_y"],
            "y_features_b": reports[b]["n_selective_world_y"],
        })
        r = subprocess.run(["aquin", "sae", "align", "--sae-a", str(sae_paths[a]),
                            "--sae-b", str(sae_paths[b])], capture_output=True, text=True)
        pairs[-1]["aquin_align_ok"] = r.returncode == 0
        pairs[-1]["aquin_align_stdout"] = r.stdout[-1500:]

    summary = {k: {m: reports[k][m] for m in
                   ("n_live", "dead_frac", "mean_l0", "n_selective_world_y",
                    "n_selective_world_x", "max_abs_corr_world_y")}
               for k in reports}
    payload = {"per_lambda": summary, "pairs": pairs,
               "top_y_features": {k: reports[k]["top_y_features"] for k in reports}}
    out.write_text(json.dumps(payload, indent=2))
    print(f"[aq_sae] wrote {out}")
    return out
