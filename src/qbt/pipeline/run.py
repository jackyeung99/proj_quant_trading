from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence

from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths

from qbt.pipeline.ingestion import ingest_all_sources
from qbt.pipeline.silver import canonicalize_all
from qbt.pipeline.gold import build_gold_model_table


StageFn = Callable[[Storage, StoragePaths, dict], None]


@dataclass(frozen=True)
class Stage:
    name: str
    fn: StageFn
    cfg_key: str               # where to look for enabled flag
    requires: tuple[str, ...] = ()  # optional dependencies


STAGES: Dict[str, Stage] = {
    "ingestion": Stage(
        name="ingestion",
        fn=ingest_all_sources,
        cfg_key="ingestion",
    ),
    "silver": Stage(
        name="silver",
        fn=canonicalize_all,
        cfg_key="silver",
        requires=("ingestion",),   # optional; set to () if you want independent runs
    ),
    "gold": Stage(
        name="gold",
        fn=build_gold_model_table,
        cfg_key="gold",
        requires=("silver",),
    ),
    "signal": Stage(
        name="signal",
        fn=build_gold_model_table,
        cfg_key="signal",
        requires=("gold",),
    ),
    "execution": Stage(
        name="signal",
        fn=build_gold_model_table,
        cfg_key="execution",
        requires=("signal",),
    ),

}


def _is_enabled(cfg: dict, cfg_key: str) -> bool:
    block = cfg.get(cfg_key, {})
    return bool(block.get("enabled", False))


def _topo_sort(selected: Sequence[str]) -> list[str]:
    """Very small dependency resolver for selected stages."""
    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(s: str):
        if s in visited:
            return
        if s in visiting:
            raise ValueError(f"Cyclic stage dependency at '{s}'")
        if s not in STAGES:
            raise KeyError(f"Unknown stage '{s}'. Known: {list(STAGES)}")

        visiting.add(s)
        for dep in STAGES[s].requires:
            if dep in selected:
                dfs(dep)
        visiting.remove(s)
        visited.add(s)
        ordered.append(s)

    for s in selected:
        dfs(s)

    return ordered


def run_pipeline(
    storage: Storage,
    paths: StoragePaths,
    cfg: dict,
    *,
    stages: Optional[Sequence[str]] = None,   # if None, consider all stages
    respect_enabled: bool = True,             # if False, run stages even if disabled
) -> None:
    # choose stages
    chosen = list(stages) if stages is not None else list(STAGES.keys())

    # order by dependencies (optional but nice)
    chosen = _topo_sort(chosen)

    # run
    for name in chosen:
        stage = STAGES[name]
        if respect_enabled and not _is_enabled(cfg, stage.cfg_key):
            continue
        stage.fn(storage, paths, cfg)
