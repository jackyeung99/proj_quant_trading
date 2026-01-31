


def run_pipeline(storage: Storage, paths: StoragePaths, cfg: dict) -> None:
    ingest_all_sources(storage, paths, cfg["ingest"])
    canonicalize_all(storage, paths, cfg["canonicalize"])
    build_gold_model_table(storage, paths, cfg["gold"])