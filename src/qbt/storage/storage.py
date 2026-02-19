from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import json
import pickle
import uuid

from qbt.core.logging import get_logger

logger = get_logger(__name__)


import s3fs
import pandas as pd
import pyarrow.parquet as pq


class Storage:
    def exists(self, key: str) -> bool:

        logger.debug(f"checking file path: {key}")
        raise NotImplementedError
    
    def list(self, prefix: str) -> list[str]:
        """
        List keys under a prefix (non-recursive or recursive depending on backend).

        Returns keys relative to storage root (same format used in read/write).
        """
        raise NotImplementedError

    def read_parquet(self, key: str) -> pd.DataFrame:
        logger.debug(f"reading file: {key}")
        raise NotImplementedError

    def write_parquet(self, df: pd.DataFrame, key: str) -> None:
        logger.info(f"Writing {len(df)} rows to {key}")
        raise NotImplementedError

    def read_json(self, key: str) -> Any:
        logger.debug(f"reading file: {key}")
        raise NotImplementedError

    def write_json(self, obj: Any, key: str) -> None:
        logger.info(f"Writing json to {key}")
        raise NotImplementedError
    

    def read_pickle(self, key):
        logger.info(f"reading file:{key}")
        raise NotImplementedError
    
    def write_pickle(self, obj: Any, key: str):
        logger.info(f"Writing Pickle to {key}")
        raise NotImplementedError



@dataclass(frozen=True)
class LocalStorage(Storage):
    base_dir: Path

    def _path(self, key: str) -> Path:
        return (self.base_dir / key).resolve()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()
    
    def list(self, prefix: str) -> list[str]:
        base = self.base_dir.resolve()
        path = (base / prefix).resolve()

        if not path.exists():
            return []

        if path.is_file():
            return [prefix]

        out = []
        for p in path.rglob("*"):
            if p.is_file():
                # make both absolute → safe
                rel = p.resolve().relative_to(base)
                out.append(str(rel).replace("\\", "/"))

        return sorted(out)

    # =====================
    # Parquet
    # =====================
    def read_parquet(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(key))

    def write_parquet(self, df: pd.DataFrame, key: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=True)
        tmp.replace(path)

    # =====================
    # JSON
    # =====================
    def read_json(self, key: str) -> Any:
        path = self._path(key)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def write_json(self, obj: Any, key: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str)
        tmp.replace(path)

    # =====================
    # Pickle
    # =====================
    def read_pickle(self, key: str) -> Any:
        path = self._path(key)
        with path.open("rb") as f:
            return pickle.load(f)

    def write_pickle(self, obj: Any, key: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)


@dataclass
class S3Storage(Storage):
    """
    S3-backed storage using s3fs/fsspec under the hood.

    - key is a path relative to (bucket, prefix)
      e.g. key="data/bronze/equities.parquet"
    - best-effort "atomic" write:
        write to temp key -> copy to final -> delete temp
      (S3 doesn't support true atomic rename)
    """
    bucket: str
    prefix: str = ""

    # credential/config controls (optional)
    region: Optional[str] = None
    profile: Optional[str] = None
    endpoint_url: Optional[str] = None  # for MinIO/localstack if needed

    # internal cached fs (don’t pass in init)
    _fs: Any = field(init=False, repr=False, default=None)

    def _init_fs(self):
        if self._fs is not None:
            return


        client_kwargs = {}
        if self.region:
            client_kwargs["region_name"] = self.region
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url

        # profile works locally (shared credentials); on ECS you typically won't set it
        self._fs = s3fs.S3FileSystem(profile=self.profile, client_kwargs=client_kwargs)

    def _key(self, key: str) -> str:
        key = key.lstrip("/")
        pref = self.prefix.strip("/")
        return f"{pref}/{key}" if pref else key

    def _uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._key(key)}"

    def exists(self, key: str) -> bool:
        self._init_fs()
        p = f"{self.bucket}/{self._key(key)}"   # NO s3://
        try:
            info = self._fs.info(p)             # raises if missing
            return info.get("type") == "file"
        except FileNotFoundError:
            return False
    

    def list(self, prefix: str) -> list[str]:
        self._init_fs()

        key_prefix = self._key(prefix)
        uri = f"s3://{self.bucket}/{key_prefix}"

        try:
            paths = self._fs.ls(uri, detail=False)
        except FileNotFoundError:
            return []

        out = []
        for p in paths:
            # p format: bucket/prefix/... or s3://bucket/...
            if p.startswith(f"{self.bucket}/"):
                k = p[len(self.bucket) + 1 :]
            elif p.startswith("s3://"):
                k = p.split("/", 3)[-1]
            else:
                k = p

            # remove configured prefix so key matches read/write API
            pref = self.prefix.strip("/")
            if pref and k.startswith(pref + "/"):
                k = k[len(pref) + 1 :]

            out.append(k)

        return sorted(out)
    
    def read_parquet(self, key: str) -> pd.DataFrame:
        # pandas will use s3fs via fsspec if installed
        return pd.read_parquet(
                self._uri(key),
                engine="pyarrow",
                filesystem=self._fs,
                partitioning=None,   # <-- disables hive inference
            )

    def write_parquet(self, df: pd.DataFrame, key: str) -> None:
        self._init_fs()
        tmp_key = f"{self._key(key)}.__tmp__{uuid.uuid4().hex}"
        tmp_uri = f"s3://{self.bucket}/{tmp_key}"

        df.to_parquet(tmp_uri, index=True, engine="pyarrow", filesystem=self._fs)

        src = f"{self.bucket}/{tmp_key}"
        dst = f"{self.bucket}/{self._key(key)}"
        self._fs.copy(src, dst)

        try:
            self._fs.rm(src)
        except Exception:
            pass


    def read_json(self, key: str) -> Any:
        self._init_fs()
        uri = self._uri(key)
        with self._fs.open(uri, "r") as f:
            return json.load(f)

    def write_json(self, obj: Any, key: str) -> None:
        self._init_fs()
        final_uri = self._uri(key)

        tmp_key = f"{self._key(key)}.__tmp__{uuid.uuid4().hex}"
        tmp_uri = f"s3://{self.bucket}/{tmp_key}"

        with self._fs.open(tmp_uri, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str)

        src = f"{self.bucket}/{tmp_key}"
        dst = f"{self.bucket}/{self._key(key)}"
        self._fs.copy(src, dst)

        try:
            self._fs.rm(src)
        except Exception:
            pass

    def read_pickle(self, key: str) -> Any:
        """
        Read a python object serialized via pickle from S3.
        """
        self._init_fs()
        uri = self._uri(key)
        with self._fs.open(uri, "rb") as f:
            return pickle.load(f)

    def write_pickle(self, obj: Any, key: str) -> None:
        """
        Write a python object to S3 using pickle, with best-effort atomic write:
        write temp -> copy to final -> delete temp.
        """
        self._init_fs()

        # temp key alongside final
        tmp_key = f"{self._key(key)}.__tmp__{uuid.uuid4().hex}"
        tmp_uri = f"s3://{self.bucket}/{tmp_key}"

        # write temp
        with self._fs.open(tmp_uri, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        # copy temp -> final (overwrite)
        src = f"{self.bucket}/{tmp_key}"
        dst = f"{self.bucket}/{self._key(key)}"
        self._fs.copy(src, dst)

        # cleanup temp
        try:
            self._fs.rm(src)
        except Exception:
            pass


def make_storage(cfg: dict) -> Storage:
    s = cfg["storage"]
    backend = s["backend"].lower()

    if backend == "local":
        return LocalStorage(base_dir=Path(s["base_dir"]))

    if backend == "s3":
        bucket = s["bucket"]
        prefix = s.get("prefix", "")
        region = s.get("region")
        profile = s.get("profile")
        endpoint_url = s.get("endpoint_url")

        logger.info(
            "Initializing S3Storage | bucket=%s prefix=%r region=%s profile=%s endpoint=%s",
            bucket,
            prefix,
            region,
            profile,
            endpoint_url,
        )

        return S3Storage(
            bucket=bucket,
            prefix=prefix,
            region=region,
            profile=profile,
            endpoint_url=endpoint_url,
        )

    raise ValueError(f"Unknown storage.backend={backend!r}")