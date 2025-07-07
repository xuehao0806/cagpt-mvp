"""
dataset.py
~~~~~~~~~~
Download-on-demand loader for classic Biogeme choice datasets.

Usage
-----
from cagpt_mvp import ChoiceDataset
ds = ChoiceDataset("lpmc")     # or "swissmetro", ...
X, y = ds.get_xy()
"""
from __future__ import annotations

import pathlib, zipfile, hashlib, os
from typing import Dict, Literal

import pandas as pd
import requests
from tqdm import tqdm

# --------------------------------------------------------------------
# 1. 数据集注册表：URL / 本地文件名 / 文件格式 / 目标列名
# --------------------------------------------------------------------
_DATA_REGISTRY: Dict[str, Dict] = {
    "lpmc": {        # London Passenger Mode Choice
        "url": "http://transp-or.epfl.ch/data/lpmc.dat",
        "file": "lpmc.dat",
        "format": "tab",
        "target": "travel_mode",
    },
    "swissmetro": {
        "url": "http://transp-or.epfl.ch/data/swissmetro.dat",
        "file": "swissmetro.dat",
        "format": "tab",
        "target": "CHOICE",
    },
    "parking": {
        "url": "http://transp-or.epfl.ch/data/parking.dat",
        "file": "parking.dat",
        "format": "tab",
        "target": "CHOICE",
    },
    "netherlands": {
        "url": "http://transp-or.epfl.ch/data/netherlands.dat",
        "file": "netherlands.dat",
        "format": "tab",
        "target": "choice",
    },
    "airline": {
        "url": "http://transp-or.epfl.ch/data/airline.dat",
        "file": "airline.dat",
        "format": "tab",
        "target": "CHOICE",   # 会在读取后动态生成
    },
    "optima": {
        "url": "http://transp-or.epfl.ch/data/optima.dat",
        "file": "optima.dat",
        "format": "tab",
        "target": "Choice",   # Optima 的列名以首字母大写发布
    },
}

# 默认缓存目录：<项目根>/data
_CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# 2. 工具函数
# --------------------------------------------------------------------
def _download(url: str, dst: pathlib.Path, name: str) -> None:
    """带进度条下载，如果文件已存在则跳过。"""
    if dst.exists():
        return
    print(f"[dataset] Downloading {name} …")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tqdm.wrapattr(open(dst, "wb"), "write", total=total,
                           unit="B", unit_scale=True) as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def _read_first_inside_zip(zip_path: pathlib.Path) -> pd.DataFrame:
    """若数据以 .zip 发布，取其中第一个 .csv/.dat/.txt 文件解析。"""
    with zipfile.ZipFile(zip_path) as zf:
        fname = next(
            f for f in zf.namelist()
            if f.lower().endswith((".csv", ".dat", ".txt"))
        )
        with zf.open(fname) as f:
            if fname.lower().endswith(".csv"):
                return pd.read_csv(f)
            else:
                return pd.read_csv(f, sep=r"\s+")


# --------------------------------------------------------------------
# 3. 核心类
# --------------------------------------------------------------------
class ChoiceDataset:
    """懒下载 Biogeme 数据集，并统一暴露 (X, y)。"""

    def __init__(
        self,
        name: Literal[
            "lpmc", "swissmetro", "parking",
            "netherlands", "airline", "optima"
        ],
        cache_dir: str | os.PathLike | None = None,
    ):
        if name not in _DATA_REGISTRY:
            raise ValueError(f"Unknown dataset key: {name}")

        self.info = _DATA_REGISTRY[name]
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.cache_dir / self.info["file"]

        # ---- 下载 ----
        _download(self.info["url"], self.path, name)

        # ---- 解析 ----
        fmt = self.info["format"]
        if fmt == "csv":
            df = pd.read_csv(self.path)
        elif fmt == "tab":
            df = pd.read_csv(self.path, sep=r"\s+")
        elif fmt == "zip":
            df = _read_first_inside_zip(self.path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        # 去掉全空列
        df = df.dropna(axis=1, how="all")

        # ---- Airline 特殊处理：由两列哑变量生成 CHOICE ----
        if name == "airline":
            # 自动匹配列名：容忍大小写/空格/下划线差异
            norm = lambda s: s.replace(" ", "").replace("_", "").lower()
            col1 = next(c for c in df.columns if norm(c) == "bestalternative1")
            col2 = next(c for c in df.columns if norm(c) == "bestalternative2")
            df["CHOICE"] = (
                1 * (df[col1] == 1) +
                2 * (df[col2] == 1)
            ).replace(0, 3)

        # ---- 提取特征 / 目标 ----
        tgt_col = self.info["target"]
        if tgt_col not in df.columns:
            raise KeyError(
                f"Target column '{tgt_col}' not found in {name}. "
                "Please update _DATA_REGISTRY or preprocessing logic."
            )
        self.df = df
        self.X = df.drop(columns=[tgt_col])
        self.y = df[tgt_col]

    # -------------------- 公共接口 -------------------- #
    def get_xy(self):
        """返回 (features DataFrame, target Series)。"""
        return self.X.copy(), self.y.copy()

    def __repr__(self) -> str:
        return f"<ChoiceDataset {self.info['file']}  shape={self.df.shape}>"
