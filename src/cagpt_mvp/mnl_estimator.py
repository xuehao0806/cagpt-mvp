# src/cagpt_mvp/mnl_estimator.py
import collections, collections.abc   # ← 新增
collections.Iterable = collections.abc.Iterable
import pandas as pd, pylogit
from .preprocess import load_meta, apply_metadata
from collections import OrderedDict
import numpy as np
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None

def to_long(df: pd.DataFrame, meta: dict, debug: bool = False):
    """
    宽表 ➜ 长表。返回 (long_df, obs_id_col)。

    约束
    -------
    • 同一 obs_id 的行严格相邻（pylogit 要求）
    • 每个 obs_id 恰有一行 choice == 1
    """

    pid = "obs_id"
    tgt = meta["target_primary"]

    # ---- 基础复制 ----
    df = df.copy().reset_index().rename(columns={"index": pid})

    # ---- 示例派生列（LPMC）----
    pt_cols = {"dur_pt_access", "dur_pt_rail", "dur_pt_bus", "dur_pt_int"}
    if pt_cols.issubset(df.columns):
        df["dur_pt_total"] = df[list(pt_cols)].sum(axis=1)

    # ---- 备选集合（剔除 generic 等虚拟 alt_id） ----
    alt_ids = sorted({
        v["alt_id"] for v in meta["fields"].values()
        if v.get("role") == "alt_attr" and v["alt_id"] not in {"generic"}
    })

    # ---- 编码映射 ----
    code_map = meta.get("alt_code_map", {})      # {"1":"walk", ...}
    mapped_mode = (
        df[tgt].astype(str)          # 统一转 str 方便映射
          .map(code_map)             # 数字 -> 名称
          .fillna(df[tgt].astype(str))   # 若未映射则保持原值
    )

    # ---- 堆叠 ----
    long_parts = []
    for alt in alt_ids:
        cols = [c for c, v in meta["fields"].items()
                if v.get("alt_id") == alt]

        sub = df[[pid] + cols].copy()
        sub["alt_id"] = alt
        sub["choice"] = (mapped_mode == alt).astype(int)

        # 去掉后缀 _alt /  alt
        sub = sub.rename(columns=lambda x: x.replace(f"_{alt}", "")
                                          .replace(f" {alt}", ""))
        long_parts.append(sub)

    long_df = pd.concat(long_parts, axis=0, ignore_index=True)

    # ---- 排序保证相邻 ----
    long_df = long_df.sort_values([pid, "alt_id"]).reset_index(drop=True)

    # ---- 调试输出 ----
    if debug:
        bad = long_df.groupby(pid)["choice"].sum()
        print("non-single choice obs_ids:", bad[bad != 1].head())
        print("orig codes  :", df[tgt].unique()[:8])
        print("mapped codes:", mapped_mode.unique()[:8])
        print("alt_ids     :", alt_ids)

    # ---- Sanity check ----
    assert long_df.groupby(pid)["choice"].sum().eq(1).all(), \
        "Each obs_id must have exactly one chosen alternative"

    return long_df, pid


def clean_for_pylogit(long_df, spec_cols):
    long_df = long_df.replace([np.inf, -np.inf], np.nan)

    for col in spec_cols:
        if long_df[col].dtype.kind in "bifc":          # 数值列
            med = long_df[col].median()
            long_df[col].fillna(med, inplace=True)
        else:                                          # object / category
            mode = long_df[col].mode(dropna=True)
            fill_val = mode.iloc[0] if not mode.empty else 0
            long_df[col].fillna(fill_val, inplace=True)

    # 若仍有 NaN, 删除该行
    long_df.dropna(subset=spec_cols, inplace=True)
    return long_df


def build_spec(meta, long_df):
    uspec   = meta.get("utility_spec", {})
    spec    = OrderedDict()
    asc_raw = uspec.get("ASC", [])
    all_alts = sorted(long_df["alt_id"].unique())

    for var, alts in uspec.items():
        if var == "ASC":
            continue
        if var not in long_df.columns:
            print(f"[warn] {var} not in DataFrame, drop from spec.")
            continue

        # ---- 统一成列表 ----
        if isinstance(alts, str):
            alts = all_alts if alts.lower() == "all" else [alts]
        elif not isinstance(alts, (list, tuple)):
            alts = [alts]

        # ---- 关键：外层再包一层 ----
        spec[var] = [list(alts)]      # <- list-of-list structure

    # ASC
    asc_list = ([asc_raw] if isinstance(asc_raw, str)
                else list(asc_raw)) if asc_raw else []

    return spec, asc_list

def fit_predict(raw_df, meta_name, verbose=0):
    meta = load_meta(meta_name)
    long_df, pid = to_long(raw_df, meta, debug=True)   # 首次调试可启用
    spec, asc_list = build_spec(meta, long_df)

    # ---- 清洗 ----
    spec_cols = list(spec.keys())
    long_df = clean_for_pylogit(long_df, spec_cols)

    # ---- 建模 ----
    model = pylogit.create_choice_model(
        data=long_df,
        alt_id_col="alt_id",
        obs_id_col=pid,
        choice_col="choice",
        specification=spec,
        model_type="MNL"
    )

    # 在调用 fit_mle 之前
    num_params = model.design.shape[1]  # 设计矩阵列数 = 参数数
    init_vals = np.zeros(num_params)  # 全 0 初始值

    model.fit_mle(init_vals=init_vals,
                  print_res=bool(verbose))

    prob_mat = model.predicted_probabilities
    # 折返到 individual 概率
    prob_ind = (
        prob_mat
        .assign(obs_id=long_df[pid])
        .groupby("obs_id")["prob_choice"].max()
    )
    y_pred = (
        prob_mat.loc[prob_mat.groupby("obs_id")["prob_choice"].idxmax(), "alt_id"]
        .reset_index(drop=True)
    )
    return prob_ind.values, y_pred.values
