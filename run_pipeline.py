#!/usr/bin/env python
"""
run_pipeline.py  ·  Step-1: 仅负责
1) 解析命令行参数
2) 调用 ChoiceDataset 读取 (X, y)
3) 打印数据维度与目标列分布
"""
import argparse
from collections import Counter
import pandas as pd
from cagpt_mvp import ChoiceDataset
from baseline import train_predict
from cagpt_mvp.preprocess import load_meta, apply_metadata
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None

# --------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="CAGPT-MVP · baseline experiment runner"
    )
    ap.add_argument(
        "-d", "--dataset",
        choices=[
            "lpmc", "swissmetro", "parking",
            "netherlands", "airline", "optima"
        ],
        default="lpmc",
        help="Biogeme dataset name (default: %(default)s)",
    )
    ap.add_argument(
        "-b", "--base",
        choices=["mnl", "ffnn"],
        default="mnl",
        help="Base model: mnl (Logit) | ffnn (MLP)  (default: %(default)s)",
    )
    ap.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase training verbosity (repeat -v for more)",
    )
    return ap.parse_args()

# --------------------------------------------------
def main():
    args = parse_args()

    # 1) 读取原始 DataFrame（含全部列）
    raw_df = ChoiceDataset(args.dataset).df

    # 2) 应用 metadata → 得到清洗后的 X,y 并打印基本信息
    meta = load_meta(args.dataset)
    X, y, _, _ = apply_metadata(raw_df, meta)

    print(f"<{args.dataset}>  raw shape = {raw_df.shape}  |  features = {X.shape}")
    print(f"Target classes : {Counter(y)}")

    # 3) 训练 + 预测
    prob_pred, y_pred = train_predict(
        raw_df, meta_name=args.dataset,
        model=args.base,
        verbose=args.verbose,
    )

    acc = (y_pred == y).mean()
    print(f"[{args.base.upper()}] Accuracy: {acc:.3f}")

# --------------------------------------------------
if __name__ == "__main__":
    main()
