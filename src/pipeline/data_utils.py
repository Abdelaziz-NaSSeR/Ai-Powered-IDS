"""
Data preprocessing utilities for CICIDS2018 dataset
"""

import pandas as pd
import numpy as np

from src.pipeline.config import SEED, DATA_PATH, LABEL_COL

# Column renaming map for CICIDS-2018
rename_map = {
    "Dst Port": "dst_port",
    "Protocol": "protocol",
    "Flow Duration": "flow_duration",
    "Flow Byts/s": "flow_byts_s",
    "Flow Pkts/s": "flow_pkts_s",
    "Fwd Pkts/s": "fwd_pkts_s",
    "Bwd Pkts/s": "bwd_pkts_s",
    "Tot Fwd Pkts": "tot_fwd_pkts",
    "Tot Bwd Pkts": "tot_bwd_pkts",
    "TotLen Fwd Pkts": "totlen_fwd_pkts",
    "TotLen Bwd Pkts": "totlen_bwd_pkts",
    "Fwd Pkt Len Max": "fwd_pkt_len_max",
    "Fwd Pkt Len Min": "fwd_pkt_len_min",
    "Fwd Pkt Len Mean": "fwd_pkt_len_mean",
    "Fwd Pkt Len Std": "fwd_pkt_len_std",
    "Bwd Pkt Len Max": "bwd_pkt_len_max",
    "Bwd Pkt Len Min": "bwd_pkt_len_min",
    "Bwd Pkt Len Mean": "bwd_pkt_len_mean",
    "Bwd Pkt Len Std": "bwd_pkt_len_std",
    "Pkt Len Min": "pkt_len_min",
    "Pkt Len Max": "pkt_len_max",
    "Pkt Len Mean": "pkt_len_mean",
    "Pkt Len Std": "pkt_len_std",
    "Pkt Len Var": "pkt_len_var",
    "Pkt Size Avg": "pkt_size_avg",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Tot": "fwd_iat_tot",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Tot": "bwd_iat_tot",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Fwd Header Len": "fwd_header_len",
    "Bwd Header Len": "bwd_header_len",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Fwd URG Flags": "fwd_urg_flags",
    "FIN Flag Cnt": "fin_flag_cnt",
    "SYN Flag Cnt": "syn_flag_cnt",
    "RST Flag Cnt": "rst_flag_cnt",
    "PSH Flag Cnt": "psh_flag_cnt",
    "ACK Flag Cnt": "ack_flag_cnt",
    "URG Flag Cnt": "urg_flag_cnt",
    "ECE Flag Cnt": "ece_flag_cnt",
    "CWE Flag Count": "cwr_flag_count",
    "Down/Up Ratio": "down_up_ratio",
    "Fwd Seg Size Avg": "fwd_seg_size_avg",
    "Bwd Seg Size Avg": "bwd_seg_size_avg",
    "Fwd Seg Size Min": "fwd_seg_size_min",
    "Fwd Act Data Pkts": "fwd_act_data_pkts",
    "Init Fwd Win Byts": "init_fwd_win_byts",
    "Init Bwd Win Byts": "init_bwd_win_byts",
    "Subflow Fwd Pkts": "subflow_fwd_pkts",
    "Subflow Bwd Pkts": "subflow_bwd_pkts",
    "Subflow Fwd Byts": "subflow_fwd_byts",
    "Subflow Bwd Byts": "subflow_bwd_byts",
    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",
    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min",
}

def load_csv(path=DATA_PATH, sample_frac=1.0):
    df = pd.read_csv(path)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=SEED)
    return df.reset_index(drop=True)

def preprocess_flow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare CICIDS-2018 dataset or live captured flows.
    Steps:
        - Rename columns
        - Drop duplicates
        - Replace infinities and invalid numeric values with NaN
        - Fill numeric NaNs with median
        - Drop constant columns
    """
    df = df.copy()

    # Rename columns if many original headers exist
    common_cols = set(rename_map.keys()) & set(df.columns)
    if len(common_cols) > 5:
        df.rename(columns=rename_map, inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fix negative durations / times
    time_cols = [c for c in df.columns if "duration" in c.lower() or "iat" in c.lower() 
                 or "active" in c.lower() or "idle" in c.lower()]
    for col in time_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] < 0, col] = np.nan
            df[col] = df[col].clip(lower=0, upper=1e8)
        except Exception:
            continue

    # Replace -1 placeholders with 0
    for c in ["init_fwd_win_byts", "init_bwd_win_byts"]:
        if c in df.columns:
            df[c] = df[c].replace(-1, 0)

    # Fill numeric NaNs with median
    try:
        numeric_median = df.median(numeric_only=True)
        df.fillna(numeric_median, inplace=True)
    except Exception:
        df.fillna(0, inplace=True)

    # Drop constant columns
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)

    return df
