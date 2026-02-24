import torch
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def compute_pair_stats(i, j, y_series, x_series, max_lag=1):
    if np.var(y_series) < 1e-6 or np.var(x_series) < 1e-6: return None
    corr_val = np.corrcoef(x_series[:-1], y_series[1:])[0, 1]
    if abs(corr_val) < 0.25: return None  # 稍微放宽一点门槛

    data = np.stack([y_series, x_series], axis=1)
    try:
        gc_res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        p_val = gc_res[1][0]['ssr_ftest'][1]
        if p_val < 0.05: return (i, j, p_val, abs(corr_val))
    except:
        pass
    return None


def main():
    print("加载超图序列...")
    hypergraph_seq = torch.load('hypergraph_seq.pt')

    # 动态确定节点数
    all_nodes = set()
    for data in hypergraph_seq:
        all_nodes.update(data['H'][0].numpy().tolist())
    NUM_NODES = max(all_nodes) + 1
    TIME_STEPS = len(hypergraph_seq)

    # 构建时间序列矩阵
    ts_matrix = np.zeros((TIME_STEPS, NUM_NODES))
    for t_idx, data in enumerate(hypergraph_seq):
        nodes = data['H'][0].numpy()
        weights = data['weights'].numpy()
        edge_indices = data['H'][1].numpy()

        # 聚合：将每条边的权重加到对应的节点上
        df_temp = pd.DataFrame({'node': nodes, 'weight': weights[edge_indices]})
        node_sums = df_temp.groupby('node')['weight'].sum()
        for nid, val in node_sums.items():
            ts_matrix[t_idx, int(nid)] = val

    # 一阶差分
    ts_diff = np.diff(ts_matrix, axis=0)
    ts_diff += np.random.normal(0, 1e-9, ts_diff.shape)  # 防止除零

    # 并行计算
    print("开始 Granger 并行计算...")
    tasks = [(i, j) for i in range(NUM_NODES) for j in range(NUM_NODES) if i != j]

    raw_results = Parallel(n_jobs=-1)(
        delayed(compute_pair_stats)(i, j, ts_diff[:, j], ts_diff[:, i])
        for i, j in tqdm(tasks)
    )
    valid_results = [r for r in raw_results if r is not None]

    if not valid_results:
        print("警告：没有找到显著边！将生成全零先验。")
        adj_prior = np.zeros((NUM_NODES, NUM_NODES))
    else:
        # FDR 校正
        p_values = [r[2] for r in valid_results]
        reject, _, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')  # 放宽到 0.05

        adj_prior = np.zeros((NUM_NODES, NUM_NODES))
        for idx, should_reject in enumerate(reject):
            if should_reject:
                i, j, raw_p, corr = valid_results[idx]
                # 权重 = (1-p) * corr，这正是我们需要的 Soft Prior
                weight = (1.0 - raw_p) * corr
                adj_prior[i, j] = weight

    print(f"保留边数: {np.count_nonzero(adj_prior)}")
    torch.save(torch.tensor(adj_prior, dtype=torch.float32), 'granger_prior.pt')
    print("已保存: granger_prior.pt")


if __name__ == "__main__":
    main()