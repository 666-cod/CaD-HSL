import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import os
from collections import Counter

# 过滤警告 & 设置绘图字体
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ================= 1. 模型结构 (必须与训练一致) =================
class HypergraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, H, edge_weights):
        x = self.linear(x)
        edge_deg = H.sum(dim=1, keepdim=True).clamp(min=1.0)
        edge_feat = torch.matmul(H.transpose(1, 2), x) / edge_deg.transpose(1, 2)
        edge_feat = edge_feat * edge_weights.unsqueeze(-1)
        node_deg = H.sum(dim=2, keepdim=True).clamp(min=1.0)
        x_new = torch.matmul(H, edge_feat) / node_deg
        return self.norm(F.elu(x_new))


class CausalStructureLearner(nn.Module):
    def __init__(self, num_nodes, prior_matrix):
        super().__init__()
        prior_logits = torch.ones_like(prior_matrix) * -5.0
        mask = prior_matrix > 1e-4
        prior_logits[mask] = 1.0
        self.register_buffer('prior_logits', prior_logits)
        self.adj_delta = nn.Parameter(torch.zeros(num_nodes, num_nodes))

    def forward(self):
        adj = torch.sigmoid(self.prior_logits + self.adj_delta)
        return adj * (adj > 0.2).float()


class CaD_HSL_Model(nn.Module):
    def __init__(self, config, prior_matrix):
        super().__init__()
        self.node_emb = nn.Embedding(config['num_nodes'], config['embed_dim'])
        self.hg_conv1 = HypergraphConv(config['embed_dim'], config['hidden_dim'])
        self.hg_conv2 = HypergraphConv(config['hidden_dim'], config['hidden_dim'])
        self.causal_learner = CausalStructureLearner(config['num_nodes'], prior_matrix)
        self.gcn_lin = nn.Linear(config['embed_dim'], config['hidden_dim'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=config['hidden_dim'] * 2, nhead=4, dim_feedforward=64,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head_cls = nn.Linear(config['hidden_dim'] * 2, 1)
        self.head_reg = nn.Linear(config['hidden_dim'] * 2, 1)

    def forward(self, x, H, W): return None, None, self.causal_learner()


# ================= 2. 数据加载 =================
CONFIG = {'embed_dim': 32, 'hidden_dim': 32}


def load_data():
    print(">>> [1/5] 加载数据...")
    with open('dictionaries.pkl', 'rb') as f:
        dicts = pickle.load(f)
    id2tech = dicts['id2tech']
    CONFIG['num_nodes'] = len(id2tech)

    seq = torch.load('hypergraph_seq.pt', weights_only=False)

    ts_matrix = np.zeros((len(seq), len(id2tech)))
    for t, item in enumerate(seq):
        if item['H'].numel() > 0:
            edge_vals = item['weights'][item['H'][1]]
            df_tmp = pd.DataFrame({'n': item['H'][0].numpy(), 'w': edge_vals.numpy()})
            for n, val in df_tmp.groupby('n')['w'].sum().items():
                if n < len(id2tech): ts_matrix[t, int(n)] = val

    scaler = MinMaxScaler()
    ts_norm = scaler.fit_transform(ts_matrix)
    df_norm = pd.DataFrame(ts_norm, columns=[id2tech[i] for i in range(len(id2tech))])

    try:
        dates = pd.date_range(start='2012-01-01', periods=len(df_norm), freq='QE')
    except:
        dates = pd.date_range(start='2012-01-01', periods=len(df_norm), freq='Q')

    df_norm.index = dates
    print(f"    数据范围: {dates[0].date()} 到 {dates[-1].date()} (共 {len(dates)} 季度)")

    return df_norm, scaler, id2tech


# ================= 3. 智能驱动提取 =================
def get_refined_drivers(id2tech, threshold=0.75):
    print(f">>> [2/5] 提取并优化因果结构 (Threshold > {threshold})...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prior = torch.load('granger_prior.pt', map_location=device, weights_only=False)

    model = CaD_HSL_Model(CONFIG, prior).to(device)
    try:
        model.load_state_dict(torch.load('cad_hsl_model.pth', map_location=device, weights_only=True))
    except:
        model.load_state_dict(torch.load('cad_hsl_model.pth', map_location=device, weights_only=False))
    model.eval()

    # 1. 获取原始邻接矩阵
    adj = model.causal_learner().detach().cpu().numpy()

    # 2. Hub 过滤
    temp_driver_counts = []
    num_techs = len(id2tech)

    for i in range(num_techs):
        incoming = adj[:, i]
        idxs = np.where(incoming > threshold)[0]
        temp_driver_counts.extend(idxs.tolist())

    driver_counter = Counter(temp_driver_counts)

    # 如果指向超过 10% 的节点，视为通用噪声
    hub_limit = max(5, int(num_techs * 0.10))
    blacklist_ids = {k for k, v in driver_counter.items() if v > hub_limit}

    blacklist_names = [id2tech[i] for i in blacklist_ids]
    print(f"    ⚠️ 屏蔽通用噪声节点 (Hubs): {len(blacklist_ids)} 个")
    if len(blacklist_names) > 0:
        print(f"       示例: {blacklist_names[:3]} ...")

    # 3. 提取
    drivers_map = {}
    valid_links = 0

    for i in range(num_techs):
        target = id2tech[i]
        incoming = adj[:, i]

        idxs = np.where(incoming > threshold)[0]
        idxs = idxs[np.argsort(incoming[idxs])[::-1]]

        refined_drivers = []
        for src_idx in idxs:
            if src_idx != i and src_idx not in blacklist_ids:
                refined_drivers.append(id2tech[src_idx])

        # 只取 Top 3
        final_drivers = refined_drivers[:3]

        if final_drivers:
            drivers_map[target] = final_drivers
            valid_links += len(final_drivers)

    print(f"    ✅ 有效因果连线: {valid_links} 条")
    return drivers_map


# ================= 4. 回测逻辑 =================
def run_evaluation_full(df, scaler, target_col, drivers, id2tech):
    df_feat = pd.DataFrame(index=df.index)
    # 增加 Diff 特征，对齐 Granger 逻辑
    for l in [1, 2, 3]: df_feat[f'Self_Lag{l}'] = df[target_col].shift(l)
    for d in drivers:
        if d in df.columns:
            df_feat[f'D_{d}_Lag1'] = df[d].shift(1)
            df_feat[f'D_{d}_Diff1'] = df[d].diff().shift(1)

    df_feat['Y'] = df[target_col]
    df_feat = df_feat.dropna()

    test_steps = 4
    if len(df_feat) < 8: return None
    train, test = df_feat.iloc[:-test_steps], df_feat.iloc[-test_steps:]

    # 训练
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=1, random_state=42)
    model.fit(train.drop('Y', axis=1), train['Y'])
    preds_norm = model.predict(test.drop('Y', axis=1))

    def inverse_vec(vec_1d):
        mat = np.zeros((len(vec_1d), len(id2tech)))
        target_idx = [k for k, v in id2tech.items() if v == target_col][0]
        mat[:, target_idx] = vec_1d
        return scaler.inverse_transform(mat)[:, target_idx]

    y_full_real = inverse_vec(df[target_col].values)
    y_test_real = inverse_vec(test['Y'].values)
    y_pred_real = inverse_vec(preds_norm)

    mae = mean_absolute_error(y_test_real, y_pred_real)
    mse = mean_squared_error(y_test_real, y_pred_real)
    mape = np.mean(np.abs((y_test_real - y_pred_real) / (y_test_real + 1e-6))) * 100

    return {
        'mae': mae, 'mse': mse, 'mape': mape,
        'y_full_real': y_full_real,
        'y_test_pred': y_pred_real,
        'test_dates': test.index,
        'full_dates': df.index
    }


def main():
    df, scaler, id2tech = load_data()
    # 提高阈值到 0.75，因为模型训练得很充分
    drivers_map = get_refined_drivers(id2tech, threshold=0.75)

    all_targets = [id2tech[i] for i in range(len(id2tech))]

    print(f"\n>>> [3/5] 开始全量回测 ({len(all_targets)} 个技术)...")
    results_data = []
    pdf_filename = "final_tech_forecasts.pdf"

    print(f">>> [4/5] 正在绘图至 {pdf_filename} ...")

    with PdfPages(pdf_filename) as pdf:
        for i, t in enumerate(all_targets):
            if (i + 1) % 50 == 0: print(f"    已处理 {i + 1}/{len(all_targets)} ...")

            res_base = run_evaluation_full(df, scaler, t, [], id2tech)
            if not res_base: continue

            drivers = drivers_map.get(t, [])
            res_causal = run_evaluation_full(df, scaler, t, drivers, id2tech)
            if not res_causal: continue

            imp_mae = res_base['mae'] - res_causal['mae']

            results_data.append({
                'Tech': t,
                'Drivers': ",".join(drivers),
                'Base_MAE': res_base['mae'], 'Causal_MAE': res_causal['mae'],
                'Imp_MAE': imp_mae,
                'Base_MAPE': res_base['mape'], 'Causal_MAPE': res_causal['mape'],
                'Base_MSE': res_base['mse'], 'Causal_MSE': res_causal['mse']
            })

            # 只为有显著提升的画图
            if len(drivers) > 0 and imp_mae > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(res_causal['full_dates'], res_causal['y_full_real'], 'k-', label='Ground Truth', linewidth=1.5)
                test_dates = res_causal['test_dates']
                plt.plot(test_dates, res_base['y_test_pred'], 'r--s', label=f'Base (MAE:{res_base["mae"]:.0f})',
                         alpha=0.7)
                plt.plot(test_dates, res_causal['y_test_pred'], 'g-^', label=f'Causal (MAE:{res_causal["mae"]:.0f})',
                         linewidth=2.5)

                drivers_str = "\n".join(drivers)
                plt.title(f"{t}\nDrivers:\n{drivers_str}", fontsize=10)
                plt.xlabel("Time");
                plt.ylabel("Heat")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout()
                pdf.savefig()
                plt.close()

    res_df = pd.DataFrame(results_data)
    # 覆盖之前的 csv，供 app 读取
    res_df.to_csv("refined_tech_metrics.csv", index=False)

    print(f"\n>>> [5/5] 完成！")

    # 打印总体指标
    avg_imp = res_df['Imp_MAE'].mean()
    avg_base = res_df['Base_MAE'].mean()
    print(f"\n📊 总体平均 MAE: Base={avg_base:.0f} vs Ours={avg_base - avg_imp:.0f}")
    print(f"🚀 平均降低: {avg_imp:.0f} ({(avg_imp / avg_base) * 100:.1f}%)")

    print("\nTop 5 真实有效提升:")
    print(res_df.sort_values('Imp_MAE', ascending=False).head(5)[['Tech', 'Imp_MAE', 'Drivers']])


if __name__ == "__main__":
    main()