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

# 过滤警告 & 设置绘图字体
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ================= 1. 模型结构 (保持不变，确保独立运行) =================
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

    # 原始数值矩阵 (Log金额)
    ts_matrix = np.zeros((len(seq), len(id2tech)))
    for t, item in enumerate(seq):
        if item['H'].numel() > 0:
            edge_vals = item['weights'][item['H'][1]]
            df_tmp = pd.DataFrame({'n': item['H'][0].numpy(), 'w': edge_vals.numpy()})
            for n, val in df_tmp.groupby('n')['w'].sum().items():
                if n < len(id2tech): ts_matrix[t, int(n)] = val

    # 保存 Scaler 以便反归一化
    scaler = MinMaxScaler()
    ts_norm = scaler.fit_transform(ts_matrix)
    df_norm = pd.DataFrame(ts_norm, columns=[id2tech[i] for i in range(len(id2tech))])

    # 时间索引 (自动处理)
    try:
        dates = pd.date_range(start='2012-01-01', periods=len(df_norm), freq='QE')
    except:
        dates = pd.date_range(start='2012-01-01', periods=len(df_norm), freq='Q')

    df_norm.index = dates
    print(f"    数据范围: {dates[0].date()} 到 {dates[-1].date()} (共 {len(dates)} 季度)")

    return df_norm, scaler, id2tech


# ================= 3. 提取驱动 =================
def get_strong_drivers(id2tech, threshold=0.7):
    print(f">>> [2/5] 提取强因果 (Threshold > {threshold})...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prior = torch.load('granger_prior.pt', map_location=device, weights_only=False)

    model = CaD_HSL_Model(CONFIG, prior).to(device)
    try:
        model.load_state_dict(torch.load('cad_hsl_model_v3.pth.pth', map_location=device, weights_only=True))
    except:
        model.load_state_dict(torch.load('cad_hsl_model_v3.pth', map_location=device, weights_only=False))
    model.eval()

    adj = model.causal_learner().detach().cpu().numpy()
    drivers_map = {}

    for i in range(len(id2tech)):
        target = id2tech[i]
        incoming = adj[:, i]
        idxs = np.where(incoming > threshold)[0]
        idxs = idxs[np.argsort(incoming[idxs])[::-1]]
        drivers = [id2tech[x] for x in idxs if x != i][:3]
        if drivers: drivers_map[target] = drivers

    return drivers_map


# ================= 4. 回测与绘图逻辑 =================
def run_evaluation_full(df, scaler, target_col, drivers, id2tech):
    # 1. 构建特征
    df_feat = pd.DataFrame(index=df.index)
    for l in [1, 2, 3]: df_feat[f'Self_Lag{l}'] = df[target_col].shift(l)
    for d in drivers:
        if d in df.columns:
            df_feat[f'D_{d}_Lag1'] = df[d].shift(1)
            df_feat[f'D_{d}_Diff1'] = df[d].diff().shift(1)

    df_feat['Y'] = df[target_col]
    df_feat = df_feat.dropna()

    # 2. 划分 (训练集: 2012-2023, 测试集: 2024)
    test_steps = 4
    if len(df_feat) < 8: return None
    train, test = df_feat.iloc[:-test_steps], df_feat.iloc[-test_steps:]

    # 3. 训练
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=1, random_state=42)
    model.fit(train.drop('Y', axis=1), train['Y'])
    preds_norm = model.predict(test.drop('Y', axis=1))

    # 4. 反归一化工具函数
    def inverse_vec(vec_1d):
        mat = np.zeros((len(vec_1d), len(id2tech)))
        target_idx = [k for k, v in id2tech.items() if v == target_col][0]
        mat[:, target_idx] = vec_1d
        return scaler.inverse_transform(mat)[:, target_idx]

    # 获取全历史真实值 (用于绘图)
    y_full_real = inverse_vec(df[target_col].values)

    # 获取测试集真实值与预测值 (用于计算指标)
    y_test_real = inverse_vec(test['Y'].values)
    y_pred_real = inverse_vec(preds_norm)

    # 5. 计算指标
    mae = mean_absolute_error(y_test_real, y_pred_real)
    mse = mean_squared_error(y_test_real, y_pred_real)
    # MAPE (避免除零)
    mape = np.mean(np.abs((y_test_real - y_pred_real) / (y_test_real + 1e-6))) * 100

    return {
        'mae': mae, 'mse': mse, 'mape': mape,
        'y_full_real': y_full_real,  # 全历史真值
        'y_test_pred': y_pred_real,  # 测试集预测
        'test_dates': test.index,
        'full_dates': df.index
    }


def main():
    df, scaler, id2tech = load_data()
    drivers_map = get_strong_drivers(id2tech, threshold=0.7)

    # 获取所有技术
    all_targets = [id2tech[i] for i in range(len(id2tech))]

    print(f"\n>>> [3/5] 开始全量回测 ({len(all_targets)} 个技术)...")
    print("    这将生成: 1. PDF图表文件  2. CSV指标汇总")

    results_data = []
    pdf_filename = "all_tech_forecasts_2012_2024_v3.pdf"

    print(f">>> [4/5] 正在处理并绘图至 {pdf_filename} ...")

    # 使用 PdfPages 打开 PDF 文件一次，循环写入
    with PdfPages(pdf_filename) as pdf:
        for i, t in enumerate(all_targets):
            # 简单的进度打印
            if (i + 1) % 50 == 0: print(f"    已处理 {i + 1}/{len(all_targets)} ...")

            # 1. Base Evaluation
            res_base = run_evaluation_full(df, scaler, t, [], id2tech)
            if not res_base: continue

            # 2. Causal Evaluation
            drivers = drivers_map.get(t, [])
            res_causal = run_evaluation_full(df, scaler, t, drivers, id2tech)
            if not res_causal: continue

            # 收集指标
            results_data.append({
                'Tech': t,
                'Drivers': ",".join(drivers),
                'Base_MAE': res_base['mae'], 'Causal_MAE': res_causal['mae'],
                'Imp_MAE': res_base['mae'] - res_causal['mae'],
                'Base_MSE': res_base['mse'], 'Causal_MSE': res_causal['mse'],
                'Base_MAPE': res_base['mape'], 'Causal_MAPE': res_causal['mape']
            })

            # --- 绘图 (2012-2024) ---
            plt.figure(figsize=(10, 6))

            # A. 绘制全历史真实值 (黑色实线)
            plt.plot(res_causal['full_dates'], res_causal['y_full_real'],
                     color='black', label='Ground Truth (2012-2024)', linewidth=1.5)

            # B. 绘制测试集预测值 (红/绿线)
            test_dates = res_causal['test_dates']
            plt.plot(test_dates, res_base['y_test_pred'],
                     'r--s', label=f'Base (MAPE: {res_base["mape"]:.1f}%)', alpha=0.7)
            plt.plot(test_dates, res_causal['y_test_pred'],
                     'g-^', label=f'Causal (MAPE: {res_causal["mape"]:.1f}%)', linewidth=2.5)

            # 标题与标注
            title_str = f"Tech: {t}\nMAE Imp: {res_base['mae'] - res_causal['mae']:.2f} | Drivers: {len(drivers)}"
            plt.title(title_str, fontsize=12)
            plt.xlabel("Time (Quarterly)")
            plt.ylabel("Heat (Log Amount)")
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()

            # 保存当前页到 PDF
            pdf.savefig()
            plt.close()  # 关闭画布释放内存

    # 保存 CSV
    res_df = pd.DataFrame(results_data)
    res_df.to_csv("all_tech_metrics_v3.csv", index=False)

    print(f"\n>>> [5/5] 全部完成！")
    print(f"    - 图表文件: {pdf_filename} (包含 {len(results_data)} 页)")
    print(f"    - 指标文件: all_tech_metrics.csv")

    # 打印前 5 名提升
    print("\n>>> MAE 提升最大的 Top 5 技术:")
    print(res_df.sort_values('Imp_MAE', ascending=False).head(5)[['Tech', 'Imp_MAE', 'Base_MAE', 'Causal_MAE']])


if __name__ == "__main__":
    main()