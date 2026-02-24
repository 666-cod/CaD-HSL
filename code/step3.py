import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import warnings

# 过滤 PyTorch 的安全警告
warnings.filterwarnings("ignore", category=FutureWarning)

# ================= 配置 =================
CONFIG = {
    'num_nodes': 293,
    'embed_dim': 32,
    'hidden_dim': 32,
    'history_len': 4,
    'batch_size': 8,  # 稍微调大 Batch，数据已归一化，显存够用
    'grad_accum_steps': 1,  # 归一化后收敛快，可以减少累积步数
    'epochs': 500,
    'lr': 0.002,  # 归一化后可以稍微加大一点学习率
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


# ================= 模型定义 (保持不变) =================
class HypergraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, H, edge_weights):
        x = self.linear(x)
        edge_deg = H.sum(dim=1, keepdim=True).clamp(min=1.0)
        edge_feat = torch.matmul(H.transpose(1, 2), x)
        edge_feat = edge_feat / edge_deg.transpose(1, 2)
        edge_feat = edge_feat * edge_weights.unsqueeze(-1)
        node_deg = H.sum(dim=2, keepdim=True).clamp(min=1.0)
        x_new = torch.matmul(H, edge_feat)
        x_new = x_new / node_deg
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
        mask = (adj > 0.2).float()
        return adj * mask


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

    def forward(self, x_seq_indices, H_seq, W_seq):
        B, T, N, _ = H_seq.shape
        base_emb = self.node_emb.weight.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        batch_out = []
        curr_adj = self.causal_learner()
        deg = curr_adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        norm_adj = curr_adj / deg

        for t in range(T):
            x_t = base_emb[:, t, :, :]
            H_t = H_seq[:, t, :, :]
            W_t = W_seq[:, t, :]
            h_hg = self.hg_conv1(x_t, H_t, W_t)
            h_hg = self.hg_conv2(h_hg, H_t, W_t)
            h_causal = torch.matmul(norm_adj, x_t)
            h_causal = F.relu(self.gcn_lin(h_causal))
            h_fused = torch.cat([h_hg, h_causal], dim=-1)
            batch_out.append(h_fused)

        seq_feat = torch.stack(batch_out, dim=1)
        seq_feat_flat = seq_feat.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        encoded = self.transformer(seq_feat_flat)
        final_feat = encoded[:, -1, :]
        pred_cls = self.head_cls(final_feat).reshape(B, N)
        pred_reg = self.head_reg(final_feat).reshape(B, N)
        return pred_cls, pred_reg, curr_adj


# ================= 数据集 (增加归一化) =================
class TechTrendDataset(Dataset):
    def __init__(self, hypergraph_seq_path, history_len=4, num_nodes=293):
        raw_data = torch.load(hypergraph_seq_path, weights_only=False)
        self.timeline_data = []

        max_edges_limit = 2000
        global_max_val = 0.0  # 用于归一化目标值

        print("正在预处理并计算全局最大值...")
        for item in raw_data:
            indices = item['H']
            weights = item['weights']
            num_edges = item['num_edges']

            # 1. 计算 Regression Target
            node_vals = torch.zeros(num_nodes)
            if indices.numel() > 0:
                edge_vals_expanded = weights[indices[1]]
                df_tmp = pd.DataFrame({'n': indices[0].numpy(), 'w': edge_vals_expanded.numpy()})
                sums = df_tmp.groupby('n')['w'].sum()
                for n, val in sums.items():
                    if n < num_nodes:
                        node_vals[int(n)] = val

            # 记录最大值
            curr_max = node_vals.max().item()
            if curr_max > global_max_val:
                global_max_val = curr_max

            # 2. 截断逻辑
            if num_edges > max_edges_limit:
                weights = weights[:max_edges_limit]
                mask = indices[1] < max_edges_limit
                indices = indices[:, mask]
                num_edges = max_edges_limit

            # 3. Dense Matrix
            H = torch.zeros(num_nodes, num_edges)
            if indices.numel() > 0:
                H[indices[0], indices[1]] = 1.0

            self.timeline_data.append({'H': H, 'W': weights, 'node_vals': node_vals})

        # 【关键修正】执行全局归一化
        print(f"Regression Target Global Max: {global_max_val:.4f} (即将执行归一化)")
        if global_max_val > 1e-6:
            for item in self.timeline_data:
                item['node_vals'] = item['node_vals'] / global_max_val

        self.samples = []
        for i in range(len(self.timeline_data) - history_len):
            self.samples.append(range(i, i + history_len + 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch_indices):
    global GLOBAL_DATA
    batch_H, batch_W, batch_Y_cls, batch_Y_reg = [], [], [], []

    max_m = 0
    for ranges in batch_indices:
        for t in ranges[:-1]:
            max_m = max(max_m, GLOBAL_DATA[t]['H'].shape[1])
    max_m = min(max_m, 2000)

    for ranges in batch_indices:
        seq_H, seq_W = [], []

        # 获取输入序列
        for t in ranges[:-1]:
            h_raw = GLOBAL_DATA[t]['H']
            w_raw = GLOBAL_DATA[t]['W']
            curr_m = h_raw.shape[1]
            eff_m = min(curr_m, max_m)

            h_pad = torch.zeros(CONFIG['num_nodes'], max_m)
            h_pad[:, :eff_m] = h_raw[:, :eff_m]

            w_pad = torch.zeros(max_m)
            if len(w_raw) > 0:
                w_pad[:eff_m] = w_raw[:eff_m]

            seq_H.append(h_pad)
            seq_W.append(w_pad)

        # --- 关键修改开始 ---
        # 目标时刻 t
        target_t = ranges[-1]
        # 前一时刻 t-1
        prev_t = ranges[-2]

        target_vals = GLOBAL_DATA[target_t]['node_vals']
        prev_vals = GLOBAL_DATA[prev_t]['node_vals']

        # 1. 回归目标：依然预测具体数值 (Log Amount)
        batch_Y_reg.append(target_vals)

        # 2. 分类目标：改为“趋势预测” (Trend Prediction)
        # 如果 本期值 > 上期值 * 1.05 (涨幅超过5%才算涨，防止噪声)，标记为 1，否则 0
        # 这样模型就必须学习“什么导致了增长”
        growth_label = (target_vals > (prev_vals * 1.05)).float()
        batch_Y_cls.append(growth_label)
        # --- 关键修改结束 ---

        batch_H.append(torch.stack(seq_H))
        batch_W.append(torch.stack(seq_W))

    return torch.stack(batch_H), torch.stack(batch_W), torch.stack(batch_Y_cls), torch.stack(batch_Y_reg)

# ================= 训练循环 =================
def train():
    global GLOBAL_DATA
    prior = torch.load('granger_prior.pt', weights_only=False)
    CONFIG['num_nodes'] = prior.shape[0]
    print(f"检测到节点数: {CONFIG['num_nodes']}")

    dataset = TechTrendDataset('hypergraph_seq.pt', history_len=CONFIG['history_len'], num_nodes=CONFIG['num_nodes'])
    GLOBAL_DATA = dataset.timeline_data

    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)

    model = CaD_HSL_Model(CONFIG, prior).to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    crit_cls = nn.BCEWithLogitsLoss()
    crit_reg = nn.MSELoss()

    print(">>> 开始训练 (Normalized Mode)...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0

        for H, W, Y_cls, Y_reg in loader:
            H, W = H.to(CONFIG['device']), W.to(CONFIG['device'])
            Y_cls, Y_reg = Y_cls.to(CONFIG['device']), Y_reg.to(CONFIG['device'])

            if W.max() > 0: W = W / W.max()

            pred_cls, pred_reg, _ = model(None, H, W)

            loss_c = crit_cls(pred_cls, Y_cls)
            loss_r = crit_reg(pred_reg, Y_reg)

            # 因为 loss_r 现在很小 (0.00x)，提高它的权重让模型重视回归
            loss = loss_c + 10.0 * loss_r

            loss = loss / CONFIG['grad_accum_steps']
            loss.backward()

            if (epoch * len(loader) + 1) % CONFIG['grad_accum_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * CONFIG['grad_accum_steps']

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1} | Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), 'cad_hsl_model_v2.pth')
    print("训练完成！")


if __name__ == "__main__":
    train()