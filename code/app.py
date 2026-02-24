import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import tempfile
import warnings
import sys

# ================= 1. 基础配置 =================
warnings.filterwarnings("ignore")

# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


# ================= 页面配置 =================
st.set_page_config(
    page_title="CaD-HSL 技术趋势平台",
    page_icon="🔮",
    layout="wide"
)


# ================= 模型结构 (保持一致) =================
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


# ================= 数据加载 =================
@st.cache_resource
def load_all_data():
    try:
        # 1. 字典
        with open(get_path('dictionaries.pkl'), 'rb') as f:
            dicts = pickle.load(f)
        id2tech = dicts['id2tech']
        num_nodes = len(id2tech)

        # 2. 原始序列
        seq = torch.load(get_path('hypergraph_seq.pt'), map_location='cpu', weights_only=False)
        ts_matrix = np.zeros((len(seq), num_nodes))
        for t, item in enumerate(seq):
            if item['H'].numel() > 0:
                edge_vals = item['weights'][item['H'][1]]
                df_tmp = pd.DataFrame({'n': item['H'][0].numpy(), 'w': edge_vals.numpy()})
                for n, val in df_tmp.groupby('n')['w'].sum().items():
                    if n < num_nodes: ts_matrix[t, int(n)] = val

        scaler = MinMaxScaler()
        ts_norm = scaler.fit_transform(ts_matrix)
        df_norm = pd.DataFrame(ts_norm, columns=[id2tech[i] for i in range(num_nodes)])
        df_real = pd.DataFrame(ts_matrix, columns=[id2tech[i] for i in range(num_nodes)])

        # 频率兼容修复
        try:
            dates = pd.date_range(end='2024-12-31', periods=len(df_norm), freq='QE')
        except:
            dates = pd.date_range(end='2024-12-31', periods=len(df_norm), freq='Q')

        df_norm.index = dates
        df_real.index = dates

        # 3. 模型
        device = 'cpu'
        prior = torch.load(get_path('granger_prior.pt'), map_location=device, weights_only=False)
        config = {'num_nodes': num_nodes, 'embed_dim': 32, 'hidden_dim': 32}
        model = CaD_HSL_Model(config, prior).to(device)

        model_path = get_path('cad_hsl_model.pth')
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        except:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

        model.eval()
        adj_matrix = model.causal_learner().detach().cpu().numpy()

        # 4. 指标
        metrics_df = pd.read_csv(get_path('refined_tech_metrics.csv'))

        return df_norm, df_real, adj_matrix, id2tech, scaler, metrics_df

    except FileNotFoundError as e:
        st.error(f"❌ 找不到文件: {e.filename}。请确保所有数据文件都在 {BASE_DIR} 目录下。")
        st.stop()
    except Exception as e:
        st.error(f"❌ 数据加载未知错误: {str(e)}")
        st.stop()


# 加载数据
with st.spinner("正在初始化数据引擎..."):
    df_norm, df_real, adj_matrix, id2tech, scaler, metrics_df = load_all_data()


# ================= 辅助函数：图构建与绘制 (关键优化) =================
def build_networkx_graph(adj, id2tech, threshold=0.7):
    G = nx.DiGraph()
    rows, cols = np.where(adj > threshold)
    for r, c in zip(rows, cols):
        if r == c: continue
        weight = float(adj[r, c])
        src = id2tech[r]
        dst = id2tech[c]
        G.add_edge(src, dst, weight=weight, title=f"{weight:.2f}")
    return G


def plot_pyvis(G, height="600px", select_node=None, mode="full"):
    """
    mode: 'full' (全景图) | 'ego' (二阶小图)
    """
    if len(G.nodes) == 0:
        return "<div>图为空</div>"

    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    net.from_nx(G)

    # 设置节点样式
    for node in net.nodes:
        if select_node and node['id'] == select_node:
            node['color'] = '#FF5733'  # 选中红
            node['size'] = 40
            node['label'] = f"★ {node['id']}"
        else:
            node['color'] = '#4B8BBE'  # 普通蓝
            node['size'] = 20

    # 【关键修改】物理引擎配置：增加阻尼，快速静止
    if mode == "full":
        # 全景图：使用 ForceAtlas2Based，但增加 damping
        net.set_options("""
        var options = {
          "nodes": { "font": { "size": 16, "strokeWidth": 2, "strokeColor": "white" } },
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08,
              "damping": 0.4 
            },
            "minVelocity": 0.75, 
            "solver": "forceAtlas2Based",
            "stabilization": {
              "enabled": true,
              "iterations": 200, 
              "updateInterval": 25,
              "fit": true
            }
          },
          "interaction": { "hover": true, "navigationButtons": true }
        }
        """)
    else:
        # 二阶图：使用 BarnesHut，超高阻尼，避免乱动
        net.set_options("""
        var options = {
          "nodes": { "font": { "size": 14 } },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -3000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.5,
              "avoidOverlap": 1
            },
            "minVelocity": 0.75,
            "solver": "barnesHut",
            "stabilization": {
              "enabled": true,
              "iterations": 200,
              "fit": true
            }
          },
          "interaction": { "hover": true, "navigationButtons": true }
        }
        """)

    # 生成 HTML (Windows 兼容)
    try:
        fd, path = tempfile.mkstemp(suffix=".html")
        os.close(fd)
        net.save_graph(path)
        with open(path, 'r', encoding='utf-8') as f:
            html = f.read()
        os.remove(path)
        return html
    except Exception as e:
        return f"<div>绘图错误: {e}</div>"


# ================= 主界面 =================
st.sidebar.title("🔮 CaD-HSL 驾驶舱")
page = st.sidebar.radio("选择功能模块",
                        ["全景技术关联图", "特定技术二阶图", "趋势预测与对比", "模型总指标"])

if page == "全景技术关联图":
    st.title("🌐 全景技术因果关联图")
    col1, col2 = st.columns([3, 1])
    with col2:
        year = st.slider("选择年份", 2012, 2024, 2024)
        threshold = st.slider("阈值", 0.5, 0.95, 0.75, 0.05)

    # 1. 构建全图
    G_full = build_networkx_graph(adj_matrix, id2tech, threshold)

    # 2. 根据年份过滤活跃节点
    target_date = str(year)
    mask = df_real.index.astype(str).str.contains(target_date)
    if mask.any():
        yearly_data = df_real[mask].sum()
        active_techs = set(yearly_data[yearly_data > 0].index)
        sub_nodes = [n for n in G_full.nodes if n in active_techs]
        G_view = G_full.subgraph(sub_nodes)
    else:
        G_view = nx.DiGraph()

    with col1:
        st.markdown(f"**节点:** {len(G_view.nodes)} | **连线:** {len(G_view.edges)}")
        if len(G_view.nodes) > 0:
            html = plot_pyvis(G_view, height="600px", mode="full")
            st.components.v1.html(html, height=610)
        else:
            st.warning(f"{year} 年无满足条件的数据。")

elif page == "特定技术二阶图":
    st.title("🕸️ 特定技术二阶关联图")
    st.markdown("仅展示选中技术及其 **上游（驱动方）** 和 **下游（被驱动方）**。")

    col1, col2 = st.columns([1, 3])
    with col1:
        tech = st.selectbox("核心技术", list(id2tech.values()))
        radius = st.slider("关联层级 (Hop)", 1, 2, 1)
        thresh = st.slider("连接强度阈值", 0.5, 0.95, 0.7)

    with col2:
        G_full = build_networkx_graph(adj_matrix, id2tech, thresh)

        if tech in G_full.nodes:
            # 提取 Ego Graph
            G_ego = nx.ego_graph(G_full, tech, radius=radius)
            # 纯净子图
            G_viz = nx.DiGraph()
            G_viz.add_nodes_from(G_ego.nodes(data=True))
            G_viz.add_edges_from(G_ego.edges(data=True))

            st.markdown(f"**{tech}** 的 {radius} 阶邻居网络")
            html = plot_pyvis(G_viz, height="600px", select_node=tech, mode="ego")
            st.components.v1.html(html, height=610)
        else:
            st.info(f"技术 **{tech}** 在当前阈值 ({thresh}) 下没有强关联节点。")

elif page == "趋势预测与对比":
    st.title("📈 趋势预测与归因")
    target = st.selectbox("技术选择", list(id2tech.values()))

    row = metrics_df[metrics_df['Tech'] == target]
    drivers = str(row['Drivers'].values[0]).split(',') if len(row) > 0 and pd.notna(row['Drivers'].values[0]) else []

    if st.button("开始预测 (2024-2025)"):
        # 构造数据
        df_feat = pd.DataFrame(index=df_norm.index)
        df_feat['Y'] = df_norm[target]
        for l in [1, 2, 3]: df_feat[f'S_L{l}'] = df_norm[target].shift(l)
        for d in drivers:
            if d in df_norm.columns:
                df_feat[f'D_{d}_L1'] = df_norm[d].shift(1)
                df_feat[f'D_{d}_D1'] = df_norm[d].diff().shift(1)
        df_feat.dropna(inplace=True)

        train, test = df_feat.iloc[:-4], df_feat.iloc[-4:]

        # 训练
        cols_base = [c for c in df_feat.columns if 'S_L' in c]
        m_base = xgb.XGBRegressor(n_estimators=100, max_depth=3).fit(train[cols_base], train['Y'])
        p_base = m_base.predict(test[cols_base])

        m_causal = xgb.XGBRegressor(n_estimators=100, max_depth=3).fit(train.drop('Y', axis=1), train['Y'])
        p_causal = m_causal.predict(test.drop('Y', axis=1))


        # 还原
        def inv(v):
            m = np.zeros((len(v), len(id2tech)))
            idx = list(id2tech.values()).index(target)
            m[:, idx] = v
            return scaler.inverse_transform(m)[:, idx]


        y_true = df_real[target].iloc[-4:].values
        y_b = inv(p_base)
        y_c = inv(p_causal)

        # 绘图
        fig = go.Figure()
        y_hist = df_real[target]
        fig.add_trace(go.Scatter(x=y_hist.index, y=y_hist.values, name='真实热度', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=test.index, y=y_b, name='Base预测', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=test.index, y=y_c, name='CaD-HSL预测', line=dict(color='green')))
        st.plotly_chart(fig, use_container_width=True)

        mae_b = np.mean(np.abs(y_true - y_b))
        mae_c = np.mean(np.abs(y_true - y_c))
        c1, c2, c3 = st.columns(3)
        c1.metric("Base MAE", f"{mae_b:.0f}")
        c2.metric("Ours MAE", f"{mae_c:.0f}")
        c3.metric("提升", f"{mae_b - mae_c:.0f}", delta_color="normal")

        if drivers:
            st.success(f"核心驱动因子: {', '.join(drivers)}")
        else:
            st.info("无显著外部驱动因子。")

elif page == "模型总指标":
    st.title("🏆 模型总指标")
    st.dataframe(metrics_df.sort_values('Imp_MAE', ascending=False).head(20), use_container_width=True)