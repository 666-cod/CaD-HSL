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
import httpx
import re  # 必须导入正则表达式库
from openai import OpenAI
from scipy import stats  # <--- 必须添加这行
# ================= 1. 基础配置 =================
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


st.set_page_config(
    page_title="CaD-HSL System",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 2. LLM 连接配置 =================
LLM_BASE_URL = "https://b3237.gpu.act.buaa.edu.cn/v1"
LLM_API_KEY = "EMPTY"
LLM_MODEL_NAME = "./DeepSeek-R1-0528-Qwen3-8B"


@st.cache_resource
def get_llm_client():
    try:
        mounts = {
            "http://": httpx.HTTPTransport(proxy=None),
            "https://": httpx.HTTPTransport(proxy=None),
        }
        http_client = httpx.Client(verify=False, timeout=60.0, mounts=mounts, trust_env=False)
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, http_client=http_client)
        client.models.list()
        return client, "Online"
    except Exception as e:
        return None, f"Offline ({str(e)})"


client, connection_status = get_llm_client()


def generate_ai_report(tech_name, drivers, growth_pct):
    """
    生成报告并清洗 DeepSeek 的思维链标签
    """
    if not client:
        return None, "Backend Offline."

    drivers_str = ", ".join(drivers) if drivers else "Self-iteration"

    prompt = f"""
    Role: Strategic Industry Analyst.
    Task: Analyze the causal link between drivers and technology trends.

    [Target]: {tech_name}
    [Forecast]: +{growth_pct:.1f}% growth.
    [Drivers]: {drivers_str}

    Please provide your response in two strict parts:

    PART 1: INTERNAL REASONING
    - Analyze the transmission mechanism.

    PART 2: FINAL DECISION REPORT
    - Executive Summary.
    - Section 1: Causal Attribution.
    - Section 2: Strategic Insight.

    !!! IMPORTANT !!!
    Separate the two parts using exactly: "@@@SEPARATOR@@@"
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a logical analytical engine."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=2000
        )
        content = response.choices[0].message.content

        # --- 关键修复：清洗 <think> 标签 ---
        # 移除 <think>...</think> 及其中的所有内容
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        # 解析分隔符
        if "@@@SEPARATOR@@@" in clean_content:
            parts = clean_content.split("@@@SEPARATOR@@@")
            thought = parts[0].strip().replace("[Internal Analysis]", "").strip()
            report = parts[1].strip().replace("[Executive Report]", "").strip()
        else:
            thought = "Automatic reasoning process..."
            report = clean_content

        return thought, report

    except Exception as e:
        return None, f"Error: {str(e)}"


# ================= 3. 模型定义 (保持不变) =================
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


# ================= 4. 数据加载 =================
@st.cache_resource
def load_all_data():
    try:
        with open(get_path('dictionaries.pkl'), 'rb') as f:
            dicts = pickle.load(f)
        id2tech = dicts['id2tech']
        num_nodes = len(id2tech)

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

        try:
            dates = pd.date_range(end='2024-12-31', periods=len(df_norm), freq='QE')
        except:
            dates = pd.date_range(end='2024-12-31', periods=len(df_norm), freq='Q')
        df_norm.index = dates;
        df_real.index = dates

        device = 'cpu'
        prior = torch.load(get_path('granger_prior.pt'), map_location=device, weights_only=False)
        config = {'num_nodes': num_nodes, 'embed_dim': 32, 'hidden_dim': 32}
        model = CaD_HSL_Model(config, prior).to(device)

        try:
            model.load_state_dict(torch.load(get_path('cad_hsl_model.pth'), map_location=device, weights_only=True))
        except:
            model.load_state_dict(torch.load(get_path('cad_hsl_model.pth'), map_location=device, weights_only=False))

        model.eval()
        adj_matrix = model.causal_learner().detach().cpu().numpy()
        metrics_df = pd.read_csv(get_path('all_tech_metrics.csv'))
        return df_norm, df_real, adj_matrix, id2tech, scaler, metrics_df
    except Exception as e:
        st.error(f"Critical Error: {str(e)}");
        st.stop()


with st.spinner("Initializing System..."):
    df_norm, df_real, adj_matrix, id2tech, scaler, metrics_df = load_all_data()


# ================= 5. 可视化函数 =================
def build_networkx_graph(adj, id2tech, threshold=0.7):
    G = nx.DiGraph()
    rows, cols = np.where(adj > threshold)
    for r, c in zip(rows, cols):
        if r == c: continue
        G.add_edge(id2tech[r], id2tech[c], weight=float(adj[r, c]))
    return G


def plot_pyvis(G, height="600px", select_node=None, mode="full"):
    if len(G.nodes) == 0: return "<div>Empty Graph</div>"
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="#333", directed=True)
    net.from_nx(G)
    for node in net.nodes:
        if select_node and node['id'] == select_node:
            node['color'], node['size'] = '#d62728', 25
        else:
            node['color'], node['size'] = '#4B8BBE', 10

    if mode == "full":
        net.set_options(
            """{"physics": {"forceAtlas2Based": {"gravitationalConstant": -50, "springLength": 100, "damping": 0.4}}}""")
    else:
        net.set_options("""{"physics": {"barnesHut": {"gravitationalConstant": -4000, "springLength": 120}}}""")

    try:
        fd, path = tempfile.mkstemp(suffix=".html")
        os.close(fd)
        net.save_graph(path)
        with open(path, 'r', encoding='utf-8') as f:
            html = f.read()
        os.remove(path)
        return html
    except:
        return "<div>Error</div>"


# ================= 6. 界面主逻辑 =================
st.sidebar.markdown("### CaD-HSL System")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation",
                        ["Global Causal Structure", "Local Ego-Network", "Trend Forecasting", "Evaluation Metrics"])

st.sidebar.markdown("---")
if "Online" in connection_status:
    st.sidebar.markdown(f"<small>LLM: <span style='color:green'>● Online</span></small>", unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"<small>LLM: <span style='color:red'>● {connection_status}</span></small>",
                        unsafe_allow_html=True)

if page == "Global Causal Structure":
    st.markdown("## Global Causal Structure")
    col1, col2 = st.columns([3, 1])
    with col2:
        threshold = st.slider("Causal Threshold", 0.5, 0.95, 0.75, 0.05)
    G = build_networkx_graph(adj_matrix, id2tech, threshold)
    with col1:
        st.info(f"Nodes: {len(G.nodes)} | Edges: {len(G.edges)}")
        if len(G.nodes) > 0: st.components.v1.html(plot_pyvis(G, mode="full"), height=610)

elif page == "Local Ego-Network":
    st.markdown("## Local Ego-Network")
    col1, col2 = st.columns([1, 3])
    with col1:
        tech = st.selectbox("Target Node", list(id2tech.values()))
        thresh = st.slider("Threshold", 0.5, 0.95, 0.7)
    with col2:
        G = build_networkx_graph(adj_matrix, id2tech, thresh)
        if tech in G.nodes:
            st.components.v1.html(plot_pyvis(nx.DiGraph(nx.ego_graph(G, tech, radius=1)), select_node=tech, mode="ego"),
                                  height=610)
        else:
            st.warning("Node isolated.")

elif page == "Trend Forecasting":
    st.markdown("## Trend Forecasting & AI Attribution")
    col1, col2 = st.columns([1, 3])
    with col1:
        target = st.selectbox("Target Technology", list(id2tech.values()))

    row = metrics_df[metrics_df['Tech'] == target]
    drivers = str(row['Drivers'].values[0]).split(',') if len(row) > 0 and pd.notna(row['Drivers'].values[0]) else []

    if 'report_final' not in st.session_state: st.session_state.report_final = None
    if 'last_target' not in st.session_state or st.session_state.last_target != target:
        st.session_state.report_final = None
        st.session_state.last_target = target

    if st.button("Execute Forecast", use_container_width=True):
        df_feat = pd.DataFrame(index=df_norm.index)
        df_feat['Y'] = df_norm[target]
        for l in [1, 2, 3]: df_feat[f'S_L{l}'] = df_norm[target].shift(l)
        for d in drivers:
            if d in df_norm.columns: df_feat[f'D_{d}_L1'] = df_norm[d].shift(1)
        df_feat.dropna(inplace=True)

        train, test = df_feat.iloc[:-4], df_feat.iloc[-4:]
        m_base = xgb.XGBRegressor().fit(train[[c for c in df_feat.columns if 'S_L' in c]], train['Y'])
        m_causal = xgb.XGBRegressor().fit(train.drop('Y', axis=1), train['Y'])

        st.session_state.p_base = m_base.predict(test[[c for c in df_feat.columns if 'S_L' in c]])
        st.session_state.p_causal = m_causal.predict(test.drop('Y', axis=1))
        st.session_state.test_idx = test.index
        st.session_state.y_hist = df_real[target]

    if 'p_causal' in st.session_state and st.session_state.last_target == target:
        idx = list(id2tech.values()).index(target)


        def inv(v):
            m = np.zeros((len(v), len(id2tech)));
            m[:, idx] = v
            return scaler.inverse_transform(m)[:, idx]


        y_b, y_c = inv(st.session_state.p_base), inv(st.session_state.p_causal)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.y_hist.index, y=st.session_state.y_hist.values, name='Ground Truth',
                                 line=dict(color='#2c3e50', width=2.5)))
        fig.add_trace(
            go.Scatter(x=st.session_state.test_idx, y=y_b, name='Baseline', line=dict(color='#95a5a6', dash='dash')))
        fig.add_trace(
            go.Scatter(x=st.session_state.test_idx, y=y_c, name='CaD-HSL (Ours)', line=dict(color='#d62728', width=3)))
        fig.update_layout(template="simple_white", height=400, margin=dict(l=40, r=40, t=40, b=40),
                          legend=dict(orientation="h", y=1.02, x=1))
        st.plotly_chart(fig, use_container_width=True)

        if drivers: st.markdown(f"**Drivers:** `{', '.join(drivers)}`")
        st.markdown("---")

        st.subheader("🤖 AI Causal Reasoning")
        col_gen, _ = st.columns([1, 4])
        if col_gen.button("Generate Strategy Report", type="primary"):
            growth = ((y_c[-1] - y_c[0]) / (y_c[0] + 1e-6)) * 100
            with st.spinner("Analyzing logic chain..."):
                th, rep = generate_ai_report(target, drivers, growth)
                st.session_state.report_thought = th
                st.session_state.report_final = rep

        if st.session_state.report_final:
            with st.expander("🧠 Chain of Thought", expanded=False):
                # 修复1：处理换行
                safe_thought = st.session_state.report_thought.replace('\n', '<br>')
                st.markdown(
                    f"<div style='background-color:#f0f2f6; padding:15px; font-family:monospace; font-size:13px;'>{safe_thought}</div>",
                    unsafe_allow_html=True)

            # 修复2：处理最终报告的换行和加粗
            final_html = st.session_state.report_final.replace('\n', '<br>')
            final_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', final_html)  # 正则替换 Markdown 加粗

            st.markdown(f"""
            <div style="background-color:#fff; border:1px solid #e1e4e8; border-top:5px solid #d62728; padding:25px; border-radius:4px; box-shadow:0 4px 12px rgba(0,0,0,0.1);">
                <h3 style="margin-top:0; color:#2c3e50;">📋 Strategic Attribution Report</h3>
                <div style="font-size:16px; line-height:1.8; text-align:justify; color:#333;">
                    {final_html}
                </div>
                <hr style="margin:20px 0; border:0; border-top:1px dashed #ccc;">
                <div style="font-size:12px; color:#666; text-align:right;">Generated by CaD-HSL + DeepSeek-R1</div>
            </div>
            """, unsafe_allow_html=True)

# --- 模块 4: 模型评估仪表盘 ---
elif page == "Evaluation Metrics":
    st.markdown("## 📊 Quantitative Evaluation Dashboard")

    # 强制重新加载按钮（用于调试）
    if st.button("🔄 Reload Data"):
        st.cache_data.clear()
        st.rerun()

    # === 1. 数据校验与预处理 (对齐 cal2.py 逻辑) ===
    if metrics_df is None or metrics_df.empty:
        st.error("❌ Error: Metrics data is empty. Please ensure 'all_tech_metrics.csv' exists.")
        st.stop()

    # --- 核心计算逻辑：确保所有绘图字段存在 ---
    with st.spinner("Calculating performance metrics..."):
        # A. 计算 RMSE
        if 'Base_MSE' in metrics_df.columns:
            metrics_df['Base_RMSE'] = np.sqrt(metrics_df['Base_MSE'])
            metrics_df['Causal_RMSE'] = np.sqrt(metrics_df['Causal_MSE'])

        # B. 计算提升指标 (基于 MAE)
        if 'Base_MAE' in metrics_df.columns and 'Causal_MAE' in metrics_df.columns:
            metrics_df['Imp_MAE'] = metrics_df['Base_MAE'] - metrics_df['Causal_MAE']
            # 使用 replace(0, np.nan) 防止除以零报错
            metrics_df['Imp_Pct'] = (metrics_df['Imp_MAE'] / metrics_df['Base_MAE'].replace(0, np.nan)) * 100

        # C. 驱动因子数量统计 (如果原始数据没有 Num_Drivers)
        if 'Num_Drivers' not in metrics_df.columns and 'Drivers' in metrics_df.columns:
            metrics_df['Num_Drivers'] = metrics_df['Drivers'].apply(
                lambda x: len(str(x).split(',')) if pd.notna(x) and x != "" else 0
            )

    # --- 2. 核心指标汇总 (Aggregates) ---
    avg_base_mape = metrics_df['Base_MAPE'].mean()
    avg_causal_mape = metrics_df['Causal_MAPE'].mean()
    avg_base_rmse = metrics_df['Base_RMSE'].mean()
    avg_causal_rmse = metrics_df['Causal_RMSE'].mean()
    avg_base_mse = metrics_df['Base_MSE'].mean()
    avg_causal_mse = metrics_df['Causal_MSE'].mean()

    # 提升率计算 (Ratio of Means)
    imp_pct_mape = ((avg_base_mape - avg_causal_mape) / avg_base_mape) * 100
    imp_pct_rmse = ((avg_base_rmse - avg_causal_rmse) / avg_base_rmse) * 100
    imp_pct_mse = ((avg_base_mse - avg_causal_mse) / avg_base_mse) * 100

    # 统计显著性
    t_stat, p_value = stats.ttest_rel(metrics_df['Base_MAPE'], metrics_df['Causal_MAPE'])
    win_rate = (metrics_df['Causal_MAPE'] < metrics_df['Base_MAPE']).mean() * 100
    std_base = metrics_df['Base_MAPE'].std()
    std_causal = metrics_df['Causal_MAPE'].std()

    # --- 3. 展示指标卡 (KPIs) ---
    st.markdown("### 1. Key Performance Indicators (KPIs)")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("MAPE Improvement", f"{imp_pct_mape:+.1f}%", delta=f"Base: {avg_base_mape:.2f}%")
    with k2:
        st.metric("RMSE Improvement", f"{imp_pct_rmse:+.1f}%", delta="Robustness Boost")
    with k3:
        st.metric("Win Rate", f"{win_rate:.1f}%", help="% of tasks where CaD-HSL < Base")
    with k4:
        # 显著性标签处理
        sig_label = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        st.metric("P-value", f"{p_value:.2e}", delta=f"Significance: {sig_label}",
                  delta_color="normal" if p_value < 0.05 else "off")

    st.markdown("---")

    # --- 4. 图表分析区 ---
    c1, c2 = st.columns([1, 1])

    # 图表 A: 散点对比图 (对角线图)
    with c1:
        st.markdown("### 2. Error Comparison (Base vs. CaD-HSL)")
        st.caption("Points **below the diagonal** indicate CaD-HSL is better.")

        max_val = max(metrics_df['Base_MAE'].max(), metrics_df['Causal_MAE'].max()) * 1.1

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=metrics_df['Base_MAE'],
            y=metrics_df['Causal_MAE'],
            mode='markers',
            text=metrics_df['Tech'],
            marker=dict(
                size=10,
                color=metrics_df['Imp_Pct'],
                colorscale='RdYlGn',  # 红色代表负优化，绿色代表提升
                showscale=True,
                colorbar=dict(title="Imp %")
            ),
            name='Tech Node'
        ))
        # 45度辅助线
        fig_scatter.add_shape(
            type="line", x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color="Gray", width=2, dash="dash"),
        )
        fig_scatter.update_layout(
            xaxis_title="Baseline MAE", yaxis_title="CaD-HSL MAE",
            height=450, template="simple_white"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 图表 B: 驱动因子效能分析
    with c2:
        st.markdown("### 3. Causal Impact Analysis")
        st.caption("How the number of causal drivers affects model improvement.")

        # 按驱动因子数量聚合
        driver_impact = metrics_df.groupby('Num_Drivers')['Imp_Pct'].mean().reset_index()

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=driver_impact['Num_Drivers'],
            y=driver_impact['Imp_Pct'],
            marker_color='#4B8BBE',
            text=[f"{v:.1f}%" for v in driver_impact['Imp_Pct']],
            textposition='outside'
        ))
        fig_bar.update_layout(
            xaxis_title="Number of Causal Drivers Identified",
            yaxis_title="Avg MAE Improvement (%)",
            height=450, template="simple_white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- 5. 详细对比表与榜单 ---
    st.markdown("### 4. Deep Evaluation Report")

    rep_col, space_col = st.columns([2, 1])
    with rep_col:
        report_data = {
            "Metric": ["MAPE (Avg)", "RMSE (Avg)", "MSE (Avg)", "Stability (Error Std)"],
            "Base (XGB)": [f"{avg_base_mape:.2f}%", f"{avg_base_rmse:.4f}", f"{avg_base_mse:.4f}", f"{std_base:.2f}"],
            "CaD-HSL (Ours)": [f"{avg_causal_mape:.2f}%", f"{avg_causal_rmse:.4f}", f"{avg_causal_mse:.4f}",
                               f"{std_causal:.2f}"],
            "Improvement (Δ)": [f"{imp_pct_mape:+.1f}%", f"{imp_pct_rmse:+.1f}%", f"{imp_pct_mse:+.1f}%",
                                f"{(std_base - std_causal):.2f} ↓"]
        }
        st.table(pd.DataFrame(report_data))

    st.markdown("### 5. Top 20 Performance Leaderboard")
    # 筛选前20名，并尝试翻译
    top_df = metrics_df[
        ['Tech', 'Num_Drivers', 'Base_MAE', 'Causal_MAE', 'Imp_MAE', 'Imp_Pct']
    ].sort_values('Imp_Pct', ascending=False).head(20).copy()

    # 应用翻译函数
    top_df['Tech'] = top_df['Tech'].apply(lambda x: get_en(x))

    st.dataframe(
        top_df.style
        .background_gradient(subset=['Imp_Pct'], cmap="Greens")
        .format({
            "Base_MAE": "{:.4f}",
            "Causal_MAE": "{:.4f}",
            "Imp_MAE": "{:.4f}",
            "Imp_Pct": "{:.2f}%"
        }),
        use_container_width=True
    )