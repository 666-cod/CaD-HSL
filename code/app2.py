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

# === 引入网络库 ===
from openai import OpenAI
import httpx

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

# ================= LLM 连接配置 (针对内网+VPN修复) =================
# 1. 您的内网地址
LLM_BASE_URL = "https://b3237.gpu.act.buaa.edu.cn/v1"
# 2. Key (内网通常为空)
LLM_API_KEY = "EMPTY"
# 3. 模型名称 (注意：必须包含 ./ 前缀，与服务器返回一致)
LLM_MODEL_NAME = "./DeepSeek-R1-0528-Qwen3-8B"


@st.cache_resource
def get_llm_client():
    """初始化连接，强制绕过 EasyConnect 代理"""
    try:
        # --- 核心修复：强制 httpx 不读取系统代理 ---
        mounts = {
            "http://": httpx.HTTPTransport(proxy=None),
            "https://": httpx.HTTPTransport(proxy=None),
        }

        # 配置 Client
        http_client = httpx.Client(
            verify=False,  # 忽略内网自签名证书
            timeout=30.0,  # 增加超时时间
            mounts=mounts,  # 挂载无代理传输
            trust_env=False  # 彻底忽略环境变量中的 HTTP_PROXY
        )

        client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            http_client=http_client
        )

        # 握手测试：尝试获取模型列表
        client.models.list()

        return client, "✅ 已连接: DeepSeek R1 (校内节点)"

    except Exception as e:
        # 如果出错，返回 None 和错误信息
        return None, f"❌ 连接失败: {str(e)}"


# 初始化连接
client, connection_status = get_llm_client()


def generate_ai_report(tech_name, drivers, growth_pct):
    """调用 DeepSeek 生成分析报告"""
    if not client:
        return f"无法生成报告。原因：{connection_status}"

    drivers_str = "、".join(drivers) if drivers else "历史惯性及自身技术迭代"

    prompt = f"""
    你是一位资深的产业科技分析师。请根据以下量化模型的数据，为政府决策者解读技术趋势。

    【分析对象】：{tech_name}
    【预测趋势】：未来一年热度预期增长 {growth_pct:.1f}%
    【核心驱动因子】：{drivers_str}

    请用专业、简练的语言（200字以内）完成以下任务：
    1. **归因分析**：解释为什么这些驱动因子（{drivers_str}）会促进 {tech_name} 的发展？（例如：产业链供需、基建赋能、或宏观项目共振）
    2. **商业洞察**：这反映了什么国家战略或行业转型趋势？

    注意：输出逻辑严密，直接给出结论，不要堆砌套话。
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个专业的科技产业分析助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 生成中断: {str(e)}"


# ================= 模型结构定义 (必须保留以加载 pth) =================
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
        # 1. 加载字典
        with open(get_path('dictionaries.pkl'), 'rb') as f:
            dicts = pickle.load(f)
        id2tech = dicts['id2tech']
        num_nodes = len(id2tech)

        # 2. 加载序列数据
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

        # 处理时间索引
        try:
            dates = pd.date_range(end='2024-12-31', periods=len(df_norm), freq='QE')
        except:
            dates = pd.date_range(end='2024-12-31', periods=len(df_norm), freq='Q')

        df_norm.index = dates
        df_real.index = dates

        # 3. 加载模型
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

        # 4. 加载指标
        metrics_df = pd.read_csv(get_path('all_tech_metrics.csv'))

        return df_norm, df_real, adj_matrix, id2tech, scaler, metrics_df

    except Exception as e:
        st.error(f"❌ 数据加载错误: {str(e)}")
        st.stop()


# 执行加载
with st.spinner("正在初始化数据引擎..."):
    df_norm, df_real, adj_matrix, id2tech, scaler, metrics_df = load_all_data()


# ================= 辅助函数：图构建 =================
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
    if len(G.nodes) == 0: return "<div>图为空</div>"
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    net.from_nx(G)
    for node in net.nodes:
        if select_node and node['id'] == select_node:
            node['color'] = '#FF5733'  # 选中高亮
            node['size'] = 30
        else:
            node['color'] = '#4B8BBE'
            node['size'] = 15

    # 物理配置
    if mode == "full":
        net.set_options("""
        var options = {
          "physics": { "forceAtlas2Based": { "gravitationalConstant": -50, "springLength": 100, "damping": 0.4 } }
        }
        """)
    else:
        net.set_options("""
        var options = {
          "physics": { "barnesHut": { "gravitationalConstant": -3000, "springLength": 95, "avoidOverlap": 1 } }
        }
        """)

    try:
        fd, path = tempfile.mkstemp(suffix=".html")
        os.close(fd)
        net.save_graph(path)
        with open(path, 'r', encoding='utf-8') as f:
            html = f.read()
        os.remove(path)
        return html
    except:
        return "<div>绘图错误</div>"


# ================= 界面主逻辑 =================
st.sidebar.title("🔮 CaD-HSL 驾驶舱")
page = st.sidebar.radio("选择功能模块", ["全景技术关联图", "特定技术二阶图", "趋势预测与对比", "模型总指标"])

st.sidebar.markdown("---")
if "✅" in connection_status:
    st.sidebar.success(connection_status)
else:
    st.sidebar.error(connection_status + "\n(建议检查VPN状态)")

# --- 1. 全景图 ---
if page == "全景技术关联图":
    st.title("🌐 全景技术因果关联图")
    col1, col2 = st.columns([3, 1])
    with col2:
        threshold = st.slider("因果强度阈值", 0.5, 0.95, 0.75, 0.05)

    G_full = build_networkx_graph(adj_matrix, id2tech, threshold)
    with col1:
        st.markdown(f"**节点数:** {len(G_full.nodes)} | **因果连线:** {len(G_full.edges)}")
        if len(G_full.nodes) > 0:
            html = plot_pyvis(G_full, height="600px", mode="full")
            st.components.v1.html(html, height=610)
        else:
            st.warning("当前阈值下无关联数据。")

# --- 2. 二阶图 ---
elif page == "特定技术二阶图":
    st.title("🕸️ 特定技术二阶关联图")
    col1, col2 = st.columns([1, 3])
    with col1:
        tech = st.selectbox("选择核心技术", list(id2tech.values()))
        thresh = st.slider("连接强度", 0.5, 0.95, 0.7)
    with col2:
        G_full = build_networkx_graph(adj_matrix, id2tech, thresh)
        if tech in G_full.nodes:
            G_ego = nx.ego_graph(G_full, tech, radius=1)
            G_viz = nx.DiGraph(G_ego)
            html = plot_pyvis(G_viz, height="600px", select_node=tech, mode="ego")
            st.components.v1.html(html, height=610)
        else:
            st.info(f"技术 **{tech}** 在当前阈值下无关联。")

# --- 3. 趋势预测 (含 LLM) ---
elif page == "趋势预测与对比":
    st.title("📈 趋势预测与归因 (Graph-RAG)")
    target = st.selectbox("技术选择", list(id2tech.values()))

    # 获取指标和驱动因子
    row = metrics_df[metrics_df['Tech'] == target]
    drivers = str(row['Drivers'].values[0]).split(',') if len(row) > 0 and pd.notna(row['Drivers'].values[0]) else []

    # Session State 初始化
    if 'report_content' not in st.session_state:
        st.session_state.report_content = None
    if 'last_target' not in st.session_state or st.session_state.last_target != target:
        st.session_state.report_content = None
        st.session_state.last_target = target

    if st.button("开始预测 (2024-2025)", use_container_width=True):
        # 1. 构造特征
        df_feat = pd.DataFrame(index=df_norm.index)
        df_feat['Y'] = df_norm[target]
        for l in [1, 2, 3]: df_feat[f'S_L{l}'] = df_norm[target].shift(l)
        for d in drivers:
            if d in df_norm.columns:
                df_feat[f'D_{d}_L1'] = df_norm[d].shift(1)
        df_feat.dropna(inplace=True)

        # 2. 训练预测
        train, test = df_feat.iloc[:-4], df_feat.iloc[-4:]
        m_base = xgb.XGBRegressor().fit(train[[c for c in df_feat.columns if 'S_L' in c]], train['Y'])
        m_causal = xgb.XGBRegressor().fit(train.drop('Y', axis=1), train['Y'])

        # 3. 存入 Session
        st.session_state.p_base = m_base.predict(test[[c for c in df_feat.columns if 'S_L' in c]])
        st.session_state.p_causal = m_causal.predict(test.drop('Y', axis=1))
        st.session_state.test_idx = test.index
        st.session_state.y_hist = df_real[target]

    # 显示预测结果
    if 'p_causal' in st.session_state and st.session_state.last_target == target:
        idx = list(id2tech.values()).index(target)


        def simple_inv(v):
            m = np.zeros((len(v), len(id2tech)))
            m[:, idx] = v
            return scaler.inverse_transform(m)[:, idx]


        y_b = simple_inv(st.session_state.p_base)
        y_c = simple_inv(st.session_state.p_causal)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.y_hist.index, y=st.session_state.y_hist.values, name='真实热度',
                                 line=dict(color='black')))
        fig.add_trace(
            go.Scatter(x=st.session_state.test_idx, y=y_b, name='Base预测', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=st.session_state.test_idx, y=y_c, name='CaD-HSL预测', line=dict(color='green')))
        st.plotly_chart(fig, use_container_width=True)

        if drivers:
            st.success(f"📌 核心驱动因子: {', '.join(drivers)}")

        st.divider()
        st.subheader("🤖 AI 深度归因分析")

        # LLM 按钮
        if st.button("生成 AI 研报", type="primary", use_container_width=True):
            growth = ((y_c[-1] - y_c[0]) / (y_c[0] + 1e-6)) * 100
            with st.spinner(f"正在调用 {LLM_MODEL_NAME} 进行推理..."):
                st.session_state.report_content = generate_ai_report(target, drivers, growth)

        # 显示报告
        if st.session_state.report_content:
            if "❌" in st.session_state.report_content:
                st.error(st.session_state.report_content)
            else:
                st.markdown(f"""
                <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; border-left: 5px solid #4B8BBE;">
                    <h4 style="color:#4B8BBE; margin-top:0;">📋 产业分析师报告</h4>
                    <p style="font-size:16px; line-height:1.6;">{st.session_state.report_content}</p>
                    <hr>
                    <p style="font-size:12px; color:grey;">* Powered by DeepSeek R1 & CaD-HSL</p>
                </div>
                """, unsafe_allow_html=True)

# --- 4. 指标表 ---
elif page == "模型总指标":
    st.title("🏆 模型总指标")
    st.dataframe(metrics_df.sort_values('Imp_MAE', ascending=False).head(20), use_container_width=True)