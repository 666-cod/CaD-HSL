import pandas as pd
import numpy as np
import torch
import glob
import os
import pickle
from tqdm import tqdm

# ================= 配置路径 =================
# 请确保这里是你的真实路径
file_paths = glob.glob(r"D:\predict\1.0\data_按年份划分结果\Data_Year_*.csv")


# 或者使用你之前的列表...

def load_and_merge_data(paths):
    all_dfs = []
    print("正在加载数据文件...")
    for p in tqdm(paths):
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                all_dfs.append(df)
            except Exception as e:
                print(f"读取文件 {p} 失败: {e}")
    if not all_dfs:
        raise ValueError("没有成功加载任何数据！")
    return pd.concat(all_dfs, ignore_index=True)


# 1. 加载与清洗
raw_df = load_and_merge_data(file_paths)

# 处理日期与金额
raw_df['Time_Extracted'] = pd.to_datetime(raw_df['Time_Extracted'], errors='coerce')
raw_df = raw_df.dropna(subset=['Time_Extracted'])
raw_df['Amount_Extracted'] = pd.to_numeric(raw_df['Amount_Extracted'], errors='coerce').fillna(0)
raw_df['norm_amount'] = np.log1p(raw_df['Amount_Extracted'])  # Log Normalization

# 按季度切分
raw_df['period'] = raw_df['Time_Extracted'].dt.to_period('Q')
periods = sorted(raw_df['period'].unique())
period2id = {p: i for i, p in enumerate(periods)}
print(f"时间跨度: {min(periods)} 到 {max(periods)}，共 {len(periods)} 个季度")

# 2. 构建技术词典 (保留相似度 > 0.4)
SIMILARITY_THRESHOLD = 0.4
tech_counter = {}


def update_counter(name, sim):
    try:
        if float(sim) >= SIMILARITY_THRESHOLD and pd.notna(name):
            tech_counter[name] = tech_counter.get(name, 0) + 1
    except:
        pass


print("正在构建技术词典...")
for row in tqdm(raw_df.itertuples(), total=len(raw_df)):
    update_counter(row.匹配技术_1, row.相似度_1)
    update_counter(row.匹配技术_2, row.相似度_2)
    update_counter(row.匹配技术_3, row.相似度_3)

# 过滤低频词
MIN_FREQ = 5
valid_techs = [k for k, v in tech_counter.items() if v >= MIN_FREQ]
tech2id = {tech: i for i, tech in enumerate(valid_techs)}
id2tech = {i: tech for tech, i in tech2id.items()}
NUM_NODES = len(valid_techs)
print(f"有效技术节点数: {NUM_NODES}")

# 【重要】保存字典，供后续步骤使用
with open('dictionaries.pkl', 'wb') as f:
    pickle.dump({'tech2id': tech2id, 'id2tech': id2tech, 'period2id': period2id}, f)
print("已保存: dictionaries.pkl")

# 3. 构建超图序列
hypergraph_sequence = []
print("正在构建超图 Tensor...")

for p in tqdm(periods):
    sub_df = raw_df[raw_df['period'] == p]
    node_indices = []
    edge_indices = []
    edge_attr = []
    current_edge_id = 0

    for row in sub_df.itertuples():
        techs = []
        for t_col, s_col in [(row.匹配技术_1, row.相似度_1), (row.匹配技术_2, row.相似度_2),
                             (row.匹配技术_3, row.相似度_3)]:
            if pd.notna(t_col) and t_col in tech2id:
                try:
                    if float(s_col) >= SIMILARITY_THRESHOLD:
                        techs.append(tech2id[t_col])
                except:
                    pass

        if len(techs) >= 1:
            node_indices.extend(techs)
            edge_indices.extend([current_edge_id] * len(techs))
            edge_attr.append(row.norm_amount)
            current_edge_id += 1

    if current_edge_id > 0:
        H_index = torch.tensor([node_indices, edge_indices], dtype=torch.long)
        H_weight = torch.tensor(edge_attr, dtype=torch.float)
        hypergraph_sequence.append({
            'H': H_index,
            'weights': H_weight,
            'num_edges': current_edge_id,
            'period_idx': period2id[p]
        })

torch.save(hypergraph_sequence, 'hypergraph_seq.pt')
print("超图序列保存完成: hypergraph_seq.pt")