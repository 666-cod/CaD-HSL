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
import re
from openai import OpenAI
from scipy import stats

# ================= 1. 基础配置与翻译字典 =================
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

# --- 核心翻译字典 (基于提供的 JSON 生成) ---
NODE_TRANS_MAP = {
    # === 先进制造 ===
    "研发服务（含基于AI的生成式设计/GtM）": "R&D Services (AI Design/GtM)",
    "设计服务（含工业软件CAD/CAE/EDA云化与智能化）": "Design Services (Cloud CAD/CAE/EDA)",
    "工业互联网平台与边缘计算技术": "Industrial Internet & Edge Computing",
    "现场总线、工业以太网与TSN（时间敏感网络）": "Fieldbus/Industrial Ethernet/TSN",
    "嵌入式系统与端侧智能控制": "Embedded Systems & Edge Control",
    "制造执行系统(MES)与APS高级排程": "MES & APS Scheduling",
    "工业生产过程综合自动化（数字孪生工厂技术）": "Industrial Automation (Digital Twin)",
    "新一代工业控制计算机技术": "Next-Gen Industrial Control Computing",
    "具身智能与人形机器人": "Embodied AI & Humanoid Robots",
    "机器人（工业协作机器人、特种机器人）": "Robotics (Collaborative/Specialized)",
    "高档数控装备与五轴联动加工技术": "High-end CNC & 5-Axis Machining",
    "增材制造技术（金属3D打印、4D打印）": "Additive Manufacturing (3D/4D Printing)",
    "大规模集成电路制造相关技术（原子级制造装备）": "VLSI Manufacturing (Atomic Level)",
    "智能装备驱动控制技术": "Smart Equipment Drive Control",
    "特种加工技术": "Non-traditional Machining",
    "高端装备再制造技术": "High-end Equipment Remanufacturing",
    "机械基础件及制造技术": "Basic Mechanical Components",
    "通用机械装备制造技术": "General Machinery Equipment",
    "极端制造与专用机械装备制造技术": "Extreme & Specialized Manufacturing",
    "纺织及其他行业专用设备制造技术": "Textile & Specialized Industry Equip.",
    "矿山安全生产技术（含AI视觉安全监测）": "Mine Safety (AI Visual Monitoring)",
    "危险化学品安全生产技术": "HazChem Safety Technology",
    "其它事故防治及处置技术": "Accident Prevention & Disposal",
    "新型传感器（MEMS、智能感知终端）": "Advanced Sensors (MEMS/Smart Terminals)",
    "新型自动化仪器仪表": "New Automation Instruments",
    "科学分析仪器（高通量生物检测仪器）": "Scientific Instruments (High-throughput Bio)",
    "精确制造中的测控仪器仪表": "Precision Mfg. Measurement Instruments",
    "微机电系统技术": "MEMS Technology",
    "检验检测认证技术": "Inspection & Certification Tech",
    "标准化服务技术": "Standardization Services",
    "高技术专业化服务（涉及集成电路设计、测试与芯片制造服务等相关技术）": "High-tech Specialized Services (IC Design/Test)",
    "制药装备技术": "Pharmaceutical Equipment Tech",

    # === 电子信息 ===
    "人工智能大模型（通用大模型、行业垂直模型、多模态技术）": "AI Large Models (General/Vertical/Multimodal)",
    "生成式人工智能 (AIGC) 关键技术": "Generative AI (AIGC) Key Tech",
    "智算中心基础设施与算力调度": "AI Computing Centers & Scheduling",
    "量子计算（超导/光/离子阱路线、量子纠错）": "Quantum Computing (Supercond/Photon/Ion)",
    "量子通信（QKD、量子隐形传态）": "Quantum Comm. (QKD/Teleportation)",
    "量子精密测量与传感": "Quantum Sensing & Metrology",
    "高性能集成电路设计（CPU/GPU/NPU、存算一体、RISC-V）": "High-Perf IC Design (CPU/GPU/RISC-V)",
    "先进封装与Chiplet技术（2.5D/3D封装）": "Advanced Packaging & Chiplets (2.5D/3D)",
    "集成电路芯片制造工艺（先进制程、特色工艺）": "IC Fab Processes (Advanced/Specialty)",
    "集成电路设计、测试与EDA工具（AI辅助EDA）": "IC Design/Test & EDA (AI-assisted)",
    "集成光电子器件设计、制造与工艺技术": "Integrated Optoelectronics Design/Mfg",
    "基础软件（服务器OS、分布式数据库、中间件）": "Basic Software (OS/DB/Middleware)",
    "工业软件（PLM、MES、仿真内核）": "Industrial Software (PLM/MES/Sim)",
    "云计算与移动互联网软件（云原生技术、Serverless）": "Cloud & Mobile Software (Cloud Native/Serverless)",
    "物联网应用软件": "IoT Application Software",
    "中文及多语种处理软件": "NLP Software (Chinese/Multilingual)",
    "图形和图像处理软件": "Graphics & Image Processing Software",
    "地理信息系统(GIS)软件": "GIS Software",
    "电子商务软件": "E-Commerce Software",
    "电子政务软件": "E-Government Software",
    "企业管理软件": "Enterprise Management Software",
    "Web 服务与集成软件": "Web Services & Integration",
    "嵌入式软件": "Embedded Software",
    "计算机辅助设计与辅助工程管理软件": "CAD/CAE Management Software",
    "第六代移动通信 (6G)（太赫兹、通感一体）": "6G Communication (THz/ISAC)",
    "卫星互联网与空天地一体化网络": "Satellite Internet & Integrated Networks",
    "5G/5G-A移动通信系统技术": "5G/5G-A Mobile Systems",
    "光传输系统（硅光子技术、CPO）": "Optical Transmission (Silicon Photonics/CPO)",
    "有线宽带接入系统技术": "Wired Broadband Access",
    "微波通信系统技术": "Microwave Communication",
    "物联网设备、部件及组网技术": "IoT Devices & Networking",
    "电信网络运营支撑管理技术": "Telecom OSS/BSS Tech",
    "电信网与互联网增值业务应用技术": "Telecom/Internet VAS Tech",
    "通信网络技术": "Communication Network Tech",
    "敏感元器件与传感器（柔性电子、生物传感器）": "Sensors (Flexible/Bio-sensors)",
    "半导体发光与显示（Micro-LED、全息显示）": "Semiconductor Display (Micro-LED/Holo)",
    "片式和集成无源元件": "Chip & Integrated Passives",
    "大功率半导体器件": "High-Power Semiconductor Devices",
    "专用特种器件": "Specialized Components",
    "中高档机电组件": "High-end Electromechanical Components",
    "平板显示器件": "Flat Panel Display Devices",
    "密码技术（抗量子密码）": "Cryptography (Post-Quantum)",
    "网络与通信安全（零信任架构、AI安全防御）": "Network Security (Zero Trust/AI Defense)",
    "认证授权技术": "Auth & Authorization Tech",
    "系统与软件安全技术": "System & Software Security",
    "安全保密技术": "Security & Confidentiality Tech",
    "安全测评技术": "Security Testing & Eval",
    "安全管理技术": "Security Management Tech",
    "应用安全技术": "Application Security Tech",
    "计算机及终端设计与制造技术": "Computer/Terminal Design & Mfg",
    "计算机外围设备设计与制造技术": "Peripheral Design & Mfg",
    "网络设备设计与制造技术": "Network Equipment Design & Mfg",
    "网络应用技术": "Network Application Tech",
    "广播电视节目采编播系统技术": "Broadcasting Production Systems",
    "广播电视业务集成与支撑系统技术": "Broadcasting Integration Systems",
    "有线传输与覆盖系统技术": "Cable Transmission Systems",
    "无线传输与覆盖系统技术": "Wireless Transmission Systems",
    "广播电视监测监管、安全运行与维护系统技术": "Broadcasting Monitoring & O&M",
    "数字电影系统技术": "Digital Cinema Systems",
    "数字电视终端技术": "Digital TV Terminals",
    "专业视频应用服务平台技术": "Pro Video Service Platforms",
    "音响、光盘技术": "Audio & Disc Technology",
    "云计算服务技术": "Cloud Computing Services",
    "数据服务技术": "Data Services",
    "其他信息服务技术": "Other Info Services",
    "电子商务技术": "E-Commerce Technology",

    # === 新材料 ===
    "第三/四代半导体材料（碳化硅、氮化镓、氧化镓）": "3rd/4th Gen Semiconductors (SiC/GaN)",
    "超导材料（高温超导带材）": "Superconducting Materials (HTS)",
    "石墨烯与二维材料": "Graphene & 2D Materials",
    "生物基与生物降解材料": "Bio-based & Biodegradable Materials",
    "精品钢材制备技术": "High-Quality Steel Production",
    "铝、铜、镁、钛合金清洁生产与深加工技术": "Al/Cu/Mg/Ti Alloy Processing",
    "稀有、稀土金属精深产品制备技术": "Rare Earth Metal Processing",
    "纳米及粉末冶金新材料制备与应用技术": "Nano & Powder Metallurgy",
    "金属及金属基复合新材料制备技术": "Metal Matrix Composites",
    "特种合金（航空级钛合金、高温合金）": "Special Alloys (Aerospace Ti/Superalloys)",
    "半导体新材料制备与应用技术": "Semiconductor Material Prep",
    "电工、微电子和光电子新材料制备与应用技术": "Electronic/Optoelectronic Materials",
    "超导、高效能电池等其它新材料制备与应用技术": "Other New Materials (Battery/Supercond)",
    "结构陶瓷及陶瓷基复合材料（陶瓷基复合材料CMC）": "Structural Ceramics & CMC",
    "功能陶瓷制备技术": "Functional Ceramics",
    "功能玻璃制备技术": "Functional Glass",
    "节能与新能源用材料制备技术": "Energy Saving/New Energy Materials",
    "环保及环境友好型材料技术": "Eco-friendly Materials",
    "新型功能高分子材料（聚酰亚胺、PEEK等特种工程塑料）": "Functional Polymers (PI/PEEK)",
    "工程和特种工程塑料制备技术": "Engineering Plastics",
    "新型橡胶的合成技术及橡胶新材料制备技术": "Synthetic Rubber & New Materials",
    "新型纤维及复合材料（碳纤维T800/T1000级）": "New Fibers & Composites (Carbon Fiber T800+)",
    "高分子材料制备及循环再利用技术": "Polymer Recycling Tech",
    "高分子材料的新型加工和应用技术": "Polymer Processing Tech",
    "新型催化剂制备及应用技术": "Advanced Catalysts",
    "电子化学品制备及应用技术": "Electronic Chemicals",
    "超细功能材料制备及应用技术": "Ultrafine Functional Materials",
    "精细化学品制备及应用技术": "Fine Chemicals",
    "高效工业酶制备与生物催化技术": "Industrial Enzymes & Biocatalysis",
    "微生物发酵技术": "Microbial Fermentation",
    "生物反应及分离技术": "Bioreaction & Separation",
    "天然产物有效成份的分离提取技术": "Natural Product Extraction",

    # === 航空航天 ===
    "低空经济与飞行器（eVTOL、工业级无人机、轻型运动飞机）": "Low-Altitude Econ (eVTOL/UAV)",
    "飞行器动力技术（航空混动/电推进系统）": "Aircraft Propulsion (Hybrid/Electric)",
    "飞行器系统与空中管制（低空空域管理系统UOM）": "Aircraft Systems & ATC (UOM)",
    "民航及通用航空运行保障技术": "Civil/General Aviation Operations",
    "飞行器": "Aircraft",
    "飞行器制造与材料技术": "Aircraft Mfg & Materials",
    "空中管制技术": "Air Traffic Control Tech",
    "商业航天与运载技术（可重复使用火箭、液氧甲烷发动机）": "Commercial Space & Launch (Reusable Rockets)",
    "卫星总体与平台（低轨卫星星座组网、平板卫星）": "Satellites (LEO Constellations)",
    "卫星应用技术（通导遥一体化服务）": "Satellite Applications (Comm/Nav/Remote Sensing)",
    "运载火箭技术": "Launch Vehicle Tech",
    "卫星有效载荷技术": "Satellite Payload Tech",
    "航天测控技术": "Space Tracking & Control",
    "航天电子与航天材料制造技术": "Space Electronics & Materials",
    "先进航天动力设计技术": "Advanced Space Propulsion",

    # === 生物医药 ===
    "合成生物学与生物制造（细胞工厂、非粮原料转化）": "Synthetic Biology & Biomanufacturing",
    "细胞与基因治疗 (CGT)（CAR-T、基因编辑CRISPR）": "Cell & Gene Therapy (CAR-T/CRISPR)",
    "新型疫苗（mRNA疫苗、重组蛋白疫苗）": "Novel Vaccines (mRNA/Recombinant)",
    "生物大分子类药物研发技术": "Biologic Macromolecule Drugs",
    "天然药物生物合成制备技术": "Natural Drug Biosynthesis",
    "生物分离介质、试剂、装置及相关检测技术": "Bioseparation & Reagents",
    "生物治疗技术和基因工程药物": "Biotherapy & Genetic Drugs",
    "诊断技术": "Diagnostic Technology",
    "脑机接口 (BCI) 技术（侵入式/非侵入式采集、神经编解码）": "Brain-Computer Interface (BCI)",
    "高端医学影像设备（超高场MRI、光子计数CT）": "High-end Medical Imaging (MRI/CT)",
    "手术机器人与智能诊疗系统": "Surgical Robots & Smart Diagnosis",
    "新型治疗、急救与康复技术": "New Therapy/First Aid/Rehab",
    "新型电生理检测和监护技术": "Electrophysiology Monitoring",
    "医学检验技术及新设备": "Medical Laboratory Tech",
    "医学专用网络新型软件": "Medical Network Software",
    "医用探测及射线计量检测技术": "Medical Detection & Dosimetry",
    "医学影像诊断技术": "Medical Imaging Diagnosis",
    "组织工程与再生医学材料（3D生物打印、类器官）": "Tissue Eng. & Regen. Med (3D Bioprinting)",
    "植入介入医疗器械材料": "Implant/Intervention Materials",
    "介入治疗器具材料制备技术": "Interventional Device Materials",
    "心脑血管外科用新型生物材料制备技术": "Cardiovascular Biomaterials",
    "骨科内置物制备技术": "Orthopedic Implants",
    "口腔材料制备技术": "Dental Materials",
    "新型敷料和止血材料制备技术": "Dressings & Hemostatics",
    "专用手术器械和材料制备技术": "Surgical Instruments & Materials",
    "其他新型医用材料及制备技术": "Other Medical Materials",
    "中药资源可持续利用与生态保护技术": "TCM Resource Sustainability",
    "创新药物研发技术": "Innovative Drug R&D",
    "中成药二次开发技术": "TCM Secondary Development",
    "中药质控及有害物质检测技术": "TCM QC & Safety Testing",
    "创新药物技术": "Innovative Drug Tech",
    "手性药物创制技术": "Chiral Drug Creation",
    "晶型药物创制技术": "Crystal Form Drug Creation",
    "国家基本药物生产技术": "Essential Drug Production",
    "创新制剂技术": "Innovative Formulation",
    "新型给药制剂技术": "New Drug Delivery Systems",
    "制剂新辅料开发及生产技术": "Excipient Development",

    # === 新能源 ===
    "清洁氢能技术（PEM/碱性电解水制氢、固态储氢）": "Clean Hydrogen (PEM/Alkaline/Solid State)",
    "太阳能（钙钛矿电池、叠层电池）": "Solar Energy (Perovskite/Tandem)",
    "风能（深远海漂浮式风电）": "Wind Energy (Deep Sea Floating)",
    "生物质能": "Biomass Energy",
    "地热能、海洋能及运动能": "Geothermal/Ocean/Kinetic Energy",
    "新型储能技术（钠离子电池、液流电池、固态电池）": "Next-Gen Storage (Na-ion/Flow/Solid State)",
    "燃料电池技术（氢燃料电池堆、膜电极）": "Fuel Cells (Stacks/MEA)",
    "高性能绿色电池(组)技术": "High-Perf Green Batteries",
    "超级电容器与热电转换技术": "Supercapacitors & Thermoelectrics",
    "新型动力电池(组)与储能电池技术": "Traction & Storage Batteries",
    "新型电力系统（源网荷储一体化、虚拟电厂）": "New Power Systems (VPP/Grid Integration)",
    "智能电网与微网技术": "Smart Grid & Microgrids",
    "发电与储能技术": "Generation & Storage Tech",
    "输电技术": "Power Transmission Tech",
    "配电与用电技术": "Distribution & Consumption Tech",
    "变电技术": "Substation Tech",
    "系统仿真与自动化技术": "System Simulation & Auto",
    "工业节能技术": "Industrial Energy Saving",
    "能量回收利用技术": "Energy Recovery Tech",
    "蓄热式燃烧技术": "Regenerative Combustion",
    "输配电系统优化技术": "T&D Optimization",
    "高温热泵技术": "High-Temp Heat Pumps",
    "建筑节能技术": "Building Energy Efficiency",
    "能源系统管理、优化与控制技术": "Energy Mgmt & Control",
    "节能监测技术": "Energy Saving Monitoring",
    "氢能": "Hydrogen Energy",

    # === 现代交通 ===
    "自动驾驶与智能座舱（L3/L4级自动驾驶算法、激光雷达）": "Autonomous Driving & Smart Cockpit",
    "节能与新能源汽车（800V高压快充、车规级芯片）": "NEVs (800V Charging/Auto Chips)",
    "车用发动机及其相关技术（氢内燃机）": "Vehicle Engines (H2 ICE)",
    "汽车关键零部件技术": "Key Auto Components",
    "机动车及发动机先进设计、制造和测试平台技术": "Vehicle Design/Mfg/Test Platforms",
    "轨道车辆及关键零部件技术": "Rail Vehicles & Components",
    "车路云一体化协同控制": "Vehicle-Road-Cloud Coordination",
    "交通控制与管理技术（城市交通大脑）": "Traffic Control (City Brain)",
    "交通基础信息采集、处理技术": "Traffic Info Collection/Processing",
    "交通运输运营管理技术": "Transportation Operations Mgmt",
    "车、船载电子设备技术": "Vehicle/Ship Electronics",
    "轨道交通车辆及运行保障技术": "Rail Transit Operations Support",
    "轨道交通运营管理与服务技术": "Rail Transit Mgmt & Services",
    "高技术船舶设计制造技术": "High-Tech Ship Design/Mfg",
    "海洋工程装备设计制造技术": "Offshore Eng. Equipment",
    "物流与供应链管理技术": "Logistics & Supply Chain Tech",

    # === 城市与社会 ===
    "智慧城市与城市生命线监测": "Smart City & Lifeline Monitoring",
    "互联网教育与数字内容（元宇宙教育场景）": "EdTech & Digital Content (Metaverse)",
    "智慧健康与养老服务": "Smart Health & Elderly Care",
    "现代体育服务支撑技术": "Modern Sports Services Tech",
    "智慧城市服务支撑技术": "Smart City Support Tech",
    "互联网教育": "Internet Education",
    "健康管理": "Health Management",
    "文化载体和介质新材料制备技术": "Cultural Media Materials",
    "艺术专用新材料制备技术": "Artistic Materials",
    "影视场景和舞台专用新材料的加工生产技术": "Set & Stage Materials",
    "文化产品印刷新材料制备技术": "Cultural Printing Materials",
    "文物保护新材料制备技术": "Relic Conservation Materials",
    "知识产权与成果转化服务（全部内容）": "IP & Tech Transfer Services",
    "创作、设计与制作技术": "Creation/Design/Production Tech",
    "传播与展示技术": "Dissemination & Display Tech",
    "文化遗产发现与再利用技术": "Heritage Discovery & Reuse",
    "运营与管理技术": "Operations & Management Tech",
    "乐器制造技术": "Musical Instrument Mfg",
    "印刷技术": "Printing Technology",

    # === 环保与资源 ===
    "碳捕集、利用与封存 (CCUS)": "CCUS",
    "重点行业减污降碳协同技术": "Pollution & Carbon Reduction",
    "重污染行业生产过程中节水、减排及资源化关键技术": "Heavy Industry Water/Emission Reduction",
    "清洁生产关键技术": "Cleaner Production Tech",
    "环保制造关键技术": "Eco-Manufacturing Tech",
    "城市矿产与动力电池回收": "Urban Mining & Battery Recycling",
    "资源勘查开采技术": "Resource Exploration/Mining",
    "提高矿产资源回收利用率的采矿、选矿技术": "Mineral Recovery Enhancement",
    "伴生有价元素的分选提取技术": "Associated Element Extraction",
    "低品位资源和尾矿资源综合利用技术": "Low-grade/Tailings Utilization",
    "绿色矿山建设技术": "Green Mine Construction",
    "城镇污水处理与资源化技术": "Urban Sewage Treatment",
    "工业废水处理与资源化技术": "Industrial Wastewater Treatment",
    "农业水污染控制技术": "Agri-Water Pollution Control",
    "流域水污染治理与富营养化综合控制技术": "Watershed Pollution Control",
    "节水与非常规水资源综合利用技术": "Water Saving/Unconventional Water",
    "饮用水安全保障技术": "Drinking Water Safety",
    "煤燃烧污染防治技术": "Coal Combustion Pollution Control",
    "机动车排放控制技术": "Vehicle Emission Control",
    "工业炉窑污染防治技术": "Industrial Kiln Pollution Control",
    "工业有害废气控制技术": "Industrial Waste Gas Control",
    "有限空间空气污染防治技术": "Confined Space Air Control",
    "危险固体废弃物处置技术": "Hazardous Waste Disposal",
    "工业固体废弃物综合利用技术": "Industrial Solid Waste Utilization",
    "生活垃圾处置与资源化技术": "MSW Disposal & Recycling",
    "建筑垃圾处置与资源化技术": "Construction Waste Recycling",
    "有机固体废物处理与资源化技术": "Organic Waste Treatment",
    "社会源固体废物处置与资源化技术": "Social Solid Waste Disposal",
    "噪声、振动污染防治技术": "Noise & Vibration Control",
    "环境监测预警技术": "Env. Monitoring & Warning",
    "应急环境监测技术": "Emergency Env. Monitoring",
    "生态环境监测技术": "Eco-Env. Monitoring",
    "非常规污染物监测技术": "Unconventional Pollutant Monitoring",
    "生态环境建设与保护技术（全部内容）": "Eco-Env Construction & Protection",
    "高技术专业化服务（涉及环境监理、监测...）": "High-tech Env. Services",

    # === 农业 ===
    "生物育种（全基因组选择、基因编辑育种）": "Bio-breeding (Genomic Selection/Editing)",
    "智慧农业与农业机器人": "Smart Agriculture & Agri-Robots",
    "农林植物优良新品种与优质高效安全生产技术": "New Plant Varieties & Safe Production",
    "畜禽水产优良新品种与健康养殖技术": "Livestock/Aquaculture Breeding",
    "重大农林生物灾害与动物疫病防控技术": "Pest & Disease Control",
    "现代农业装备与信息化技术": "Modern Agri-Equipment & IT",
    "农业面源和重金属污染农田综合防治与修复技术": "Farmland Pollution Remediation",
    "食品安全生产与评价技术": "Food Safety Production/Eval",
    "食品安全检测技术": "Food Safety Testing",

    # === 核应用 ===
    "核能（可控核聚变前沿技术、小型模块化反应堆）": "Nuclear Energy (Fusion/SMR)",
    "核能": "Nuclear Energy",
    "核与辐射安全防治技术": "Nuclear & Radiation Safety",
    "矿山安全生产技术（涉及放射性/核相关安全）": "Mine Safety (Radioactive)",
    "放射性资源勘查开发技术": "Radioactive Resource Exploration",
    "放射性废物处理处置技术": "Radioactive Waste Treatment"
}


def get_en(name):
    """Safely get English name, return original if not found"""
    return NODE_TRANS_MAP.get(name, name)


# ================= 2. LLM 连接配置 =================
LLM_BASE_URL = "https://725ce.gpu.act.buaa.edu.cn/v1"
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


def generate_ai_report(tech_name_en, drivers, growth_pct):
    """
    生成报告并清洗 DeepSeek 的思维链标签
    """
    if not client:
        return None, "Backend Offline."

    drivers_str = ", ".join([get_en(d) for d in drivers]) if drivers else "Self-iteration"

    prompt = f"""
    Role: Strategic Industry Analyst.
    Task: Analyze the causal link between drivers and technology trends.

    [Target Tech]: {tech_name_en}
    [Forecast]: +{growth_pct:.1f}% growth.
    [Key Drivers]: {drivers_str}

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
                {"role": "system", "content": "You are a logical analytical engine. Respond in English."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=2000
        )
        content = response.choices[0].message.content

        # --- 关键修复：清洗 <think> 标签 ---
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

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


# ================= 5. 可视化函数 (已修改支持英文) =================
def build_networkx_graph(adj, id2tech, threshold=0.7):
    G = nx.DiGraph()
    rows, cols = np.where(adj > threshold)
    for r, c in zip(rows, cols):
        if r == c: continue
        # 获取中文名
        cn_src = id2tech[r]
        cn_dst = id2tech[c]
        # 转换为英文名
        en_src = get_en(cn_src)
        en_dst = get_en(cn_dst)
        G.add_edge(en_src, en_dst, weight=float(adj[r, c]))
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
        # 下拉菜单显示英文，但返回中文 key，方便数据处理
        tech_cn = st.selectbox(
            "Target Node",
            list(id2tech.values()),
            format_func=lambda x: get_en(x)
        )
        tech_en = get_en(tech_cn)  # 图中使用的是英文ID

        thresh = st.slider("Threshold", 0.5, 0.95, 0.7)
    with col2:
        G = build_networkx_graph(adj_matrix, id2tech, thresh)
        if tech_en in G.nodes:
            st.components.v1.html(
                plot_pyvis(nx.DiGraph(nx.ego_graph(G, tech_en, radius=1)), select_node=tech_en, mode="ego"),
                height=610
            )
        else:
            st.warning(f"Node '{tech_en}' isolated at current threshold.")

elif page == "Trend Forecasting":
    st.markdown("## Trend Forecasting & AI Attribution")
    col1, col2 = st.columns([1, 3])
    with col1:
        target = st.selectbox(
            "Target Technology",
            list(id2tech.values()),
            format_func=lambda x: get_en(x)
        )
        target_en = get_en(target)

    row = metrics_df[metrics_df['Tech'] == target]
    drivers = str(row['Drivers'].values[0]).split(',') if len(row) > 0 and pd.notna(row['Drivers'].values[0]) else []
    # 驱动因子转英文用于展示
    drivers_en = [get_en(d) for d in drivers]

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

        fig.update_layout(
            title=f"Forecast: {target_en}",
            template="simple_white",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", y=1.02, x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        if drivers_en: st.markdown(f"**Drivers:** `{', '.join(drivers_en)}`")
        st.markdown("---")

        st.subheader("🤖 AI Causal Reasoning")
        col_gen, _ = st.columns([1, 4])
        if col_gen.button("Generate Strategy Report", type="primary"):
            growth = ((y_c[-1] - y_c[0]) / (y_c[0] + 1e-6)) * 100
            with st.spinner("Analyzing logic chain..."):
                # 传入英文名给 LLM
                th, rep = generate_ai_report(target_en, drivers, growth)
                st.session_state.report_thought = th
                st.session_state.report_final = rep

        if st.session_state.report_final:
            with st.expander("🧠 Chain of Thought", expanded=False):
                safe_thought = st.session_state.report_thought.replace('\n', '<br>')
                st.markdown(
                    f"<div style='background-color:#f0f2f6; padding:15px; font-family:monospace; font-size:13px;'>{safe_thought}</div>",
                    unsafe_allow_html=True)

            final_html = st.session_state.report_final.replace('\n', '<br>')
            final_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', final_html)

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
# # --- 模块 4: 模型评估仪表盘 ---
# elif page == "Evaluation Metrics":
#     st.markdown("## 📊 Quantitative Evaluation Dashboard")
#
#     if st.button("🔄 Reload Data"):
#         st.cache_data.clear()
#         st.rerun()
#
#     if metrics_df is None or metrics_df.empty:
#         st.error("❌ Error: Metrics data is empty.")
#         st.stop()
#
#     if 'Base_MSE' in metrics_df.columns:
#         metrics_df['Base_RMSE'] = np.sqrt(metrics_df['Base_MSE'])
#         metrics_df['Causal_RMSE'] = np.sqrt(metrics_df['Causal_MSE'])
#
#     avg_base_mape = metrics_df['Base_MAPE'].mean()
#     avg_causal_mape = metrics_df['Causal_MAPE'].mean()
#     avg_base_rmse = metrics_df['Base_RMSE'].mean()
#     avg_causal_rmse = metrics_df['Causal_RMSE'].mean()
#     avg_base_mse = metrics_df['Base_MSE'].mean()
#     avg_causal_mse = metrics_df['Causal_MSE'].mean()
#
#     imp_pct_mape = ((avg_base_mape - avg_causal_mape) / avg_base_mape) * 100
#     imp_pct_rmse = ((avg_base_rmse - avg_causal_rmse) / avg_base_rmse) * 100
#     imp_pct_mse = ((avg_base_mse - avg_causal_mse) / avg_base_mse) * 100
#
#     t_stat, p_value = stats.ttest_rel(metrics_df['Base_MAPE'], metrics_df['Causal_MAPE'])
#     win_rate = (metrics_df['Causal_MAPE'] < metrics_df['Base_MAPE']).mean() * 100
#     std_base = metrics_df['Base_MAPE'].std()
#     std_causal = metrics_df['Causal_MAPE'].std()
#
#     st.markdown("### 1. Key Performance Indicators (KPIs)")
#     k1, k2, k3, k4 = st.columns(4)
#     with k1:
#         st.metric("MAPE Improvement", f"+{imp_pct_mape:.1f}%", delta=f"Base: {avg_base_mape:.2f}%")
#     with k2:
#         st.metric("RMSE Improvement", f"+{imp_pct_rmse:.1f}%", delta="Robustness")
#     with k3:
#         st.metric("Win Rate", f"{win_rate:.1f}%", help="% of tasks where Ours < Base")
#     with k4:
#         sig_label = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
#         st.metric("P-value", f"{p_value:.2e}", delta=sig_label, delta_color="off")
#
#     c1, c2 = st.columns([1, 1])
#
#     with c1:
#         st.markdown("### 1. Model Comparison (Base vs. CaD-HSL)")
#         st.caption("Points **below the diagonal line** indicate CaD-HSL has lower error (Better).")
#
#         if not metrics_df.empty:
#             max_val = max(metrics_df['Base_MAE'].max(), metrics_df['Causal_MAE'].max())
#
#             # 为散点图添加英文标签列
#             metrics_df['Tech_EN'] = metrics_df['Tech'].map(lambda x: get_en(x))
#
#             fig_scatter = go.Figure()
#             fig_scatter.add_trace(go.Scatter(
#                 x=metrics_df['Base_MAE'],
#                 y=metrics_df['Causal_MAE'],
#                 mode='markers',
#                 text=metrics_df['Tech_EN'],  # 使用英文名
#                 marker=dict(
#                     size=8,
#                     color=metrics_df['Imp_Pct'],
#                     colorscale='Bluered',
#                     showscale=True,
#                     colorbar=dict(title="Improvement %")
#                 ),
#                 name='Technology Node'
#             ))
#             fig_scatter.add_shape(
#                 type="line", x0=0, y0=0, x1=max_val, y1=max_val,
#                 line=dict(color="Gray", width=2, dash="dash"),
#             )
#             fig_scatter.update_layout(
#                 xaxis_title="Baseline MAE",
#                 yaxis_title="CaD-HSL MAE",
#                 height=400,
#                 template="simple_white",
#                 margin=dict(l=40, r=40, t=20, b=40)
#             )
#             st.plotly_chart(fig_scatter, use_container_width=True)
#         else:
#             st.info("No data for scatter plot.")
#
#     with c2:
#         st.markdown("### 2. Causal Drivers Impact Analysis")
#         st.caption("Does having more causal drivers lead to better prediction accuracy?")
#
#         if not metrics_df.empty:
#             driver_impact = metrics_df.groupby('Num_Drivers')['Imp_Pct'].mean().reset_index()
#
#             fig_bar = go.Figure()
#             fig_bar.add_trace(go.Bar(
#                 x=driver_impact['Num_Drivers'],
#                 y=driver_impact['Imp_Pct'],
#                 marker_color='#4B8BBE',
#                 text=[f"{v:.1f}%" for v in driver_impact['Imp_Pct']],
#                 textposition='auto'
#             ))
#             fig_bar.update_layout(
#                 xaxis_title="Number of Causal Drivers Identified",
#                 yaxis_title="Average MAE Improvement (%)",
#                 height=400,
#                 template="simple_white",
#                 margin=dict(l=40, r=40, t=20, b=40)
#             )
#             st.plotly_chart(fig_bar, use_container_width=True)
#         else:
#             st.info("No data for bar chart.")
#
#         st.markdown("### 2. Deep Evaluation Report")
#
#         report_data = {
#             "Metric": ["MAPE", "RMSE", "MSE", "Stability (Std)"],
#             "Base (XGB)": [
#                 f"{avg_base_mape:.2f}%",
#                 f"{avg_base_rmse:.4f}",
#                 f"{avg_base_mse:.4f}",
#                 f"{std_base:.2f}"
#             ],
#             "CaD-HSL (Ours)": [
#                 f"{avg_causal_mape:.2f}%",
#                 f"{avg_causal_rmse:.4f}",
#                 f"{avg_causal_mse:.4f}",
#                 f"{std_causal:.2f}"
#             ],
#             "Improvement (Ratio of Means)": [
#                 f"+{imp_pct_mape:.1f}%",
#                 f"+{imp_pct_rmse:.1f}%",
#                 f"+{imp_pct_mse:.1f}%",
#                 f"{std_base - std_causal:.2f} (lower is better)"
#             ]
#         }
#         st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)
#         st.markdown("---")
#
#     st.markdown("### 3. Top Performance Leaderboard")
#     st.markdown("Technologies benefitting most from causal structure learning.")
#
#     if not metrics_df.empty:
#         # 准备显示用的表格，替换中文为英文
#         top_df = metrics_df[
#             ['Tech', 'Num_Drivers', 'Base_MAE', 'Causal_MAE', 'Imp_MAE', 'Imp_Pct', 'Drivers']].sort_values('Imp_MAE',
#                                                                                                             ascending=False).head(
#             20)
#
#         # 翻译 Tech 列
#         top_df['Tech'] = top_df['Tech'].map(lambda x: get_en(x))
#         # 翻译 Drivers 列 (如果需要)
#         # top_df['Drivers'] = top_df['Drivers'].apply(lambda x: ", ".join([get_en(d) for d in str(x).split(',')]) if pd.notna(x) else "")
#
#         st.dataframe(
#             top_df.style
#             .background_gradient(subset=['Imp_Pct'], cmap="Greens")
#             .format({
#                 "Base_MAE": "{:.4f}",
#                 "Causal_MAE": "{:.4f}",
#                 "Imp_MAE": "{:.4f}",
#                 "Imp_Pct": "{:.1f}%"
#             }),
#             use_container_width=True,
#             height=500
#         )
#     else:
#         st.info("No data to display in leaderboard.")