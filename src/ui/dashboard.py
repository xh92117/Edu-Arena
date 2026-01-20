"""
Edu-Arena 可视化面板 v5.0
- 以大模型名称标识环境
- 单面板+下拉菜单设计
- 拟合曲线展示
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 确保可从项目根目录导入 src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(
    page_title="Edu-Arena 教育模拟",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 常量定义
# ==========================================

# 环境ID到模型名称的映射
ENV_MODEL_MAP = {
    0: "DeepSeek",
    1: "Qwen",
    2: "Kimi",
    3: "ChatGPT",
    4: "Gemini",
    5: "Claude",
    6: "Grok"
}

# 模型颜色映射
MODEL_COLORS = {
    "DeepSeek": "#4361ee",
    "Qwen": "#7209b7",
    "Kimi": "#f72585",
    "ChatGPT": "#06d6a0",
    "Gemini": "#ffd166",
    "Claude": "#ef476f",
    "Grok": "#118ab2"
}

# 指标配置
METRICS_CONFIG = {
    "knowledge": {"label": "知识储备", "color": "#06d6a0", "unit": "分"},
    "stress": {"label": "压力水平", "color": "#ef476f", "unit": "分"},
    "health": {"label": "身体健康", "color": "#118ab2", "unit": "分"},
    "avg_relationship": {"label": "亲子关系", "color": "#ffd166", "unit": "分"},
    "savings": {"label": "家庭存款", "color": "#073b4c", "unit": "元"},
    "father_rel": {"label": "父亲关系", "color": "#4361ee", "unit": "分"},
    "mother_rel": {"label": "母亲关系", "color": "#7209b7", "unit": "分"},
}

# ==========================================
# 3. 样式定义
# ==========================================
st.markdown("""
<style>
    /* 全局样式 */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* 标题 */
    h1 { color: #1a1a2e !important; font-weight: 700 !important; }
    h2 { color: #16213e !important; }
    h3 { color: #0f3460 !important; }
    
    /* 卡片 */
    .model-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 16px;
        color: white;
        text-align: center;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 13px;
        opacity: 0.9;
    }
    
    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* 模型标签 */
    .model-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    
    /* 隐藏默认元素 */
    header { visibility: hidden; }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. 日志文件管理
# ==========================================
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


def get_available_log_files() -> List[Tuple[str, str, datetime]]:
    """获取所有可用的日志文件"""
    if not os.path.exists(LOG_DIR):
        return []
    
    log_files = []
    patterns = ["simulation_*.jsonl", "simulation_log.jsonl"]
    
    for pattern in patterns:
        for filepath in glob.glob(os.path.join(LOG_DIR, pattern)):
            filename = os.path.basename(filepath)
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            size = os.path.getsize(filepath)
            
            if filename.startswith("simulation_") and "_" in filename:
                try:
                    timestamp_str = filename.replace("simulation_", "").replace(".jsonl", "")
                    file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    display_name = f"{file_time.strftime('%m-%d %H:%M')} ({size/1024:.1f}KB)"
                except:
                    display_name = f"{filename}"
            else:
                display_name = f"{filename}"
            
            log_files.append((filepath, display_name, mtime))
    
    log_files.sort(key=lambda x: x[2], reverse=True)
    return log_files


# ==========================================
# 5. 数据加载
# ==========================================
@st.cache_data(ttl=5, show_spinner=False)
def load_log_data(log_file: str) -> pd.DataFrame:
    """加载日志文件并添加模型名称"""
    if not os.path.exists(log_file):
        return pd.DataFrame()
    
    data = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    js = json.loads(line.strip())
                    
                    env_id = js.get("env_id", 0)
                    
                    row = {
                        "timestamp": pd.to_datetime(js.get("timestamp", "")),
                        "env_id": env_id,
                        "model": ENV_MODEL_MAP.get(env_id, f"Model_{env_id}"),
                        "week": js.get("week", 0),
                    }
                    
                    # 孩子状态
                    child = js.get("child_state", {})
                    row.update({
                        "knowledge": child.get("knowledge", 0),
                        "stress": child.get("stress", 0),
                        "health": child.get("physical_health", 100),
                        "father_rel": child.get("father_relationship", 100),
                        "mother_rel": child.get("mother_relationship", 100),
                        "grandfather_rel": child.get("grandfather_relationship", 100),
                        "grandmother_rel": child.get("grandmother_relationship", 100),
                    })
                    
                    # 家庭状态
                    family = js.get("family_state", {})
                    row["savings"] = family.get("family_savings", 0)
                    
                    # 决策信息
                    decision = js.get("coordinated_decision", {})
                    row.update({
                        "action": decision.get("action_type", ""),
                        "member": decision.get("member", ""),
                        "dialogue": decision.get("dialogue", ""),
                        "cost": decision.get("cost", 0),
                    })
                    
                    # DM结果
                    dm = js.get("dm_result", {})
                    row["success"] = dm.get("success", True)
                    
                    data.append(row)
                except:
                    continue
    except Exception as e:
        st.error(f"加载失败: {e}")
        return pd.DataFrame()
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # 计算综合关系分
    if not df.empty:
        df["avg_relationship"] = (
            df["father_rel"] + df["mother_rel"] + 
            df["grandfather_rel"] + df["grandmother_rel"]
        ) / 4
    
    return df


# ==========================================
# 6. 拟合曲线计算
# ==========================================
def calculate_trend_line(x: np.ndarray, y: np.ndarray, degree: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算多项式拟合曲线
    
    参数:
        x: x轴数据
        y: y轴数据
        degree: 多项式次数
    
    返回:
        (x_smooth, y_smooth): 平滑的拟合曲线
    """
    if len(x) < 2:
        return x, y
    
    try:
        # 确保是numpy数组
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        
        # 移除NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        
        if len(x) < 2:
            return x, y
        
        # 限制多项式次数
        degree = min(degree, len(x) - 1)
        
        # 多项式拟合
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        
        # 生成平滑曲线
        x_smooth = np.linspace(x.min(), x.max(), 100)
        y_smooth = poly(x_smooth)
        
        return x_smooth, y_smooth
    except:
        return x, y


# ==========================================
# 7. UI 组件
# ==========================================
def render_sidebar() -> Tuple[Optional[str], bool]:
    """渲染侧边栏"""
    st.sidebar.markdown("## 📚 Edu-Arena")
    st.sidebar.markdown("*多模型教育决策模拟*")
    st.sidebar.markdown("---")
    
    # 日志文件选择
    st.sidebar.markdown("### 📁 数据源")
    log_files = get_available_log_files()
    
    if not log_files:
        st.sidebar.warning("无日志文件")
        return None, False
    
    options = [f[0] for f in log_files]
    labels = [f[1] for f in log_files]
    
    selected_idx = st.sidebar.selectbox(
        "选择日志",
        range(len(options)),
        format_func=lambda i: labels[i],
        key="log_selector"
    )
    
    selected_file = options[selected_idx] if selected_idx is not None else None
    
    st.sidebar.markdown("---")
    
    # 设置
    st.sidebar.markdown("### ⚙️ 设置")
    auto_refresh = st.sidebar.checkbox("自动刷新", value=False)
    
    if st.sidebar.button("🔄 刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    return selected_file, auto_refresh


def render_model_selector(df: pd.DataFrame) -> List[str]:
    """渲染模型选择器"""
    if df.empty:
        return []
    
    models = sorted(df["model"].unique())
    
    # 使用多选框
    selected = st.multiselect(
        "选择模型",
        models,
        default=models,
        key="model_selector"
    )
    
    return selected


def render_metric_selector() -> str:
    """渲染指标选择器"""
    options = list(METRICS_CONFIG.keys())
    labels = [METRICS_CONFIG[k]["label"] for k in options]
    
    selected_idx = st.selectbox(
        "选择指标",
        range(len(options)),
        format_func=lambda i: labels[i],
        key="metric_selector"
    )
    
    return options[selected_idx]


def render_overview(df: pd.DataFrame, selected_models: List[str]):
    """渲染概览卡片"""
    if df.empty or not selected_models:
        return
    
    filtered = df[df["model"].isin(selected_models)]
    latest = filtered.sort_values("timestamp").groupby("model").tail(1)
    
    cols = st.columns(len(selected_models))
    
    for col, model in zip(cols, selected_models):
        model_data = latest[latest["model"] == model]
        if model_data.empty:
            continue
        
        row = model_data.iloc[0]
        color = MODEL_COLORS.get(model, "#666")
        
        with col:
            st.markdown(f"""
            <div class="model-card">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span class="model-tag" style="background: {color}; color: white;">{model}</span>
                    <span style="color: #888; font-size: 12px;">第{int(row['week'])}周</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div>
                        <div style="font-size: 11px; color: #888;">知识</div>
                        <div style="font-size: 20px; font-weight: 600; color: #06d6a0;">{row['knowledge']:.1f}</div>
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #888;">压力</div>
                        <div style="font-size: 20px; font-weight: 600; color: #ef476f;">{row['stress']:.1f}</div>
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #888;">健康</div>
                        <div style="font-size: 20px; font-weight: 600; color: #118ab2;">{row['health']:.1f}</div>
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #888;">存款</div>
                        <div style="font-size: 20px; font-weight: 600; color: #073b4c;">¥{row['savings']/1000:.1f}k</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_trend_chart(df: pd.DataFrame, selected_models: List[str], metric: str):
    """渲染趋势图（带拟合曲线）"""
    if df.empty or not selected_models:
        st.info("请选择至少一个模型")
        return
    
    filtered = df[df["model"].isin(selected_models)]
    
    if filtered.empty:
        return
    
    config = METRICS_CONFIG.get(metric, {"label": metric, "color": "#666", "unit": ""})
    
    fig = go.Figure()
    
    for model in selected_models:
        model_data = filtered[filtered["model"] == model].sort_values("week")
        
        if model_data.empty:
            continue
        
        color = MODEL_COLORS.get(model, "#666")
        x = model_data["week"].values
        y = model_data[metric].values
        
        # 原始数据点（半透明）
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=f"{model} (数据)",
            marker=dict(color=color, size=6, opacity=0.4),
            showlegend=False,
            hovertemplate=f"{model}<br>第%{{x}}周<br>{config['label']}: %{{y:.1f}}{config['unit']}<extra></extra>"
        ))
        
        # 拟合曲线
        x_smooth, y_smooth = calculate_trend_line(x, y, degree=3)
        
        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode="lines",
            name=model,
            line=dict(color=color, width=3),
            hovertemplate=f"{model}<br>第%{{x:.0f}}周<br>{config['label']}: %{{y:.1f}}{config['unit']}<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(
            text=f"📈 {config['label']}趋势对比",
            font=dict(size=18)
        ),
        xaxis_title="周数",
        yaxis_title=f"{config['label']} ({config['unit']})",
        height=450,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.9)",
        hovermode="x unified"
    )
    
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.1)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.1)", zeroline=False)
    
    st.plotly_chart(fig, use_container_width=True)


def render_comparison_chart(df: pd.DataFrame, selected_models: List[str]):
    """渲染模型对比图"""
    if df.empty or not selected_models:
        return
    
    filtered = df[df["model"].isin(selected_models)]
    latest = filtered.sort_values("timestamp").groupby("model").tail(1)
    
    if latest.empty:
        return
    
    # 雷达图
    categories = ["知识", "健康", "关系", "低压力", "经济"]
    
    fig = go.Figure()
    
    for _, row in latest.iterrows():
        model = row["model"]
        color = MODEL_COLORS.get(model, "#666")
        
        # 归一化数据 (0-100)
        values = [
            row["knowledge"],
            row["health"],
            row["avg_relationship"],
            100 - row["stress"],  # 压力越低越好
            min(100, row["savings"] / 1000),  # 存款归一化
        ]
        values.append(values[0])  # 闭合
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=model,
            line_color=color,
            fillcolor=color,
            opacity=0.3
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title=dict(
            text="🎯 模型综合能力对比",
            font=dict(size=18)
        ),
        height=400,
        margin=dict(l=80, r=80, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_decision_log(df: pd.DataFrame, selected_models: List[str]):
    """渲染决策日志"""
    if df.empty or not selected_models:
        return
    
    filtered = df[df["model"].isin(selected_models)]
    display_df = filtered.sort_values("timestamp", ascending=False).head(15)
    
    if display_df.empty:
        st.info("暂无决策记录")
        return
    
    for _, row in display_df.iterrows():
        model = row["model"]
        color = MODEL_COLORS.get(model, "#666")
        
        st.markdown(f"""
        <div style="background: white; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; border-left: 3px solid {color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="background: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600;">{model}</span>
                    <span style="margin-left: 8px; color: #333; font-weight: 500;">{row['action']}</span>
                    <span style="margin-left: 8px; color: #888; font-size: 12px;">[{row['member']}]</span>
                </div>
                <span style="color: #888; font-size: 12px;">第{int(row['week'])}周</span>
            </div>
            <div style="color: #666; font-size: 13px; margin-top: 6px; font-style: italic;">
                "{row['dialogue'][:80]}{'...' if len(str(row['dialogue'])) > 80 else ''}"
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_welcome():
    """渲染欢迎页面"""
    st.markdown("""
    <div style="text-align: center; padding: 80px 20px;">
        <h1 style="font-size: 48px; margin-bottom: 16px;">📚 Edu-Arena</h1>
        <p style="font-size: 20px; color: #666; margin-bottom: 40px;">
            多模型教育决策模拟平台
        </p>
        <div style="display: flex; justify-content: center; gap: 16px; flex-wrap: wrap;">
    """, unsafe_allow_html=True)
    
    for model, color in MODEL_COLORS.items():
        st.markdown(f"""
            <span style="background: {color}; color: white; padding: 8px 20px; border-radius: 20px; font-weight: 600;">{model}</span>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.warning("⚠️ 请先运行 `python main.py` 启动模拟")


# ==========================================
# 8. 主程序
# ==========================================
def main():
    # 侧边栏
    log_file, auto_refresh = render_sidebar()
    
    # 主标题
    st.markdown("# 📚 Edu-Arena 教育模拟监控")
    
    if not log_file:
        render_welcome()
        return
    
    # 加载数据
    df = load_log_data(log_file)
    
    if df.empty:
        render_welcome()
        return
    
    # 控制面板
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_models = render_model_selector(df)
    
    with col2:
        metric = render_metric_selector()
    
    with col3:
        view_mode = st.selectbox(
            "视图模式",
            ["趋势分析", "模型对比", "决策记录"],
            key="view_mode"
        )
    
    st.markdown("---")
    
    # 概览卡片
    render_overview(df, selected_models)
    
    # 主视图
    st.markdown("---")
    
    if view_mode == "趋势分析":
        render_trend_chart(df, selected_models, metric)
    elif view_mode == "模型对比":
        render_comparison_chart(df, selected_models)
    else:
        render_decision_log(df, selected_models)
    
    # 自动刷新
    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
