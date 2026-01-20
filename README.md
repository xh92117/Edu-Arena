# 📚 Edu-Arena - 多模型教育决策模拟系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Version-v2.1.0-green.svg" alt="Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<p align="center">
  <b>让AI扮演家长，模拟22年养育过程，寻找最优教育决策策略</b>
</p>

---

## 🎯 项目简介

**Edu-Arena** 是一个创新的多大模型教育决策模拟系统，让多个大语言模型（LLM）扮演中国普通家庭的"家长"角色，在22年的模拟周期中（2010-2032年）做出教育决策。通过对比不同模型的决策效果，找出最优的教育策略，为真实家庭教育提供科学参考。

### 核心特点

- 🤖 **7大模型竞争**：支持 DeepSeek、Qwen、Kimi、ChatGPT、Gemini、Claude、Grok 同时参与
- 📊 **量化评估体系**：高考成绩、大学录取、性格特质、幸福指数等多维度评估
- 👨‍👩‍👧 **家庭动态模拟**：父亲、母亲、祖父、祖母四位家庭成员的协同决策
- 📈 **实时可视化**：Streamlit 监控面板，实时追踪模拟进展
- 🎲 **随机事件系统**：教育政策、家庭事件、连锁反应等丰富事件

---

## 📢 最新更新 (v2.1.0)

- ✅ **财务计算修复**: 工资增长不再重复累积，数据更准确
- ✅ **更新时机优化**: 财务每周更新，避免累积误差
- ✅ **类型安全**: 完善类型提示，提升代码质量
- ✅ **性能提升**: 支持异步日志写入，提升30%性能
- ✅ **稳定性**: 优化错误处理，系统更加健壮

---

## 🏗️ 系统架构

### 模拟设定

| 项目 | 设定值 |
|------|--------|
| 模拟周期 | 2010年1月 - 2032年7月（约1170周） |
| 时间单位 | 1步 = 1周 |
| 现实时间映射 | 1小时 ≈ 24周模拟时间 |
| 初始存款 | 50,000 元 |
| 家庭收入 | 父亲6000元/月 + 母亲4000元/月 + 祖父2000元/月 + 祖母1500元/月 |

### 孩子初始属性

| 属性 | 初始值 | 范围 |
|------|--------|------|
| 智商 (IQ) | 120 | 80-200 |
| 知识储备 | 0 | 0-100 |
| 压力值 | 0 | 0-100 |
| 身体健康 | 100 | 0-100 |
| 家庭关系 | 100 (各成员) | 0-100 |

---

## 📊 评分系统

### 硬性指标 (60%)

#### 高考分数计算

| 因素 | 权重 | 说明 |
|------|------|------|
| 知识储备 | 40% | 22年知识积累 |
| 智商 IQ | 20% | 先天智力因素 |
| 压力影响 | -15% | 高压力降低发挥 |
| 身体健康 | 10% | 影响学习效率 |
| 家庭投入 | 15% | 教育经济投入 |

#### 大学录取等级

| 等级 | 分数线 |
|------|--------|
| 985高校 | ≥ 680分 |
| 211高校 | ≥ 650分 |
| 普通本科 | ≥ 550分 |
| 大专院校 | ≥ 400分 |

### 软指标 (40%)

- **幸福指数**：人际关系 × 0.4 + 健康 × 0.25 + 成就 × 0.2 - 压力 × 0.15
- **社交适应度**：关系平均值 × 0.7 + (100-压力) × 0.3
- **情绪稳定性**：(100-压力) × 0.6 + 健康 × 0.4
- **性格特质识别**：自信、外向、抗压、乐观、独立、创造性等12种特质

### 综合评价等级

| 等级 | 分数范围 | 说明 |
|------|----------|------|
| A | ≥ 90分 | 优秀 |
| B | 80-89分 | 良好 |
| C | 70-79分 | 中等 |
| D | 60-69分 | 及格 |
| F | < 60分 | 不及格 |

---

## 🛠️ 技术栈

- **Python 3.11+** - 编程语言
- **AsyncIO** - 异步并发框架
- **Pydantic** - 数据验证
- **Streamlit** - 可视化监控面板
- **Plotly** - 交互式图表
- **OpenAI SDK** - LLM API 集成

---

## 📁 项目结构

```
Edu-Arena/
├── src/                          # 源代码
│   ├── agents/                   # AI代理模块
│   │   ├── base.py               # 智能体基类
│   │   ├── factory.py            # 智能体工厂
│   │   ├── llm_agents.py         # LLM适配器
│   │   └── adaptive_agent.py     # 自适应代理
│   ├── core/                     # 核心功能模块
│   │   ├── config.py             # 配置管理
│   │   ├── state.py              # 状态定义
│   │   ├── runner.py             # 模拟运行器
│   │   ├── evaluation_system.py  # 评价系统
│   │   ├── llm_client.py         # LLM客户端
│   │   ├── emotional_system.py   # 情绪系统
│   │   ├── decision_coordinator.py # 决策协调器
│   │   └── constants.py          # 常量定义
│   ├── engine/                   # 模拟引擎
│   │   ├── simulation.py         # 模拟环境
│   │   └── dungeon_master.py     # 裁判系统(DM)
│   ├── data/                     # 数据处理
│   │   ├── event_system.py       # 事件系统
│   │   └── timeline.py           # 时间线数据
│   └── ui/                       # 用户界面
│       └── dashboard.py          # 可视化面板
├── examples/                     # 示例文件
│   └── custom_events_example.json
├── scripts/                      # 工具脚本
│   ├── check_config.py           # 配置检查
│   └── analyze_data.py           # 数据分析
├── logs/                         # 日志目录
├── main.py                       # 主程序入口
├── presentation.html             # 项目演示PPT
├── requirements.txt              # 依赖列表
├── env.example                   # 环境变量示例
└── README.md                     # 项目说明
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp env.example .env
```

编辑 `.env` 文件，配置 LLM API 密钥：

```env
# DeepSeek 配置
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# Qwen 配置
QWEN_API_KEY=your_api_key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-turbo

# 更多模型配置...
```

### 3. 运行模拟

```bash
# 运行所有7个模型
python main.py

# 或运行指定模型
python run_specific_models.py
```

### 4. 启动可视化面板

```bash
streamlit run src/ui/dashboard.py
```

---

## 🔌 支持的大模型

| 模型 | 环境变量前缀 | 说明 |
|------|-------------|------|
| DeepSeek | `DEEPSEEK_` | 国产大模型 |
| Qwen (通义千问) | `QWEN_` | 阿里云大模型 |
| Kimi (月之暗面) | `KIMI_` | Moonshot AI |
| ChatGPT | `CHATGPT_` | OpenAI |
| Gemini | `GEMINI_` | Google |
| Claude | `CLAUDE_` | Anthropic |
| Grok | `GROK_` | xAI |

每个模型需配置三个环境变量：
- `{PREFIX}_API_KEY` - API 密钥
- `{PREFIX}_BASE_URL` - API 基础 URL
- `{PREFIX}_MODEL` - 模型名称

---

## 📈 可视化监控

Streamlit 面板提供以下功能：

- 📊 多模型同时对比展示
- 📈 知识/压力/健康/存款趋势图
- 🎯 多项式拟合曲线分析
- 🕸️ 雷达图综合能力对比
- 📝 实时决策日志展示
- 🔄 支持自动刷新

---

## 🎲 事件系统

### 事件类型

- **教育里程碑**：入学、考试、升学等
- **外部事件**：社会政策变化
- **家庭事件**：生病、旅行、获奖等
- **随机事件**：概率触发的意外情况
- **连锁事件**：一个事件触发后续事件链

### 自定义事件

```python
from src.data.event_system import get_enhanced_event_system

event_system = get_enhanced_event_system()
event_system.load_custom_events("custom_events.json")
```

---

## 🧒 发展敏感期

基于蒙特梭利教育理论，系统模拟孩子在不同年龄的发展敏感期：

| 敏感期 | 年龄范围 | 说明 |
|--------|----------|------|
| 语言敏感期 | 0-6岁 | 语言学习黄金期 |
| 秩序敏感期 | 1-3岁 | 规则意识形成 |
| 感官敏感期 | 0-6岁 | 感知能力发展 |
| 社交敏感期 | 2.5-6岁 | 社交能力培养 |
| 数学敏感期 | 4-6岁 | 数学思维启蒙 |
| 阅读敏感期 | 4.5-7岁 | 阅读兴趣培养 |

---

## 📊 日志格式

模拟数据以 JSONL 格式记录：

```json
{
  "timestamp": "2015-06-15",
  "env_id": 0,
  "week": 280,
  "child_state": {
    "knowledge": 45.2,
    "stress": 30.5,
    "physical_health": 85.0,
    "father_relationship": 78.5
  },
  "family_state": {
    "family_savings": 125000.0
  },
  "coordinated_decision": {
    "action_type": "辅导",
    "member": "父亲",
    "dialogue": "女儿，今天爸爸帮你复习一下数学。"
  },
  "dm_result": {
    "success": true,
    "state_changes": {}
  }
}
---

## 🔧 工具脚本

### 配置检查

```bash
python scripts/check_config.py
```

### 数据分析

```bash
python scripts/analyze_data.py
```

### 数据导出

```bash
python scripts/analyze_data.py --export
```

---

---

## 🗺️ 未来规划

- [ ] 扩展更多家庭成员类型
- [ ] 支持更复杂的成员交互关系
- [ ] 增加机器学习模型支持
- [ ] 优化模拟效率和准确性
- [ ] 提供更丰富的可视化图表
- [ ] 支持历史数据回放功能
- [ ] Web 端部署支持

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

<p align="center">
  <b>Edu-Arena v2.1.0</b><br>
  让 AI 探索最优教育决策，为真实家庭提供科学参考
</p>
