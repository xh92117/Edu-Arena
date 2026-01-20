from pydantic_settings import BaseSettings
from pydantic import Field, validator, AliasChoices
from typing import List, Optional, Dict
import os
from dotenv import load_dotenv

# 预加载 .env，确保无前缀变量可被 os.getenv 获取
load_dotenv(dotenv_path=".env", override=False)


class SimulationConfig(BaseSettings):
    """模拟系统配置类"""

    # 环境设置
    num_environments: int = Field(default=7, ge=1, le=20, description="运行的环境数量")
    simulation_speed: float = Field(default=1/24, gt=0, description="模拟速度 (1小时现实时间=多少周模拟时间)")
    log_dir: str = Field(default="logs", description="日志目录")
    log_file: str = Field(default="", description="日志文件名（留空则自动生成带时间戳的文件名）")
    log_max_size_mb: float = Field(default=100.0, ge=1.0, le=1000.0, description="日志文件最大大小（MB），超过此大小将轮转")
    log_max_backups: int = Field(default=10, ge=1, le=100, description="最大备份文件数量")
    log_rotate_on_start: bool = Field(default=True, description="启动时是否轮转现有日志")
    log_use_timestamp: bool = Field(default=True, description="每次启动时使用新的带时间戳日志文件")

    # 模型设置
    supported_models: List[str] = Field(
        default=["deepseek", "qwen", "kimi", "chatgpt", "gemini", "claude", "grok"],
        description="支持的模型列表"
    )
    specific_models: Optional[List[str]] = Field(
        default=None,
        description="指定运行的模型列表，为None时运行所有支持的模型"
    )

    # 特定模型运行的默认配置
    default_specific_models: List[str] = Field(
        default=["deepseek", "qwen"],
        description="默认的特定模型列表"
    )

    # LLM配置 - 通用设置
    llm_timeout: int = Field(default=30, ge=1, le=300, description="LLM请求超时时间(秒)")
    llm_enable_emotion_analysis: bool = Field(default=True, description="是否启用情感分析")

    # 各模型独立配置
    deepseek_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("EDU_ARENA_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
        description="DeepSeek API密钥"
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        validation_alias=AliasChoices("EDU_ARENA_DEEPSEEK_BASE_URL", "DEEPSEEK_BASE_URL"),
        description="DeepSeek API基础URL"
    )
    deepseek_model: str = Field(
        default="deepseek-chat",
        validation_alias=AliasChoices("EDU_ARENA_DEEPSEEK_MODEL", "DEEPSEEK_MODEL"),
        description="DeepSeek模型名称"
    )

    qwen_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("EDU_ARENA_QWEN_API_KEY", "QWEN_API_KEY"),
        description="Qwen API密钥"
    )
    qwen_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias=AliasChoices("EDU_ARENA_QWEN_BASE_URL", "QWEN_BASE_URL"),
        description="Qwen API基础URL"
    )
    qwen_model: str = Field(
        default="qwen-turbo",
        validation_alias=AliasChoices("EDU_ARENA_QWEN_MODEL", "QWEN_MODEL"),
        description="Qwen模型名称"
    )

    kimi_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("EDU_ARENA_KIMI_API_KEY", "KIMI_API_KEY"),
        description="Kimi API密钥"
    )
    kimi_base_url: str = Field(
        default="https://api.moonshot.cn/v1",
        validation_alias=AliasChoices("EDU_ARENA_KIMI_BASE_URL", "KIMI_BASE_URL"),
        description="Kimi API基础URL"
    )
    kimi_model: str = Field(
        default="moonshot-v1-8k",
        validation_alias=AliasChoices("EDU_ARENA_KIMI_MODEL", "KIMI_MODEL"),
        description="Kimi模型名称"
    )

    chatgpt_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("EDU_ARENA_CHATGPT_API_KEY", "CHATGPT_API_KEY"),
        description="ChatGPT API密钥"
    )
    chatgpt_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("EDU_ARENA_CHATGPT_BASE_URL", "CHATGPT_BASE_URL"),
        description="ChatGPT API基础URL"
    )
    chatgpt_model: str = Field(
        default="gpt-3.5-turbo",
        validation_alias=AliasChoices("EDU_ARENA_CHATGPT_MODEL", "CHATGPT_MODEL"),
        description="ChatGPT模型名称"
    )

    gemini_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("EDU_ARENA_GEMINI_API_KEY", "GEMINI_API_KEY"),
        description="Gemini API密钥"
    )
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        validation_alias=AliasChoices("EDU_ARENA_GEMINI_BASE_URL", "GEMINI_BASE_URL"),
        description="Gemini API基础URL"
    )
    gemini_model: str = Field(
        default="gemini-pro",
        validation_alias=AliasChoices("EDU_ARENA_GEMINI_MODEL", "GEMINI_MODEL"),
        description="Gemini模型名称"
    )

    claude_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("EDU_ARENA_CLAUDE_API_KEY", "CLAUDE_API_KEY"),
        description="Claude API密钥"
    )
    claude_base_url: str = Field(
        default="https://api.anthropic.com/v1",
        validation_alias=AliasChoices("EDU_ARENA_CLAUDE_BASE_URL", "CLAUDE_BASE_URL"),
        description="Claude API基础URL"
    )
    claude_model: str = Field(
        default="claude-3-haiku-20240307",
        validation_alias=AliasChoices("EDU_ARENA_CLAUDE_MODEL", "CLAUDE_MODEL"),
        description="Claude模型名称"
    )

    grok_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("EDU_ARENA_GROK_API_KEY", "GROK_API_KEY"),
        description="Grok API密钥"
    )
    grok_base_url: str = Field(
        default="https://api.x.ai/v1",
        validation_alias=AliasChoices("EDU_ARENA_GROK_BASE_URL", "GROK_BASE_URL"),
        description="Grok API基础URL"
    )
    grok_model: str = Field(
        default="grok-1",
        validation_alias=AliasChoices("EDU_ARENA_GROK_MODEL", "GROK_MODEL"),
        description="Grok模型名称"
    )

    # 向后兼容的旧配置（已废弃，请使用各模型独立配置）
    llm_api_key: Optional[str] = Field(default=None, description="[已废弃] LLM API密钥，请使用各模型独立配置")
    llm_base_url: Optional[str] = Field(default=None, description="[已废弃] LLM API基础URL，请使用各模型独立配置")
    llm_model: str = Field(default="gpt-3.5-turbo", description="[已废弃] 默认使用的LLM模型，请使用各模型独立配置")

    # 重试配置
    max_retries: int = Field(default=3, ge=0, description="最大重试次数")
    retry_delay: float = Field(default=1.0, gt=0, description="重试基础延迟(秒)")
    
    # 经济系统配置
    inflation_rate_2010_2020: float = Field(default=0.03, ge=0, le=0.1, description="2010-2020年通胀率（年化）")
    inflation_rate_2020_2030: float = Field(default=0.025, ge=0, le=0.1, description="2020-2030年通胀率（年化）")
    salary_growth_rate: float = Field(default=0.025, ge=0, le=0.1, description="工资年化增长率")
    base_monthly_expenses: float = Field(default=5000.0, ge=0, description="2010年基准月均生活成本（元）")
    
    # 并发控制配置
    max_concurrent_tasks: int = Field(default=10, ge=1, le=100, description="最大并发任务数")
    enable_rate_limiting: bool = Field(default=False, description="是否启用速率限制")
    rate_limit_max_calls: int = Field(default=100, ge=1, description="速率限制：时间窗口内最大调用次数")
    rate_limit_window: float = Field(default=60.0, gt=0, description="速率限制：时间窗口（秒）")
    enable_error_isolation: bool = Field(default=True, description="是否启用错误隔离")
    error_isolation_max_failures: int = Field(default=5, ge=1, description="错误隔离：最大失败次数")
    error_isolation_window: float = Field(default=60.0, gt=0, description="错误隔离：时间窗口（秒）")
    
    # 内存管理配置
    memory_cleanup_interval: float = Field(default=3600.0, gt=0, description="内存清理间隔（秒）")
    memory_max_usage_mb: Optional[float] = Field(default=None, ge=0, description="最大内存使用（MB），None表示不限制")
    enable_memory_persistence: bool = Field(default=False, description="是否启用内存持久化")
    memory_persistence_dir: str = Field(default="data/cache", description="内存持久化目录")
    
    # 评价系统配置（可选，使用默认值）
    gaokao_base_score: Optional[int] = Field(default=None, ge=0, le=1000, description="高考基础分数")
    gaokao_max_score: Optional[int] = Field(default=None, ge=0, le=1000, description="高考满分")
    gaokao_weights: Optional[Dict[str, float]] = Field(default=None, description="高考分数计算权重")
    evaluation_weights: Optional[Dict[str, float]] = Field(default=None, description="综合评价权重")

    class Config:
        env_prefix = "EDU_ARENA_"
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("deepseek_api_key", "qwen_api_key", "kimi_api_key", "chatgpt_api_key", "gemini_api_key", "claude_api_key", "grok_api_key", always=True)
    def normalize_api_key_placeholders(cls, v):
        if not v:
            return v
        lowered = v.strip().lower()
        if lowered.startswith("your_") or "your_" in lowered:
            return None
        return v

    @validator("log_file", always=True)
    def build_log_file_path(cls, v, values):
        """构建完整的日志文件路径（支持时间戳）"""
        from datetime import datetime
        log_dir = values.get("log_dir", "logs")
        use_timestamp = values.get("log_use_timestamp", True)
        
        if use_timestamp or not v:
            # 使用带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_{timestamp}.jsonl"
        else:
            filename = v if v else "simulation_log.jsonl"
        
        return os.path.join(log_dir, filename)

    @validator("specific_models", always=True)
    def validate_specific_models(cls, v, values):
        """验证特定模型列表"""
        if v is not None:
            supported_models = values.get("supported_models", [])
            invalid_models = [model for model in v if model not in supported_models]
            if invalid_models:
                raise ValueError(f"不支持的模型: {invalid_models}")
        return v

    def get_models_to_run(self) -> List[str]:
        """获取要运行的模型列表"""
        if self.specific_models:
            return self.specific_models
        return self.supported_models[:self.num_environments]

    def get_simulation_speed_info(self) -> str:
        """获取模拟速度的描述信息"""
        return f"1小时现实时间 = {self.simulation_speed * 24:.1f}周模拟时间 (每周 = {self.simulation_speed * 3600:.1f} 秒)"

    def get_model_config(self, model_name: str) -> Dict[str, str]:
        """
        获取指定模型的配置

        参数:
            model_name: 模型名称

        返回:
            模型配置字典，包含 api_key, base_url, model
        """
        model_name = model_name.lower()

        # 模型配置映射
        model_configs = {
            "deepseek": {
                "api_key": self.deepseek_api_key,
                "base_url": self.deepseek_base_url,
                "model": self.deepseek_model
            },
            "qwen": {
                "api_key": self.qwen_api_key,
                "base_url": self.qwen_base_url,
                "model": self.qwen_model
            },
            "kimi": {
                "api_key": self.kimi_api_key,
                "base_url": self.kimi_base_url,
                "model": self.kimi_model
            },
            "chatgpt": {
                "api_key": self.chatgpt_api_key,
                "base_url": self.chatgpt_base_url,
                "model": self.chatgpt_model
            },
            "gemini": {
                "api_key": self.gemini_api_key,
                "base_url": self.gemini_base_url,
                "model": self.gemini_model
            },
            "claude": {
                "api_key": self.claude_api_key,
                "base_url": self.claude_base_url,
                "model": self.claude_model
            },
            "grok": {
                "api_key": self.grok_api_key,
                "base_url": self.grok_base_url,
                "model": self.grok_model
            }
        }

        if model_name not in model_configs:
            raise ValueError(f"不支持的模型: {model_name}")

        config = model_configs[model_name]

        # 检查配置完整性
        if not config["api_key"]:
            raise ValueError(f"未配置 {model_name} 的 API 密钥")

        return config

    def get_available_models(self) -> List[str]:
        """
        获取已配置可用模型列表

        返回:
            已配置的模型名称列表
        """
        available_models = []
        supported_models = ["deepseek", "qwen", "kimi", "chatgpt", "gemini", "claude", "grok"]

        for model in supported_models:
            try:
                self.get_model_config(model)
                available_models.append(model)
            except ValueError:
                continue

        return available_models


# 创建默认配置实例
def get_default_config() -> SimulationConfig:
    """获取默认配置"""
    return SimulationConfig()


def get_config_for_all_models() -> SimulationConfig:
    """获取运行所有模型的配置"""
    return SimulationConfig()


def get_config_for_specific_models(models: Optional[List[str]] = None) -> SimulationConfig:
    """获取运行指定模型的配置"""
    if models is None:
        models = ["deepseek", "qwen"]
    return SimulationConfig(specific_models=models)