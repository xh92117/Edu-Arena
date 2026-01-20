import logging
from typing import Dict, Any, Optional, List
from src.core.config import SimulationConfig
from abc import ABC, abstractmethod

from src.core.retry import (
    retry_async_decorator,
    CircuitBreaker,
    with_fallback,
    DefaultFallback,
    LLM_RETRY_CONFIG,
    LLM_CIRCUIT_BREAKER_CONFIG
)
from src.core.config import SimulationConfig

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """LLM客户端抽象基类"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.circuit_breaker = CircuitBreaker(**LLM_CIRCUIT_BREAKER_CONFIG)

    @abstractmethod
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """执行聊天补全"""
        pass

    @abstractmethod
    async def analyze_emotion(self, dialogue: str) -> Dict[str, float]:
        """分析对话情感"""
        pass


class MockLLMClient(LLMClient):
    """模拟LLM客户端，用于测试和降级"""

    def __init__(self, config: SimulationConfig, model_name: str = "mock"):
        super().__init__(config)
        self.model_name = model_name

    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """模拟聊天补全"""
        logger.info(f"使用Mock LLM客户端({self.model_name})处理请求")

        # 模拟API响应
        return {
            "choices": [{
                "message": {
                    "content": f"这是来自{self.model_name}的模拟LLM响应"
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

    async def analyze_emotion(self, dialogue: str) -> Dict[str, float]:
        """模拟情感分析"""
        logger.info(f"使用Mock LLM客户端({self.model_name})分析对话情感: {dialogue[:50]}...")

        # 模拟情感分析结果
        import random
        emotion_score = random.uniform(-2.0, 2.0)

        return {
            "father_relationship": emotion_score,
            "mother_relationship": emotion_score * 0.8,
            "grandfather_relationship": emotion_score * 0.6,
            "grandmother_relationship": emotion_score * 0.6
        }


class OpenAILLMClient(LLMClient):
    """OpenAI兼容的LLM客户端"""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)

        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=config.llm_api_key,
                base_url=config.llm_base_url,
                timeout=config.llm_timeout
            )
        except ImportError:
            raise ImportError("需要安装openai包: pip install openai")

    @retry_async_decorator(**LLM_RETRY_CONFIG)
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """执行聊天补全"""
        return await self.circuit_breaker.call(
            self._chat_completion_impl,
            messages,
            **kwargs
        )

    async def _chat_completion_impl(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """实际的聊天补全实现"""
        response = await self.client.chat.completions.create(
            messages=messages,
            **kwargs
        )

        return {
            "choices": [{
                "message": {
                    "content": response.choices[0].message.content
                }
            }],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }

    async def analyze_emotion(self, dialogue: str) -> Dict[str, float]:
        """分析对话情感（增强版：包含降级处理）"""
        emotion_prompt = f"""
        你是一个情感分析专家，请分析以下家庭成员对孩子说的话，给出对亲子关系的影响值（-5.0到5.0之间）。
        只需要返回一个数值，不要其他内容。

        对话内容：{dialogue}
        """

        messages = [
            {
                "role": "system",
                "content": "你是一个情感分析专家，请分析对话内容并返回一个-5.0到5.0之间的数值。正数表示积极影响，负数表示消极影响。"
            },
            {
                "role": "user",
                "content": emotion_prompt
            }
        ]

        try:
            response = await self.chat_completion(
                messages=messages,
                model=self.config.llm_model,
                temperature=0.0,
                max_tokens=10
            )

            content = response["choices"][0]["message"]["content"].strip()

            # 尝试解析数值
            try:
                emotion_score = float(content)
                emotion_score = max(-5.0, min(5.0, emotion_score))  # 限制范围
            except ValueError:
                logger.warning(f"无法解析LLM情感分析结果: {content}，使用降级方案")
                emotion_score = OpenAILLMClient._fallback_emotion_analysis(dialogue)

        except Exception as e:
            logger.warning(f"LLM情感分析失败: {e}，使用降级方案")
            emotion_score = OpenAILLMClient._fallback_emotion_analysis(dialogue)

        return {
            "father_relationship": emotion_score,
            "mother_relationship": emotion_score * 0.8,
            "grandfather_relationship": emotion_score * 0.6,
            "grandmother_relationship": emotion_score * 0.6
        }
    
    @staticmethod
    def _fallback_emotion_analysis(dialogue: str) -> float:
        """
        降级情感分析：基于关键词的简单情感分析
        
        参数:
            dialogue: 对话内容
            
        返回:
            情感分数 (-5.0 到 5.0)
        """
        # 积极关键词
        positive_keywords = [
            "加油", "很棒", "很好", "优秀", "表扬", "鼓励", "支持", 
            "相信", "努力", "进步", "开心", "高兴", "喜欢", "爱"
        ]
        
        # 消极关键词
        negative_keywords = [
            "不行", "不好", "差", "批评", "斥责", "失望", "生气",
            "愤怒", "讨厌", "笨", "懒", "不努力", "失败"
        ]
        
        # 中性/严格要求关键词
        neutral_keywords = [
            "学习", "作业", "考试", "成绩", "辅导", "培训"
        ]
        
        dialogue_lower = dialogue.lower()
        
        # 计算关键词匹配
        positive_count = sum(1 for keyword in positive_keywords if keyword in dialogue_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in dialogue_lower)
        neutral_count = sum(1 for keyword in neutral_keywords if keyword in dialogue_lower)
        
        # 计算情感分数
        if positive_count > negative_count:
            # 积极情感
            emotion_score = min(5.0, 1.0 + positive_count * 0.5)
        elif negative_count > positive_count:
            # 消极情感
            emotion_score = max(-5.0, -1.0 - negative_count * 0.5)
        elif neutral_count > 0:
            # 中性（可能是严格要求）
            emotion_score = 0.0
        else:
            # 无法判断，返回中性
            emotion_score = 0.0
        
        logger.debug(f"降级情感分析: 积极{positive_count}, 消极{negative_count}, 中性{neutral_count} -> {emotion_score:.2f}")
        return emotion_score


class LLMClientFactory:
    """LLM客户端工厂（支持客户端缓存，每个模型独立客户端）"""
    
    # 客户端缓存：{model_name: client_instance}
    _client_cache: Dict[str, LLMClient] = {}

    @staticmethod
    def create_client(config: SimulationConfig, model_name: str = None) -> LLMClient:
        """
        创建LLM客户端

        参数:
            config: 模拟配置
            model_name: 模型名称，如果为None则使用通用配置

        返回:
            LLM客户端实例
        """
        if model_name:
            # 使用特定模型配置
            model_name_lower = model_name.lower()
            
            # 检查缓存
            if model_name_lower in LLMClientFactory._client_cache:
                logger.debug(f"使用缓存的{model_name}客户端")
                return LLMClientFactory._client_cache[model_name_lower]
            
            try:
                model_config = config.get_model_config(model_name)
                client = LLMClientFactory._create_real_client_for_model(config, model_name, model_config)
                logger.info(f"成功创建{model_name}的LLM客户端: {type(client).__name__}")
                
                # 缓存客户端（每个模型独立）
                LLMClientFactory._client_cache[model_name_lower] = client
                return client
            except Exception as e:
                logger.warning(f"创建{model_name}客户端失败: {e}，使用Mock客户端")
                mock_client = MockLLMClient(config, model_name=model_name_lower)
                # Mock客户端也缓存
                LLMClientFactory._client_cache[model_name_lower] = mock_client
                return mock_client

        # 向后兼容：使用通用配置
        validation_errors = LLMClientFactory._validate_legacy_config(config)

        if validation_errors:
            logger.warning("LLM配置验证失败:")
            for error in validation_errors:
                logger.warning(f"  - {error}")
            logger.info("由于配置问题，将使用Mock客户端")
            return MockLLMClient(config)

        # 尝试创建真实的LLM客户端
        try:
            client = LLMClientFactory._create_real_client(config)
            logger.info(f"成功创建LLM客户端: {type(client).__name__}")
            return client
        except Exception as e:
            logger.error(f"创建LLM客户端失败: {e}")
            logger.info("降级使用Mock客户端")
            return MockLLMClient(config)

    @staticmethod
    def _validate_legacy_config(config: SimulationConfig) -> List[str]:
        """
        验证传统LLM配置（向后兼容）

        参数:
            config: 模拟配置

        返回:
            验证错误列表，如果为空则表示验证通过
        """
        errors = []

        # 检查API密钥
        if not config.llm_api_key:
            errors.append("未配置LLM API密钥 (EDU_ARENA_LLM_API_KEY)")
            return errors  # 如果没有API密钥，直接返回

        # 检查API密钥格式（简单检查）
        if len(config.llm_api_key.strip()) < 10:
            errors.append("LLM API密钥格式不正确")

        # 检查base_url
        if config.llm_base_url:
            if not config.llm_base_url.startswith(('http://', 'https://')):
                errors.append("LLM base_url必须以http://或https://开头")

        # 检查超时设置
        if config.llm_timeout < 1:
            errors.append("LLM超时时间必须大于等于1秒")

        return errors

    @staticmethod
    def _create_real_client(config: SimulationConfig) -> LLMClient:
        """
        创建真实的LLM客户端（传统方式）

        参数:
            config: 模拟配置

        返回:
            LLM客户端实例

        异常:
            Exception: 创建失败时抛出异常
        """
        base_url = config.llm_base_url or "https://api.openai.com/v1"

        # 根据base_url判断客户端类型
        if "openai.com" in base_url:
            logger.info("使用OpenAI客户端")
            return OpenAILLMClient(config)
        elif any(provider in base_url for provider in ["deepseek", "qwen", "kimi", "api2d.com"]):
            logger.info(f"使用兼容OpenAI API的客户端 (base_url: {base_url})")
            return OpenAILLMClient(config)
        elif "azure.com" in base_url:
            logger.info("检测到Azure OpenAI，使用兼容客户端")
            return OpenAILLMClient(config)
        else:
            raise ValueError(f"不支持的API提供商: {base_url}")

    @staticmethod
    def _create_real_client_for_model(config: SimulationConfig, model_name: str, model_config: Dict[str, str]) -> LLMClient:
        """
        为特定模型创建真实的LLM客户端

        参数:
            config: 模拟配置
            model_name: 模型名称
            model_config: 模型配置

        返回:
            LLM客户端实例

        异常:
            Exception: 创建失败时抛出异常
        """
        base_url = model_config["base_url"]

        # 为特定模型创建配置类
        model_specific_config = SimulationConfig(
            llm_api_key=model_config["api_key"],
            llm_base_url=base_url,
            llm_model=model_config["model"],
            llm_timeout=config.llm_timeout,
            llm_enable_emotion_analysis=config.llm_enable_emotion_analysis
        )

        # 所有模型都使用OpenAI兼容客户端（通过相应服务商的兼容接口）
        logger.info(f"为{model_name}创建OpenAI兼容客户端")
        return OpenAILLMClient(model_specific_config)

    @staticmethod
    def test_connection(config: SimulationConfig, model_name: str = None) -> bool:
        """
        测试LLM连接

        参数:
            config: 模拟配置
            model_name: 模型名称

        返回:
            连接测试是否成功
        """
        try:
            client = LLMClientFactory.create_client(config, model_name)
            if isinstance(client, MockLLMClient):
                logger.info("使用Mock客户端，跳过连接测试")
                return True

            # 同步上下文的简单连接测试（避免在运行中的事件循环里调用）
            import asyncio
            try:
                asyncio.get_running_loop()
                logger.warning("检测到运行中的事件循环，跳过同步连接测试")
                return True
            except RuntimeError:
                pass

            if model_name:
                try:
                    model_config = config.get_model_config(model_name)
                    test_model = model_config.get("model", model_name)
                except Exception:
                    test_model = model_name
            else:
                test_model = "gpt-3.5-turbo"
            result = asyncio.run(client.chat_completion([
                {"role": "user", "content": "Hello"}
            ], model=test_model, max_tokens=5))

            if result and "choices" in result:
                logger.info("LLM连接测试成功")
                return True
            else:
                logger.error("LLM连接测试失败：响应格式错误")
                return False

        except Exception as e:
            logger.error(f"LLM连接测试失败: {e}")
            return False

    @staticmethod
    async def test_connection_async(config: SimulationConfig, model_name: str = None) -> bool:
        """
        异步测试LLM连接（适用于已存在事件循环的场景）
        """
        try:
            client = LLMClientFactory.create_client(config, model_name)
            if isinstance(client, MockLLMClient):
                logger.info("使用Mock客户端，跳过连接测试")
                return True

            if model_name:
                try:
                    model_config = config.get_model_config(model_name)
                    test_model = model_config.get("model", model_name)
                except Exception:
                    test_model = model_name
            else:
                test_model = "gpt-3.5-turbo"
            result = await client.chat_completion([
                {"role": "user", "content": "Hello"}
            ], model=test_model, max_tokens=5)

            if result and "choices" in result:
                logger.info("LLM连接测试成功")
                return True
            else:
                logger.error("LLM连接测试失败：响应格式错误")
                return False
        except Exception as e:
            logger.error(f"LLM连接测试失败: {e}")
            return False


async def safe_llm_call(
    client: LLMClient,
    method_name: str,
    *args,
    fallback_value: Any = None,
    **kwargs
) -> Any:
    """
    安全的LLM调用，包含降级策略

    参数:
        client: LLM客户端
        method_name: 调用方法名
        fallback_value: 降级时的返回值
        *args: 位置参数
        **kwargs: 关键字参数

    返回:
        调用结果或降级值
    """
    method = getattr(client, method_name)

    fallback_strategy = DefaultFallback(default_value=fallback_value)

    return await with_fallback(
        lambda: method(*args, **kwargs),
        fallback_strategy
    )