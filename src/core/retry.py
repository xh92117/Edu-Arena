import asyncio
import logging
from typing import Callable, Any, Optional, TypeVar, Union
from functools import wraps
import random
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryError(Exception):
    """重试失败异常"""
    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


async def exponential_backoff_with_jitter(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    计算指数退避延迟时间，包含抖动以避免惊群效应

    参数:
        attempt: 当前重试次数 (从1开始)
        base_delay: 基础延迟时间(秒)
        max_delay: 最大延迟时间(秒)

    返回:
        延迟时间(秒)
    """
    # 指数退避: base_delay * (2 ^ (attempt - 1))
    delay = base_delay * (2 ** (attempt - 1))

    # 添加抖动: ±25%的随机变化
    jitter = delay * 0.25 * (random.random() * 2 - 1)
    delay = delay + jitter

    # 限制最大延迟
    delay = min(delay, max_delay)

    return delay


async def retry_async(
    func: Callable[..., Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_backoff: bool = True,
    retry_on: Optional[tuple] = None,
    logger: Optional[logging.Logger] = None
) -> Any:
    """
    异步函数重试装饰器

    参数:
        func: 要重试的异步函数
        max_retries: 最大重试次数
        base_delay: 基础延迟时间(秒)
        max_delay: 最大延迟时间(秒)
        exponential_backoff: 是否使用指数退避
        retry_on: 指定重试的异常类型，为None时重试所有异常
        logger: 日志记录器

    返回:
        函数执行结果

    异常:
        RetryError: 重试失败时抛出
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    last_exception = None

    for attempt in range(max_retries + 1):  # +1 因为第一次执行也算一次尝试
        try:
            if attempt > 0:
                if exponential_backoff:
                    delay = await exponential_backoff_with_jitter(attempt, base_delay, max_delay)
                else:
                    delay = base_delay

                logger.warning(f"第{attempt}次重试，延迟{delay:.2f}秒...")
                await asyncio.sleep(delay)

            return await func()

        except Exception as e:
            last_exception = e

            # 检查是否应该重试此异常
            if retry_on is not None and not isinstance(e, retry_on):
                logger.error(f"异常类型 {type(e).__name__} 不在重试列表中，直接失败")
                raise RetryError(f"函数执行失败: {e}", e) from e

            if attempt < max_retries:
                logger.warning(f"函数执行失败 ({type(e).__name__}): {e}")
            else:
                logger.error(f"重试{max_retries}次后仍然失败")

    # 所有重试都失败了
    raise RetryError(f"函数执行失败，重试{max_retries}次后仍然失败", last_exception)


def retry_async_decorator(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_backoff: bool = True,
    retry_on: Optional[tuple] = None
):
    """
    异步重试装饰器

    参数:
        max_retries: 最大重试次数
        base_delay: 基础延迟时间(秒)
        max_delay: 最大延迟时间(秒)
        exponential_backoff: 是否使用指数退避
        retry_on: 指定重试的异常类型

    返回:
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_async(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_backoff=exponential_backoff,
                retry_on=retry_on
            )
        return wrapper
    return decorator


class CircuitBreaker:
    """
    熔断器实现
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: tuple = (Exception,)
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open

    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置熔断器"""
        if self.state != 'open':
            return False

        if self.last_failure_time is None:
            return True

        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _record_success(self):
        """记录成功调用"""
        if self.state == 'half_open':
            self.state = 'closed'
            self.failure_count = 0
            logger.info("熔断器重置为关闭状态")

    def _record_failure(self):
        """记录失败调用"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            logger.warning(f"熔断器打开，失败次数达到阈值: {self.failure_count}")

    async def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        调用函数，通过熔断器控制

        参数:
            func: 要调用的函数
            *args: 位置参数
            **kwargs: 关键字参数

        返回:
            函数执行结果

        异常:
            CircuitBreakerOpen: 熔断器打开时抛出
        """
        if self.state == 'open':
            if self._should_attempt_reset():
                self.state = 'half_open'
                logger.info("熔断器进入半开状态，尝试恢复")
            else:
                raise CircuitBreakerOpen("熔断器已打开")

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except self.expected_exception as e:
            self._record_failure()
            raise


class CircuitBreakerOpen(Exception):
    """熔断器打开异常"""
    pass


class FallbackStrategy:
    """
    降级策略基类
    """

    async def execute(self, *args, **kwargs) -> Any:
        """执行降级策略"""
        raise NotImplementedError


class DefaultFallback(FallbackStrategy):
    """
    默认降级策略：返回默认值或抛出异常
    """

    def __init__(self, default_value: Any = None, raise_exception: bool = False):
        self.default_value = default_value
        self.raise_exception = raise_exception

    async def execute(self, *args, **kwargs) -> Any:
        if self.raise_exception:
            raise RuntimeError("服务降级，无法提供服务")
        return self.default_value


async def with_fallback(
    primary_func: Callable[..., Any],
    fallback_strategy: FallbackStrategy,
    *args,
    **kwargs
) -> Any:
    """
    执行函数，失败时使用降级策略

    参数:
        primary_func: 主函数
        fallback_strategy: 降级策略
        *args: 位置参数
        **kwargs: 关键字参数

    返回:
        函数执行结果或降级结果
    """
    try:
        return await primary_func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"主函数执行失败，使用降级策略: {e}")
        return await fallback_strategy.execute(*args, **kwargs)


# LLM调用专用重试和降级配置
LLM_RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 1.0,
    "max_delay": 30.0,
    "retry_on": (ConnectionError, TimeoutError, RuntimeError)
}

LLM_CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 5,
    "recovery_timeout": 60.0
}