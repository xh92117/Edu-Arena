"""
异步并发控制模块
提供限流、错误隔离和并发管理功能
"""
import asyncio
from typing import Callable, Any, Optional, List
from datetime import datetime, timedelta
import logging
from collections import deque

logger = logging.getLogger(__name__)


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, max_calls: int, time_window: float):
        """
        初始化速率限制器
        
        参数:
            max_calls: 时间窗口内的最大调用次数
            time_window: 时间窗口（秒）
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """获取调用许可，如果超过限制则等待"""
        async with self.lock:
            now = datetime.now()
            
            # 清理过期记录
            while self.calls and (now - self.calls[0]).total_seconds() > self.time_window:
                self.calls.popleft()
            
            # 如果超过限制，等待
            if len(self.calls) >= self.max_calls:
                oldest_call = self.calls[0]
                wait_time = self.time_window - (now - oldest_call).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # 重新清理过期记录
                    now = datetime.now()
                    while self.calls and (now - self.calls[0]).total_seconds() > self.time_window:
                        self.calls.popleft()
            
            # 记录本次调用
            self.calls.append(now)


class ErrorIsolator:
    """错误隔离器"""
    
    def __init__(self, max_failures: int = 5, failure_window: float = 60.0):
        """
        初始化错误隔离器
        
        参数:
            max_failures: 时间窗口内的最大失败次数
            failure_window: 时间窗口（秒）
        """
        self.max_failures = max_failures
        self.failure_window = failure_window
        self.failures = deque()
        self.isolated = False
        self.isolation_until: Optional[datetime] = None
        self.lock = asyncio.Lock()
    
    async def record_failure(self):
        """记录一次失败"""
        async with self.lock:
            now = datetime.now()
            self.failures.append(now)
            
            # 清理过期记录
            while self.failures and (now - self.failures[0]).total_seconds() > self.failure_window:
                self.failures.popleft()
            
            # 检查是否需要隔离
            if len(self.failures) >= self.max_failures:
                self.isolated = True
                self.isolation_until = now + timedelta(seconds=self.failure_window)
                logger.warning(f"错误隔离激活，将在{self.isolation_until}后恢复")
    
    async def record_success(self):
        """记录一次成功，可能解除隔离"""
        async with self.lock:
            if self.isolated and self.isolation_until:
                if datetime.now() >= self.isolation_until:
                    self.isolated = False
                    self.isolation_until = None
                    self.failures.clear()
                    logger.info("错误隔离已解除")
    
    async def is_isolated(self) -> bool:
        """检查是否处于隔离状态"""
        async with self.lock:
            if self.isolated and self.isolation_until:
                if datetime.now() >= self.isolation_until:
                    # 隔离期已过，自动解除
                    self.isolated = False
                    self.isolation_until = None
                    self.failures.clear()
                    return False
                return True
            return False


class ConcurrencyController:
    """并发控制器"""
    
    def __init__(
        self,
        max_concurrent: int = 10,
        rate_limiter: Optional[RateLimiter] = None,
        error_isolator: Optional[ErrorIsolator] = None
    ):
        """
        初始化并发控制器
        
        参数:
            max_concurrent: 最大并发数
            rate_limiter: 速率限制器（可选）
            error_isolator: 错误隔离器（可选）
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = rate_limiter
        self.error_isolator = error_isolator
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        执行函数，应用并发控制
        
        参数:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        返回:
            函数执行结果
            
        异常:
            如果函数执行失败，会记录到错误隔离器
        """
        # 检查错误隔离（优雅降级，不中断模拟）
        if self.error_isolator and await self.error_isolator.is_isolated():
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("服务处于错误隔离状态，跳过本次执行")
            # 返回None表示跳过，由调用者处理
            return None
        
        # 获取并发许可
        async with self.semaphore:
            # 应用速率限制
            if self.rate_limiter:
                await self.rate_limiter.acquire()
            
            try:
                # 执行函数
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # 记录成功
                if self.error_isolator:
                    await self.error_isolator.record_success()
                
                return result
            except Exception as e:
                # 记录失败
                if self.error_isolator:
                    await self.error_isolator.record_failure()
                raise


class BatchProcessor:
    """批处理器"""
    
    def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
        """
        初始化批处理器
        
        参数:
            batch_size: 批处理大小
            max_concurrent: 最大并发批次数
        """
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(
        self,
        items: List[Any],
        processor: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """
        批处理项目列表
        
        参数:
            items: 要处理的项目列表
            processor: 处理函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        返回:
            处理结果列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            async with self.semaphore:
                # 并发处理批次内的项目
                batch_tasks = []
                for item in batch:
                    if asyncio.iscoroutinefunction(processor):
                        task = processor(item, *args, **kwargs)
                    else:
                        task = asyncio.to_thread(processor, item, *args, **kwargs)
                    batch_tasks.append(task)
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                results.extend(batch_results)
        
        return results
