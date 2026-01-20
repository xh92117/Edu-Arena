import asyncio
import time
import json
import os
from datetime import date
from typing import List, Dict, Any

from src.engine.simulation import SimulationEnv
from src.core.config import SimulationConfig, get_config_for_all_models, get_config_for_specific_models
from src.core.llm_client import LLMClientFactory
from src.core.log_rotation import LogRotationManager
from src.core.memory_manager import MemoryManager
from src.core.concurrency_control import ConcurrencyController, RateLimiter, ErrorIsolator
from src.agents.factory import AgentFactory
import logging

# 尝试导入aiofiles，如果不可用则使用同步方式
try:
    import aiofiles
    ASYNC_FILE_AVAILABLE = True
except ImportError:
    ASYNC_FILE_AVAILABLE = False
    logging.warning("aiofiles未安装，使用同步文件写入（建议: pip install aiofiles）")

logger = logging.getLogger(__name__)


class SimulationRunner:
    """统一模拟运行器，管理多个模拟环境的并发执行"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._ensure_log_directory()
        # 初始化日志轮转管理器
        self.log_rotation = LogRotationManager(
            log_file=self.config.log_file,
            max_size_mb=self.config.log_max_size_mb,
            max_backups=self.config.log_max_backups,
            rotate_on_start=self.config.log_rotate_on_start
        )
        
        # 初始化并发控制器
        rate_limiter = None
        error_isolator = None
        
        # 如果启用速率限制
        if self.config.enable_rate_limiting:
            rate_limiter = RateLimiter(
                max_calls=self.config.rate_limit_max_calls,
                time_window=self.config.rate_limit_window
            )
        
        # 如果启用错误隔离
        if self.config.enable_error_isolation:
            error_isolator = ErrorIsolator(
                max_failures=self.config.error_isolation_max_failures,
                failure_window=self.config.error_isolation_window
            )
        
        self.concurrency_controller = ConcurrencyController(
            max_concurrent=self.config.max_concurrent_tasks,
            rate_limiter=rate_limiter,
            error_isolator=error_isolator
        )
        
        # 初始化内存管理器
        self.memory_manager = MemoryManager(
            cleanup_interval_seconds=self.config.memory_cleanup_interval,
            max_memory_mb=self.config.memory_max_usage_mb,
            enable_persistence=self.config.enable_memory_persistence,
            persistence_dir=self.config.memory_persistence_dir
        )

    def _ensure_log_directory(self):
        """确保日志目录存在"""
        os.makedirs(self.config.log_dir, exist_ok=True)

    async def validate_configuration(self) -> bool:
        """
        验证配置并输出结果

        返回:
            配置是否有效
        """
        print("=== 配置验证 ===")

        # 验证各模型配置
        available_models = self.config.get_available_models()
        print(f"[OK] 已配置模型: {available_models}")
        print(f"[INFO] 配置模型数量: {len(available_models)}/{len(self.config.supported_models)}")

        if not available_models:
            print("[WARN] 没有配置任何LLM，所有模型将使用Mock客户端")
        else:
            # 测试每个已配置模型的连接
            connection_tests = []
            for model in available_models:
                try:
                    if await LLMClientFactory.test_connection_async(self.config, model):
                        connection_tests.append(f"{model}: ✅")
                        print(f"[OK] {model}连接测试: 成功")
                    else:
                        connection_tests.append(f"{model}: ❌")
                        print(f"[ERR] {model}连接测试: 失败")
                except Exception as e:
                    connection_tests.append(f"{model}: ❌")
                    print(f"[ERR] {model}连接测试失败: {e}")

            successful_connections = sum(1 for test in connection_tests if "✅" in test)
            print(f"[INFO] 连接测试结果: {successful_connections}/{len(available_models)} 成功")

            if successful_connections == 0:
                print("[WARN] 所有模型连接测试失败，将使用Mock客户端")

        # 验证其他配置
        print(f"[OK] 环境数量: {self.config.num_environments}")
        print(f"[OK] 模拟速度: {self.config.get_simulation_speed_info()}")
        print(f"[OK] 日志目录: {self.config.log_dir}")
        print(f"[OK] 日志文件: {os.path.basename(self.config.log_file)}")

        print("=== 配置验证完成 ===\n")
        
        # 打印模型决策模式状态
        AgentFactory.set_config(self.config)
        AgentFactory.print_model_status(self.config)
        
        return True

    def _get_models_to_run(self) -> List[str]:
        """获取要运行的模型列表"""
        return self.config.get_models_to_run()

    def _create_environments(self) -> List[SimulationEnv]:
        """创建模拟环境"""
        models = self._get_models_to_run()
        environments = []

        for i, model_name in enumerate(models):
            env = SimulationEnv(
                env_id=i,
                start_date=date(2010, 1, 1),
                model_name=model_name,
                config=self.config
            )
            environments.append(env)

        return environments

    async def run_single_environment(self, env: SimulationEnv) -> Dict[str, Any]:
        """运行单个环境的一周模拟"""
        try:
            return await env.run_week()
        except Exception as e:
            # 错误处理，确保模拟继续运行
            print(f"环境 {env.env_id} 发生错误: {e}")
            return {
                "timestamp": str(date.today()),
                "env_id": env.env_id,
                "error": str(e)
            }

    async def _write_logs_async(self, results: List[Any]) -> None:
        """
        异步写入日志（如果aiofiles可用）
        
        参数:
            results: 模拟结果列表
        """
        log_entries = [self._process_log_entry(result) for result in results]
        
        if ASYNC_FILE_AVAILABLE:
            # 使用异步文件写入
            async with aiofiles.open(self.config.log_file, "a", encoding="utf-8") as log_file:
                for log_entry in log_entries:
                    await log_file.write(log_entry + "\n")
        else:
            # 降级到同步写入
            with open(self.config.log_file, "a", encoding="utf-8") as log_file:
                for log_entry in log_entries:
                    log_file.write(log_entry + "\n")
                log_file.flush()
    
    def _process_log_entry(self, result: Any) -> str:
        """处理单个日志条目，返回JSON字符串"""
        # 自定义JSON编码器，处理date对象
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                return super().default(obj)

        if isinstance(result, Exception):
            # 处理异常情况
            error_log = {
                "timestamp": str(date.today()),
                "error": str(result)
            }
            return json.dumps(error_log, ensure_ascii=False)
        else:
            # 处理正常结果，直接使用中文输出
            return json.dumps(result, ensure_ascii=False, cls=DateTimeEncoder)

    async def run_simulation(self):
        """运行模拟的主循环"""
        # 配置验证
        if not await self.validate_configuration():
            print("配置验证失败，退出程序")
            return

        # 每次启动都创建新的日志文件（带时间戳）
        print(f"[INFO] 创建新日志文件: {self.config.log_file}")
        
        # 确保日志文件是新的（清空或创建）
        with open(self.config.log_file, 'w', encoding='utf-8') as f:
            pass  # 创建空文件

        # 创建模拟环境
        envs = self._create_environments()

        models = self._get_models_to_run()
        print(f"初始化完成，启动 {len(envs)} 个平行模拟环境...")
        print(f"模拟模型: {models}")
        print(f"模拟速度: {self.config.get_simulation_speed_info()}")

        # 主循环：持续运行模拟
        while True:
            # 记录开始时间，用于计算执行时间
            start_time = time.time()

            # 并行运行所有环境的一周模拟（使用并发控制）
            async def run_with_control(env):
                """包装函数，用于并发控制"""
                return await self.concurrency_controller.execute(
                    self.run_single_environment,
                    env
                )
            
            tasks = [run_with_control(env) for env in envs]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 检查是否需要轮转日志
            if self.log_rotation.should_rotate():
                self.log_rotation.rotate()
            
            # 处理结果并写入日志（异步写入，避免阻塞）
            await self._write_logs_async(results)

            # 定期内存清理
            if self.memory_manager.should_cleanup():
                cleanup_stats = self.memory_manager.cleanup()
                if cleanup_stats.get("cleaned"):
                    logger.debug(f"内存清理: {cleanup_stats}")
            
            # 计算实际执行时间
            actual_execution_time = time.time() - start_time

            # 计算需要睡眠的时间，确保每周模拟时间为指定小时数
            required_sleep_time = (self.config.simulation_speed * 3600) - actual_execution_time

            if required_sleep_time > 0:
                # 睡眠剩余时间，确保模拟速度符合要求
                print(f"本周模拟完成，实际执行时间: {actual_execution_time:.2f}秒，睡眠: {required_sleep_time:.2f}秒")
                await asyncio.sleep(required_sleep_time)
            else:
                # 如果执行时间超过了要求，不睡眠，直接继续下一周
                required_time = self.config.simulation_speed * 3600
                print(f"本周模拟完成，实际执行时间: {actual_execution_time:.2f}秒，超过了要求的{required_time:.2f}秒")


def create_runner_for_all_models() -> SimulationRunner:
    """创建运行所有模型的运行器"""
    config = get_config_for_all_models()
    return SimulationRunner(config)


def create_runner_for_specific_models(models: List[str]) -> SimulationRunner:
    """创建运行指定模型的运行器"""
    config = get_config_for_specific_models(models)
    return SimulationRunner(config)