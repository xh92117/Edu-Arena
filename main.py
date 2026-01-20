import asyncio
from src.core.runner import create_runner_for_all_models


async def main():
    """主程序入口，运行所有7个模型的平行模拟环境"""
    runner = create_runner_for_all_models()
    await runner.run_simulation()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n模拟已被用户中断")