#!/usr/bin/env python3
"""
Edu-Arena 配置检查脚本

检查LLM API配置和系统环境
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import SimulationConfig
from src.core.llm_client import LLMClientFactory


def check_environment():
    """检查系统环境"""
    print("🔍 Edu-Arena 配置检查工具")
    print("=" * 50)

    # 检查Python版本
    print(f"✅ Python版本: {sys.version.split()[0]}")

    # 检查项目结构
    required_dirs = ['src', 'logs', 'examples']
    for dir_name in required_dirs:
        if (project_root / dir_name).exists():
            print(f"✅ 目录存在: {dir_name}/")
        else:
            print(f"❌ 目录缺失: {dir_name}/")

    # 检查关键文件
    required_files = ['main.py', 'requirements.txt', 'README.md']
    for file_name in required_files:
        if (project_root / file_name).exists():
            print(f"✅ 文件存在: {file_name}")
        else:
            print(f"❌ 文件缺失: {file_name}")

    print()


def check_configuration():
    """检查配置"""
    print("⚙️ 配置检查")
    print("-" * 30)

    try:
        config = SimulationConfig()

        # 基本配置检查
        print(f"✅ 环境数量: {config.num_environments}")
        print(f"✅ 模拟速度: {config.get_simulation_speed_info()}")
        print(f"✅ 日志目录: {config.log_dir}")

        # LLM配置检查
        print(f"\n🤖 LLM配置检查")
        print("-" * 20)

        available_models = config.get_available_models()
        if available_models:
            print(f"✅ 已配置模型 ({len(available_models)}/{len(config.supported_models)}):")
            for model in available_models:
                print(f"   • {model}")
        else:
            print("⚠️  没有配置任何LLM模型，将使用Mock客户端")

        # 连接测试
        print(f"\n🔌 连接测试")
        print("-" * 15)

        test_results = []
        for model in config.supported_models:
            try:
                success = LLMClientFactory.test_connection(config, model)
                status = "✅" if success else "❌"
                test_results.append(f"{model}: {status}")
            except Exception as e:
                test_results.append(f"{model}: ❌ ({str(e)[:30]}...)")

        for result in test_results:
            print(f"   {result}")

    except Exception as e:
        print(f"❌ 配置检查失败: {e}")
        return False

    print()
    return True


def check_dependencies():
    """检查依赖"""
    print("📦 依赖检查")
    print("-" * 20)

    required_packages = [
        'streamlit',
        'pydantic',
        'plotly',
        'pandas',
        'asyncio'
    ]

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (未安装)")

    print()


def main():
    """主函数"""
    check_environment()
    success = check_configuration()
    check_dependencies()

    print("🎯 检查完成")
    print("-" * 20)

    if success:
        print("✅ 系统配置正常，可以开始运行！")
        print("\n🚀 快速开始:")
        print("   python main.py                    # 运行所有模型")
        print("   python run_specific_models.py     # 运行指定模型")
        print("   streamlit run src/ui/dashboard.py # 启动可视化界面")
    else:
        print("❌ 配置存在问题，请检查上述错误信息")

    print(f"\n📖 更多信息请查看: README.md")


if __name__ == "__main__":
    main()