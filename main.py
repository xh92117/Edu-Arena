import asyncio
import os
import sys
import signal
import atexit
from pathlib import Path
from src.core.runner import create_runner_for_all_models

# PID 文件路径（存放在项目根目录）
PID_FILE = Path(__file__).parent / ".main_pid.lock"


def kill_previous_instance():
    """终止之前运行的实例"""
    if not PID_FILE.exists():
        return
    
    try:
        with open(PID_FILE, 'r') as f:
            old_pid = int(f.read().strip())
        
        # 检查进程是否存在并终止
        if sys.platform == 'win32':
            # Windows 平台
            import subprocess
            # 先检查进程是否存在
            result = subprocess.run(
                ['tasklist', '/FI', f'PID eq {old_pid}', '/NH'],
                capture_output=True, text=True
            )
            if str(old_pid) in result.stdout:
                subprocess.run(['taskkill', '/F', '/PID', str(old_pid)], 
                             capture_output=True)
                print(f"已终止之前的运行实例 (PID: {old_pid})")
        else:
            # Unix/Linux/macOS 平台
            try:
                os.kill(old_pid, signal.SIGTERM)
                print(f"已终止之前的运行实例 (PID: {old_pid})")
            except ProcessLookupError:
                pass  # 进程已不存在
            except PermissionError:
                print(f"警告: 无权限终止进程 {old_pid}")
    except (ValueError, FileNotFoundError, IOError):
        pass  # 文件无效或不存在，忽略
    finally:
        # 删除旧的 PID 文件
        try:
            PID_FILE.unlink()
        except FileNotFoundError:
            pass


def write_pid_file():
    """写入当前进程的 PID"""
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))


def cleanup_pid_file():
    """清理 PID 文件"""
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


async def main():
    """主程序入口，运行所有7个模型的平行模拟环境"""
    runner = create_runner_for_all_models()
    await runner.run_simulation()


if __name__ == "__main__":
    # 单实例机制：终止之前的运行实例
    kill_previous_instance()
    
    # 写入当前 PID
    write_pid_file()
    
    # 注册退出时清理 PID 文件
    atexit.register(cleanup_pid_file)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n模拟已被用户中断")
    finally:
        cleanup_pid_file()