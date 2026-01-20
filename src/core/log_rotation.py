"""
日志轮转管理器
支持按大小和时间轮转日志文件
"""
import os
import shutil
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LogRotationManager:
    """日志轮转管理器"""
    
    def __init__(
        self,
        log_file: str,
        max_size_mb: float = 100.0,
        max_backups: int = 10,
        rotate_on_start: bool = False
    ):
        """
        初始化日志轮转管理器
        
        参数:
            log_file: 日志文件路径
            max_size_mb: 最大文件大小（MB），超过此大小将轮转
            max_backups: 最大备份文件数量
            rotate_on_start: 启动时是否轮转现有日志
        """
        self.log_file = log_file
        self.max_size_bytes = max_size_mb * 1024 * 1024  # 转换为字节
        self.max_backups = max_backups
        self.rotate_on_start = rotate_on_start
    
    def should_rotate(self) -> bool:
        """
        检查是否需要轮转日志
        
        返回:
            是否需要轮转
        """
        if not os.path.exists(self.log_file):
            return False
        
        file_size = os.path.getsize(self.log_file)
        return file_size >= self.max_size_bytes
    
    def rotate(self) -> Optional[str]:
        """
        执行日志轮转
        
        返回:
            轮转后的备份文件名，如果轮转失败则返回None
        """
        if not os.path.exists(self.log_file):
            return None
        
        # 生成备份文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(self.log_file)
        dir_name = os.path.dirname(self.log_file)
        backup_name = f"{base_name}.{timestamp}.bak"
        backup_path = os.path.join(dir_name, backup_name)
        
        try:
            # 移动当前日志文件到备份
            shutil.move(self.log_file, backup_path)
            logger.info(f"日志文件已轮转: {self.log_file} -> {backup_path}")
            
            # 清理旧备份文件
            self._cleanup_old_backups()
            
            return backup_path
        except Exception as e:
            logger.error(f"日志轮转失败: {e}")
            return None
    
    def _cleanup_old_backups(self):
        """清理旧的备份文件"""
        if not os.path.exists(os.path.dirname(self.log_file)):
            return
        
        # 获取所有备份文件
        base_name = os.path.basename(self.log_file)
        dir_name = os.path.dirname(self.log_file)
        
        backup_files = []
        for filename in os.listdir(dir_name):
            if filename.startswith(base_name) and filename.endswith(".bak"):
                filepath = os.path.join(dir_name, filename)
                backup_files.append((filepath, os.path.getmtime(filepath)))
        
        # 按修改时间排序（最新的在前）
        backup_files.sort(key=lambda x: x[1], reverse=True)
        
        # 删除超过最大数量的备份文件
        if len(backup_files) > self.max_backups:
            for filepath, _ in backup_files[self.max_backups:]:
                try:
                    os.remove(filepath)
                    logger.debug(f"删除旧备份文件: {filepath}")
                except Exception as e:
                    logger.warning(f"删除旧备份文件失败: {filepath}, {e}")
    
    def ensure_rotation_on_start(self):
        """启动时确保日志轮转（如果启用）"""
        if self.rotate_on_start and os.path.exists(self.log_file):
            # 检查文件大小
            if os.path.getsize(self.log_file) > 0:
                self.rotate()
    
    def get_backup_files(self) -> list:
        """
        获取所有备份文件列表
        
        返回:
            备份文件路径列表
        """
        if not os.path.exists(os.path.dirname(self.log_file)):
            return []
        
        base_name = os.path.basename(self.log_file)
        dir_name = os.path.dirname(self.log_file)
        
        backup_files = []
        for filename in os.listdir(dir_name):
            if filename.startswith(base_name) and filename.endswith(".bak"):
                filepath = os.path.join(dir_name, filename)
                backup_files.append(filepath)
        
        # 按修改时间排序（最新的在前）
        backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return backup_files
