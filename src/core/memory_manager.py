"""
内存管理模块
提供定期清理和持久化功能
"""
import gc
import weakref
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import json
import os

logger = logging.getLogger(__name__)


class MemoryManager:
    """内存管理器"""
    
    def __init__(
        self,
        cleanup_interval_seconds: float = 3600.0,  # 1小时
        max_memory_mb: Optional[float] = None,
        enable_persistence: bool = False,
        persistence_dir: str = "data/cache"
    ):
        """
        初始化内存管理器
        
        参数:
            cleanup_interval_seconds: 清理间隔（秒）
            max_memory_mb: 最大内存使用（MB），超过此值触发清理
            enable_persistence: 是否启用持久化
            persistence_dir: 持久化目录
        """
        self.cleanup_interval = cleanup_interval_seconds
        self.max_memory_mb = max_memory_mb
        self.enable_persistence = enable_persistence
        self.persistence_dir = persistence_dir
        self.last_cleanup_time = datetime.now()
        self.weak_refs: List[weakref.ref] = []
        
        if enable_persistence:
            os.makedirs(persistence_dir, exist_ok=True)
    
    def register_object(self, obj: Any, key: str = None):
        """
        注册对象到弱引用列表
        
        参数:
            obj: 要注册的对象
            key: 可选的键名
        """
        ref = weakref.ref(obj, lambda r: self._on_object_deleted(key))
        self.weak_refs.append(ref)
    
    def _on_object_deleted(self, key: Optional[str]):
        """对象被删除时的回调"""
        if key:
            logger.debug(f"对象已删除: {key}")
    
    def should_cleanup(self) -> bool:
        """检查是否需要清理"""
        now = datetime.now()
        time_since_cleanup = (now - self.last_cleanup_time).total_seconds()
        return time_since_cleanup >= self.cleanup_interval
    
    def cleanup(self, force: bool = False) -> Dict[str, Any]:
        """
        执行内存清理
        
        参数:
            force: 是否强制清理
            
        返回:
            清理统计信息
        """
        if not force and not self.should_cleanup():
            return {"cleaned": False, "reason": "未到清理时间"}
        
        stats = {
            "cleaned": True,
            "before_gc": self._get_memory_usage(),
            "weak_refs_before": len(self.weak_refs)
        }
        
        # 清理弱引用列表中的无效引用
        self.weak_refs = [ref for ref in self.weak_refs if ref() is not None]
        stats["weak_refs_after"] = len(self.weak_refs)
        
        # 执行Python垃圾回收
        collected = gc.collect()
        stats["gc_collected"] = collected
        
        stats["after_gc"] = self._get_memory_usage()
        stats["memory_freed_mb"] = stats["before_gc"] - stats["after_gc"]
        
        self.last_cleanup_time = datetime.now()
        
        logger.info(f"内存清理完成: 释放 {stats['memory_freed_mb']:.2f} MB, GC回收 {collected} 个对象")
        
        return stats
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用（MB）"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # 如果没有psutil，返回0
            return 0.0
    
    def persist_data(self, key: str, data: Any) -> bool:
        """
        持久化数据到磁盘
        
        参数:
            key: 数据键
            data: 要持久化的数据
            
        返回:
            是否成功
        """
        if not self.enable_persistence:
            return False
        
        try:
            filepath = os.path.join(self.persistence_dir, f"{key}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, default=str)
            logger.debug(f"数据已持久化: {key}")
            return True
        except Exception as e:
            logger.error(f"持久化数据失败: {key}, {e}")
            return False
    
    def load_persisted_data(self, key: str) -> Optional[Any]:
        """
        从磁盘加载持久化数据
        
        参数:
            key: 数据键
            
        返回:
            数据对象，如果不存在则返回None
        """
        if not self.enable_persistence:
            return None
        
        try:
            filepath = os.path.join(self.persistence_dir, f"{key}.json")
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"数据已加载: {key}")
            return data
        except Exception as e:
            logger.error(f"加载持久化数据失败: {key}, {e}")
            return None
    
    def clear_persisted_data(self, key: Optional[str] = None) -> int:
        """
        清除持久化数据
        
        参数:
            key: 要清除的数据键，如果为None则清除所有
            
        返回:
            清除的文件数量
        """
        if not self.enable_persistence:
            return 0
        
        try:
            if key:
                filepath = os.path.join(self.persistence_dir, f"{key}.json")
                if os.path.exists(filepath):
                    os.remove(filepath)
                    return 1
                return 0
            else:
                # 清除所有
                count = 0
                for filename in os.listdir(self.persistence_dir):
                    if filename.endswith('.json'):
                        os.remove(os.path.join(self.persistence_dir, filename))
                        count += 1
                logger.info(f"已清除 {count} 个持久化文件")
                return count
        except Exception as e:
            logger.error(f"清除持久化数据失败: {e}")
            return 0
