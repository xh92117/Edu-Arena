import logging
from typing import Dict, Any, Optional, List
from collections import deque
from src.core.state import ChildState, FamilyState
from src.core.state_manager import StateManager
from src.core.llm_client import LLMClient, safe_llm_call
from src.core.llm_evaluator import LLMEvaluator, create_llm_evaluator, ActionEvaluationResult
import random

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dungeon_master.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 行为影响规则字典：定义不同行为类型对孩子状态的影响
# 现在使用具体的家庭成员关系替代通用的parent_relationship
ACTION_EFFECTS: Dict[str, Dict[str, float]] = {
    # ==================== 婴幼儿期专属行为 (0-3岁) ====================
    "亲子互动": {
        "knowledge": 0.2,       # 早期认知发展
        "stress": -2.5,         # 安全感带来的放松
        "father_relationship": 3.5,  # 亲密接触增进感情
        "mother_relationship": 3.5,
        "grandfather_relationship": 2.5,
        "grandmother_relationship": 2.5,
        "physical_health": 0.5,
        "_description": "抱抱、逗玩、眼神交流、肌肤接触"
    },
    "日常照料": {
        "knowledge": 0.1,       # 基础认知
        "stress": -1.5,         # 被照顾的安心感
        "father_relationship": 2.0,
        "mother_relationship": 2.5,  # 母亲照料更亲密
        "grandfather_relationship": 1.5,
        "grandmother_relationship": 2.0,
        "physical_health": 1.5,  # 健康照料
        "_description": "喂奶、换尿布、哄睡、洗澡"
    },
    "感官刺激": {
        "knowledge": 0.4,       # 感官发育促进认知
        "stress": -1.0,
        "father_relationship": 1.5,
        "mother_relationship": 1.5,
        "grandfather_relationship": 1.2,
        "grandmother_relationship": 1.2,
        "physical_health": 0.3,
        "_description": "听音乐、看色彩、触摸不同材质、听故事"
    },
    "户外活动": {
        "knowledge": 0.3,       # 探索世界
        "stress": -3.0,         # 新鲜空气和阳光
        "father_relationship": 2.5,
        "mother_relationship": 2.5,
        "grandfather_relationship": 2.0,
        "grandmother_relationship": 2.0,
        "physical_health": 2.5,  # 身体发育
        "_description": "推婴儿车晒太阳、公园玩耍、看花草"
    },
    "早期阅读": {
        "knowledge": 0.5,       # 语言启蒙
        "stress": -0.5,
        "father_relationship": 2.0,
        "mother_relationship": 2.0,
        "grandfather_relationship": 1.8,
        "grandmother_relationship": 1.8,
        "physical_health": 0.0,
        "_description": "读绘本、讲故事、指认图片"
    },
    "安抚陪伴": {
        "knowledge": 0.0,
        "stress": -4.0,         # 情绪安抚效果显著
        "father_relationship": 3.0,
        "mother_relationship": 3.5,
        "grandfather_relationship": 2.5,
        "grandmother_relationship": 3.0,
        "physical_health": 0.5,
        "_description": "哭闹时的安慰、夜间陪伴、轻声哼歌"
    },
    "社交接触": {
        "knowledge": 0.2,       # 社会认知启蒙
        "stress": 0.5,          # 陌生环境略有压力
        "father_relationship": 1.0,
        "mother_relationship": 1.0,
        "grandfather_relationship": 0.8,
        "grandmother_relationship": 0.8,
        "physical_health": 0.3,
        "_description": "带孩子见亲戚朋友、接触其他小朋友"
    },
    # ==================== 通用行为 ====================
    "辅导": {
        "knowledge": 1.5,       # 知识储备增加1.5点
        "stress": 2.0,          # 压力增加2.0点
        "father_relationship": 0.5,  # 与父亲关系增加0.5点
        "mother_relationship": 0.5,  # 与母亲关系增加0.5点
        "grandfather_relationship": 0.3,  # 与祖父关系增加0.3点
        "grandmother_relationship": 0.3,  # 与祖母关系增加0.3点
        "physical_health": 0.0  # 身体素质无变化
    },
    "鼓励": {
        "knowledge": 0.0,
        "stress": -3.0,         # 压力减少3.0点
        "father_relationship": 2.0,  # 与父亲关系增加2.0点
        "mother_relationship": 2.0,  # 与母亲关系增加2.0点
        "grandfather_relationship": 1.5,  # 与祖父关系增加1.5点
        "grandmother_relationship": 1.5,  # 与祖母关系增加1.5点
        "physical_health": 0.5  # 身体素质略有提升
    },
    "花钱培训": {
        "knowledge": 3.0,       # 知识储备增加3.0点
        "stress": 4.0,          # 压力增加4.0点
        "father_relationship": 0.0,
        "mother_relationship": 0.0,
        "grandfather_relationship": 0.0,
        "grandmother_relationship": 0.0,
        "physical_health": -1.0  # 身体素质略有下降（因为学习时间增加）
    },
    "陪伴": {
        "knowledge": 0.0,
        "stress": -4.0,         # 压力减少4.0点
        "father_relationship": 3.0,  # 与父亲关系大幅增加
        "mother_relationship": 3.0,  # 与母亲关系大幅增加
        "grandfather_relationship": 2.5,  # 与祖父关系增加2.5点
        "grandmother_relationship": 2.5,  # 与祖母关系增加2.5点
        "physical_health": 1.0  # 身体素质提升
    },
    "严格要求": {
        "knowledge": 2.0,       # 知识储备增加2.0点
        "stress": 6.0,          # 压力大幅增加
        "father_relationship": -1.5,  # 与父亲关系下降
        "mother_relationship": -1.5,  # 与母亲关系下降
        "grandfather_relationship": -1.0,  # 与祖父关系下降
        "grandmother_relationship": -1.0,  # 与祖母关系下降
        "physical_health": -0.5  # 身体素质略有下降
    },
    "监督学习": {
        "knowledge": 2.5,       # 知识储备增加2.5点
        "stress": 5.0,          # 压力增加5.0点
        "father_relationship": -0.5,  # 与父亲关系略有下降
        "mother_relationship": -0.5,  # 与母亲关系略有下降
        "grandfather_relationship": -0.3,  # 与祖父关系略有下降
        "grandmother_relationship": -0.3,  # 与祖母关系略有下降
        "physical_health": -0.5  # 身体素质略有下降
    },
    "沟通": {
        "knowledge": 0.0,
        "stress": -2.5,         # 压力减少2.5点
        "father_relationship": 2.5,  # 与父亲关系增加
        "mother_relationship": 2.5,  # 与母亲关系增加
        "grandfather_relationship": 2.0,  # 与祖父关系增加
        "grandmother_relationship": 2.0,  # 与祖母关系增加
        "physical_health": 0.5  # 身体素质略有提升
    },
    "健康教育": {
        "knowledge": 0.0,
        "stress": -1.0,         # 压力减少1.0点
        "father_relationship": 1.0,  # 与父亲关系增加
        "mother_relationship": 1.0,  # 与母亲关系增加
        "grandfather_relationship": 0.8,  # 与祖父关系增加
        "grandmother_relationship": 0.8,  # 与祖母关系增加
        "physical_health": 3.0  # 身体素质大幅提升
    },
    "创新活动": {
        "knowledge": 1.5,       # 知识储备增加1.5点
        "stress": -2.0,         # 压力减少2.0点
        "father_relationship": 2.0,  # 与父亲关系增加
        "mother_relationship": 2.0,  # 与母亲关系增加
        "grandfather_relationship": 1.5,  # 与祖父关系增加
        "grandmother_relationship": 1.5,  # 与祖母关系增加
        "physical_health": 1.0  # 身体素质提升
    },
    "个性化计划": {
        "knowledge": 2.0,       # 知识储备增加2.0点
        "stress": 3.0,          # 压力增加3.0点
        "father_relationship": 1.5,  # 与父亲关系增加
        "mother_relationship": 1.5,  # 与母亲关系增加
        "grandfather_relationship": 1.0,  # 与祖父关系增加
        "grandmother_relationship": 1.0,  # 与祖母关系增加
        "physical_health": 0.0
    },
    "实践活动": {
        "knowledge": 2.0,       # 知识储备增加2.0点
        "stress": -3.0,         # 压力减少3.0点
        "father_relationship": 2.5,  # 与父亲关系增加
        "mother_relationship": 2.5,  # 与母亲关系增加
        "grandfather_relationship": 2.0,  # 与祖父关系增加
        "grandmother_relationship": 2.0,  # 与祖母关系增加
        "physical_health": 1.5  # 身体素质提升
    },
    "启蒙教育": {
        "knowledge": 0.5,       # 知识储备略有增加
        "stress": -1.0,         # 压力减少1.0点
        "father_relationship": 1.5,  # 与父亲关系增加
        "mother_relationship": 1.5,  # 与母亲关系增加
        "grandfather_relationship": 1.2,  # 与祖父关系增加
        "grandmother_relationship": 1.2,  # 与祖母关系增加
        "physical_health": 0.5  # 身体素质略有提升
    },
    "游戏互动": {
        "knowledge": 0.0,
        "stress": -3.0,         # 压力减少3.0点
        "father_relationship": 2.5,  # 与父亲关系增加
        "mother_relationship": 2.5,  # 与母亲关系增加
        "grandfather_relationship": 2.0,  # 与祖父关系增加
        "grandmother_relationship": 2.0,  # 与祖母关系增加
        "physical_health": 1.0  # 身体素质提升
    },
    "简单辅导": {
        "knowledge": 1.0,       # 知识储备增加1.0点
        "stress": 1.0,          # 压力略有增加
        "father_relationship": 1.0,  # 与父亲关系增加
        "mother_relationship": 1.0,  # 与母亲关系增加
        "grandfather_relationship": 0.8,  # 与祖父关系增加
        "grandmother_relationship": 0.8,  # 与祖母关系增加
        "physical_health": 0.0  # 身体素质无变化
    },
    "简单兴趣培养": {
        "knowledge": 1.0,       # 知识储备增加1.0点
        "stress": 2.0,          # 压力略有增加
        "father_relationship": 1.5,  # 与父亲关系增加
        "mother_relationship": 1.5,  # 与母亲关系增加
        "grandfather_relationship": 1.2,  # 与祖父关系增加
        "grandmother_relationship": 1.2,  # 与祖母关系增加
        "physical_health": 0.5  # 身体素质略有提升
    }
}


class DungeonMaster:
    """
    裁判系统核心类，负责判定父亲行为对孩子的影响
    
    增强版：集成LLM智能评分系统
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None, enable_llm_evaluation: bool = True):
        """
        初始化裁判系统

        参数:
            llm_client: LLM客户端实例，用于处理非线性数值判定
            enable_llm_evaluation: 是否启用LLM智能评分（会增加API调用）
        """
        self.llm_client = llm_client
        self.enable_llm_evaluation = enable_llm_evaluation
        
        # 初始化LLM评估器
        self.evaluator: Optional[LLMEvaluator] = None
        if llm_client and enable_llm_evaluation:
            try:
                self.evaluator = create_llm_evaluator(llm_client, enable_detailed=True)
                logger.info("LLM智能评分系统已启用")
            except Exception as e:
                logger.warning(f"LLM评估器初始化失败: {e}，将使用规则评估")
        
        # 行为历史记录（用于上下文感知评估）
        self.action_history: deque = deque(maxlen=52)  # 最近52周（1年）的行为
        
        # 评估统计
        self.evaluation_stats = {
            "total_evaluations": 0,
            "llm_evaluations": 0,
            "fallback_evaluations": 0,
            "average_multiplier": 1.0
        }
        
        logger.info("DungeonMaster initialized")
    
    async def evaluate_outcome(self, 
                              child_state: ChildState, 
                              family_state: FamilyState, 
                              action: Dict[str, Any], 
                              current_event: str, 
                              member: str = "father") -> Dict[str, Any]:
        """
        评估家庭成员行为的结果，更新游戏状态
        
        参数:
            child_state: 孩子当前状态对象
            family_state: 家庭当前状态对象
            action: 家庭成员执行的行为，包含action_type、dialogue和cost
            current_event: 当前触发的事件描述
            member: 执行行为的家庭成员，默认是父亲
            
        返回:
            评估结果字典，包含执行状态、消耗资金和状态变化
        """
        # 成员名称中文映射
        member_names = {
            "father": "父亲",
            "mother": "母亲", 
            "grandfather": "祖父",
            "grandmother": "祖母",
            "family": "家庭",
            "family_consensus": "家庭共识"
        }
        member_cn = member_names.get(member, member)
        logger.info(f"开始评估行为: {action}, 执行成员: {member_cn}, 当前事件: {current_event}")
        
        result = {
            "success": True,
            "message": "行为执行成功",
            "cost": action["cost"],
            "member": member_cn,
            "member_id": member,
            "state_changes": {}
        }
        
        # 1. 年龄校验机制：检查行为是否适合孩子当前年龄
        age = child_state.calculate_age(family_state.current_date)
        age_group = child_state.get_age_group(family_state.current_date)
        
        # 年龄限制规则：使用允许列表模式（更明确）
        age_appropriate_actions = {
            "infant": [  # 0-3岁：婴幼儿专属行为
                "亲子互动", "日常照料", "感官刺激", "户外活动", "早期阅读",
                "安抚陪伴", "社交接触", "陪伴", "启蒙教育", "健康教育", "游戏互动", "鼓励"
            ],
            "preschool": [  # 3-6岁：学前阶段，可以开始简单学习
                "陪伴", "启蒙教育", "游戏互动", "简单辅导", "鼓励", "简单兴趣培养",
                "健康教育", "创新活动", "沟通", "户外活动", "早期阅读", "亲子互动",
                "感官刺激", "社交接触", "安抚陪伴"
            ],
            "primary": [  # 6岁以上：全部行为都可用
                "辅导", "鼓励", "花钱培训", "陪伴", "严格要求", "监督学习",
                "健康教育", "创新活动", "个性化计划", "实践活动", "沟通",
                "启蒙教育", "游戏互动", "简单辅导", "简单兴趣培养",
                "亲子互动", "户外活动", "早期阅读", "社交接触"
            ]
        }
        
        # 兼容性：保留禁止列表用于日志
        age_restrictions = {
            "infant": {"禁止": ["辅导", "花钱培训", "严格要求", "监督学习", "个性化计划", "实践活动", "简单辅导", "简单兴趣培养"]},
            "preschool": {"禁止": ["辅导", "花钱培训", "严格要求", "监督学习", "个性化计划"]},
            "primary": {"禁止": []}
        }
        
        action_type = action["action_type"]
        
        # 检查行为是否在允许列表中（优先使用允许列表）
        allowed_actions = age_appropriate_actions.get(age_group, age_appropriate_actions["primary"])
        
        # 如果行为不在允许列表中，检查是否在 ACTION_EFFECTS 中有定义
        if action_type not in allowed_actions:
            # 如果行为在禁止列表中，明确拒绝
            if action_type in age_restrictions.get(age_group, {}).get("禁止", []):
                result["success"] = False
                result["message"] = f"该行为「{action_type}」不适合{age:.1f}岁的孩子，建议选择：{', '.join(allowed_actions[:5])}等"
                result["cost"] = 0
                
                # 行动失败时，增加孩子压力值
                child_state.stress = min(100.0, child_state.stress + 5.0)
                result["state_changes"]["stress"] = 5.0
                
                logger.warning(f"行为执行失败: {result['message']}")
                return result
            # 如果行为不在允许列表但也不在禁止列表，尝试宽容处理（使用默认效果）
            elif action_type not in ACTION_EFFECTS:
                logger.info(f"行为类型「{action_type}」不在预定义列表中，将使用默认效果")
        
        # 2. 资金检查机制
        if family_state.family_savings < action["cost"]:
            # 资金不足，行动失败
            result["success"] = False
            result["message"] = "资金不足，无法执行该行为"
            result["cost"] = 0
            
            # 行动失败时，增加孩子压力值
            child_state.stress = min(100.0, child_state.stress + 5.0)
            result["state_changes"]["stress"] = 5.0
            
            logger.warning(f"行为执行失败: {result['message']}")
            return result
        
        # 2. 扣除行为成本
        family_state.family_savings -= action["cost"]
        family_state.family_savings = max(0.0, family_state.family_savings)  # 确保存款不为负
        
        # 3. 获取行为影响规则
        action_type = action["action_type"]
        effects = ACTION_EFFECTS.get(action_type, {})
        
        if not effects:
            # 未知行为类型，使用默认效果（轻微的正面影响）
            logger.warning(f"未知行为类型: {action_type}，使用默认效果")
            effects = {
                "knowledge": 0.5,
                "stress": 1.0,
                "father_relationship": 0.5,
                "mother_relationship": 0.5,
                "grandfather_relationship": 0.3,
                "grandmother_relationship": 0.3,
                "physical_health": 0.0
            }
            result["message"] = f"行为类型 {action_type} 不在预定义列表中，使用默认效果"
        
        # 4. 获取执行成员的影响力权重
        member_obj = getattr(family_state, member, None)
        if not member_obj:
            logger.warning(f"未知家庭成员: {member}")
            result["message"] = f"未知家庭成员: {member}"
            return result
        
        influence_weight = member_obj.influence_weight
        logger.info(f"{member_cn}的影响力权重: {influence_weight}")
        
        # 4.5 LLM智能评估：获取效果修正系数
        eval_result: Optional[ActionEvaluationResult] = None
        effect_multiplier = 1.0
        bonus_effects = {}
        
        if self.evaluator:
            try:
                # 构建评估上下文
                context = {
                    "recent_actions": list(self.action_history)[-10:],
                    "current_event": current_event,
                    "member": member
                }
                
                # 调用LLM评估
                eval_result = await self.evaluator.evaluate_action(
                    action, child_state, family_state, context
                )
                
                effect_multiplier = eval_result.effect_multiplier
                bonus_effects = eval_result.bonus_effects
                
                # 更新统计
                self.evaluation_stats["total_evaluations"] += 1
                if eval_result.reasoning != "标准效果":
                    self.evaluation_stats["llm_evaluations"] += 1
                else:
                    self.evaluation_stats["fallback_evaluations"] += 1
                
                # 更新平均修正系数
                total = self.evaluation_stats["total_evaluations"]
                avg = self.evaluation_stats["average_multiplier"]
                self.evaluation_stats["average_multiplier"] = (avg * (total - 1) + effect_multiplier) / total
                
                logger.info(f"LLM评估结果: 效果系数={effect_multiplier:.2f}, 理由={eval_result.reasoning}")
                
                # 记录评估建议（如有）
                if eval_result.suggestion:
                    result["suggestion"] = eval_result.suggestion
                    
            except Exception as e:
                logger.warning(f"LLM评估失败: {e}，使用默认效果系数")
                effect_multiplier = 1.0
        
        # 记录行为到历史
        self.action_history.append({
            "action_type": action_type,
            "week": len(self.action_history),
            "member": member
        })
        
        # 5. 更新孩子状态（使用统一状态管理接口）
        # 注意：所有效果都会乘以 effect_multiplier（来自LLM评估）
        state_changes = {}
        child_state_changes = {}
        
        # 5.1 更新知识储备
        if "knowledge" in effects:
            knowledge_effect = effects["knowledge"] * influence_weight * effect_multiplier
            # 添加LLM评估的额外加成
            if "knowledge" in bonus_effects:
                knowledge_effect += bonus_effects["knowledge"]
            child_state_changes["knowledge"] = knowledge_effect
        
        # 5.2 更新压力值
        if "stress" in effects:
            stress_effect = effects["stress"] * influence_weight * effect_multiplier
            # 压力的额外影响（LLM评估可能增加惩罚）
            if "stress" in bonus_effects:
                stress_effect += bonus_effects["stress"]
            child_state_changes["stress"] = stress_effect
        
        # 5.3 更新家庭成员关系
        # 与执行行为的家庭成员关系变化
        member_relationship_key = f"{member}_relationship"
        if member_relationship_key in effects:
            relationship_effect = effects[member_relationship_key] * influence_weight * effect_multiplier
            # 关系的额外影响
            if "relationship" in bonus_effects:
                relationship_effect += bonus_effects["relationship"]
            child_state_changes[member_relationship_key] = relationship_effect
        
        # 5.4 其他家庭成员关系的间接影响（成员间交互关系）
        # 例如：父亲的行为也会间接影响与母亲、祖父母的关系
        indirect_relationship_effects = {
            "father": {"mother_relationship": 0.3, "grandfather_relationship": 0.2, "grandmother_relationship": 0.2},
            "mother": {"father_relationship": 0.3, "grandfather_relationship": 0.2, "grandmother_relationship": 0.2},
            "grandfather": {"grandmother_relationship": 0.5, "father_relationship": 0.2, "mother_relationship": 0.2},
            "grandmother": {"grandfather_relationship": 0.5, "father_relationship": 0.2, "mother_relationship": 0.2}
        }
        
        if member in indirect_relationship_effects:
            for other_member, indirect_weight in indirect_relationship_effects[member].items():
                other_relationship_key = f"{other_member}_relationship"
                if other_relationship_key in effects:
                    # 间接影响是直接影响的一部分（也受效果系数影响）
                    indirect_effect = effects[other_relationship_key] * influence_weight * indirect_weight * effect_multiplier
                    # 累加到已有的变化中（如果已有）
                    if other_relationship_key in child_state_changes:
                        child_state_changes[other_relationship_key] += indirect_effect
                    else:
                        child_state_changes[other_relationship_key] = indirect_effect
        
        # 5.5 更新身体素质
        if "physical_health" in effects:
            health_effect = effects["physical_health"] * influence_weight * effect_multiplier
            child_state_changes["physical_health"] = health_effect
        
        # 5.6 应用LLM评估的其他额外效果
        for bonus_key, bonus_value in bonus_effects.items():
            if bonus_key not in ["knowledge", "stress", "relationship"]:
                # 尝试直接应用到孩子状态（如果属性存在）
                if hasattr(child_state, bonus_key):
                    if bonus_key not in child_state_changes:
                        child_state_changes[bonus_key] = bonus_value
                    else:
                        child_state_changes[bonus_key] += bonus_value
        
        # 记录效果修正信息
        if effect_multiplier != 1.0:
            result["effect_multiplier"] = effect_multiplier
            result["evaluation_reasoning"] = eval_result.reasoning if eval_result else "规则评估"
        
        # 使用统一状态管理器更新所有状态
        state_changes = StateManager.update_child_state(
            child_state,
            child_state_changes,
            apply_limits=True
        )
        
        # 6. 非线性数值判定：调用LLM接口
        if self.llm_client and action.get("dialogue"):
            try:
                # 使用安全的LLM调用，包含重试和降级机制
                llm_effect = await safe_llm_call(
                    self.llm_client,
                    "analyze_emotion",
                    action["dialogue"],
                    fallback_value=None
                )

                # 应用LLM返回的影响（例如：根据对话情感调整亲子关系）
                if llm_effect:
                    # 影响执行行为的家庭成员关系
                    if member_relationship_key in llm_effect:
                        relationship_effect = llm_effect[member_relationship_key] * influence_weight
                        # 使用统一状态管理器更新
                        llm_changes = StateManager.update_child_state(
                            child_state,
                            {member_relationship_key: relationship_effect},
                            apply_limits=True
                        )
                        state_changes.update(llm_changes)
                        logger.info(f"LLM情感分析结果影响{member_cn}关系: {relationship_effect}")

            except Exception as e:
                logger.warning(f"LLM情感分析失败，使用默认行为: {e}")
        
        result["state_changes"] = state_changes
        logger.info(f"行为执行成功，状态变化: {state_changes}")
        
        # 7. 检查高压力崩溃机制（压力>90触发Mental Breakdown）
        if child_state.stress > 90:
            breakdown_result = self._handle_mental_breakdown(child_state, family_state)
            result["mental_breakdown"] = breakdown_result
            logger.warning(f"触发心理崩溃事件！压力值: {child_state.stress:.1f}")
        
        return result
    
    def _handle_mental_breakdown(self, child_state: ChildState, family_state: FamilyState) -> Dict[str, Any]:
        """
        处理心理崩溃事件
        
        当压力>90时触发：
        - 跳过本周决策（不执行任何行动）
        - 降低知识储备、关系值、健康值
        - 压力值降低（因为崩溃后需要恢复）
        
        参数:
            child_state: 孩子当前状态
            family_state: 家庭当前状态
            
        返回:
            崩溃事件处理结果
        """
        breakdown_effects = {
            "knowledge": -5.0,  # 知识储备下降
            "stress": -15.0,    # 压力降低（崩溃后需要恢复期）
            "father_relationship": -3.0,
            "mother_relationship": -3.0,
            "grandfather_relationship": -2.0,
            "grandmother_relationship": -2.0,
            "physical_health": -5.0  # 健康下降
        }
        
        # 使用统一状态管理器更新崩溃影响
        state_changes = StateManager.update_child_state(
            child_state,
            breakdown_effects,
            apply_limits=True
        )
        
        # 计算崩溃前的压力值（应用变化前）
        stress_before = child_state.stress - state_changes.get("stress", 0.0)
        
        return {
            "triggered": True,
            "message": f"孩子因压力过大（{stress_before:.1f}）发生心理崩溃，本周无法正常学习和活动",
            "state_changes": state_changes,
            "skip_decision": True  # 标记跳过本周决策
        }
    
    
    def get_action_effects(self, action_type: str) -> Optional[Dict[str, float]]:
        """
        获取指定行为类型的影响规则
        
        参数:
            action_type: 行为类型
            
        返回:
            行为影响规则字典，若行为类型不存在则返回None
        """
        return ACTION_EFFECTS.get(action_type)
    
    def add_custom_effect(self, action_type: str, effects: Dict[str, float]) -> None:
        """
        添加自定义行为影响规则
        
        参数:
            action_type: 行为类型
            effects: 行为影响规则字典
        """
        ACTION_EFFECTS[action_type] = effects
        logger.info(f"添加自定义行为影响规则: {action_type} -> {effects}")
    
    async def evaluate_development_stage(self, 
                                          child_state: ChildState,
                                          family_state: FamilyState) -> Dict[str, Any]:
        """
        阶段性发展评估（建议每月或每年调用一次）
        
        参数:
            child_state: 孩子状态
            family_state: 家庭状态
            
        返回:
            发展评估结果字典
        """
        if not self.evaluator:
            # 无评估器时返回基本评估
            return self._basic_development_evaluation(child_state, family_state)
        
        try:
            from src.core.llm_evaluator import DevelopmentEvaluationResult
            
            result = await self.evaluator.evaluate_development_stage(
                child_state, 
                family_state,
                history=list(self.action_history)
            )
            
            return {
                "overall_score": result.overall_score,
                "dimension_scores": result.dimension_scores,
                "assessment": result.assessment,
                "strengths": result.strengths,
                "weaknesses": result.weaknesses,
                "recommendations": result.recommendations,
                "evaluated_by": "llm"
            }
            
        except Exception as e:
            logger.warning(f"阶段性评估失败: {e}")
            return self._basic_development_evaluation(child_state, family_state)
    
    def _basic_development_evaluation(self, 
                                        child_state: ChildState,
                                        family_state: FamilyState) -> Dict[str, Any]:
        """基础发展评估（无LLM时使用）"""
        age = child_state.calculate_age(family_state.current_date)
        
        # 计算各维度分数
        scores = {
            "学业": child_state.knowledge,
            "情商": (100 - child_state.stress) * 0.5 + getattr(child_state, 'security_feeling', 70) * 0.5,
            "社交": getattr(child_state, 'social_skill', 50),
            "健康": child_state.physical_health,
            "家庭关系": (child_state.father_relationship + child_state.mother_relationship) / 2,
            "创造力": getattr(child_state, 'creativity', 50)
        }
        
        overall = sum(scores.values()) / len(scores)
        
        # 识别优势和不足
        strengths = [k for k, v in scores.items() if v > 70]
        weaknesses = [k for k, v in scores.items() if v < 50]
        
        # 生成评语
        if overall > 70:
            assessment = f"孩子在{age:.1f}岁时发展良好，各方面表现积极。"
        elif overall > 50:
            assessment = f"孩子在{age:.1f}岁时发展一般，部分领域需要加强。"
        else:
            assessment = f"孩子在{age:.1f}岁时发展需要关注，建议调整教育方式。"
        
        # 生成建议
        recommendations = []
        if "学业" in weaknesses:
            recommendations.append("增加启蒙教育和阅读活动")
        if "情商" in weaknesses:
            recommendations.append("需要更多陪伴和情感交流")
        if "社交" in weaknesses:
            recommendations.append("鼓励参与社交活动")
        if "健康" in weaknesses:
            recommendations.append("增加户外活动和体育锻炼")
        if "家庭关系" in weaknesses:
            recommendations.append("改善沟通方式，增加亲子互动")
        
        return {
            "overall_score": overall,
            "dimension_scores": scores,
            "assessment": assessment,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations[:3] if recommendations else ["继续保持良好的教育方式"],
            "evaluated_by": "rules"
        }
    
    async def evaluate_final_achievement(self,
                                          child_state: ChildState,
                                          family_state: FamilyState) -> Dict[str, Any]:
        """
        最终成就评估（模拟结束时调用）
        
        参数:
            child_state: 孩子最终状态
            family_state: 家庭最终状态
            
        返回:
            最终成就评估结果
        """
        if self.evaluator:
            try:
                result = await self.evaluator.evaluate_final_achievement(
                    child_state,
                    family_state,
                    full_history=list(self.action_history)
                )
                
                return {
                    "achievement_level": result.achievement_level,
                    "scores": {
                        "academic": result.academic_score,
                        "emotional": result.emotional_score,
                        "social": result.social_score,
                        "health": result.health_score,
                        "relationship": result.relationship_score,
                        "creativity": result.creativity_score,
                        "resilience": result.resilience_score
                    },
                    "career_prediction": result.career_prediction,
                    "final_assessment": result.final_assessment,
                    "evaluation_stats": self.evaluation_stats
                }
                
            except Exception as e:
                logger.warning(f"最终评估失败: {e}")
        
        # 降级评估
        return self._basic_final_evaluation(child_state, family_state)
    
    def _basic_final_evaluation(self,
                                 child_state: ChildState,
                                 family_state: FamilyState) -> Dict[str, Any]:
        """基础最终评估"""
        # 计算综合分数
        academic = child_state.knowledge
        emotional = (100 - child_state.stress) * 0.5 + getattr(child_state, 'self_confidence', 60) * 0.5
        social = getattr(child_state, 'social_skill', 50)
        health = child_state.physical_health
        relationship = (
            child_state.father_relationship * 0.3 +
            child_state.mother_relationship * 0.3 +
            child_state.grandfather_relationship * 0.2 +
            child_state.grandmother_relationship * 0.2
        )
        creativity = getattr(child_state, 'creativity', 50)
        resilience = getattr(child_state, 'resilience', 50)
        
        total = (academic * 0.25 + emotional * 0.20 + social * 0.15 + 
                 health * 0.10 + relationship * 0.15 + creativity * 0.10 + resilience * 0.05)
        
        # 确定等级
        if total >= 90:
            level, career = "S", "成为领域专家或企业高管的潜力很大"
        elif total >= 80:
            level, career = "A", "有望成为优秀的专业人才"
        elif total >= 70:
            level, career = "B", "能够找到稳定的工作，生活幸福"
        elif total >= 60:
            level, career = "C", "平稳发展，需要更多努力"
        elif total >= 50:
            level, career = "D", "面临一些挑战"
        else:
            level, career = "F", "发展遇到困难"
        
        return {
            "achievement_level": level,
            "scores": {
                "academic": academic,
                "emotional": emotional,
                "social": social,
                "health": health,
                "relationship": relationship,
                "creativity": creativity,
                "resilience": resilience
            },
            "total_score": total,
            "career_prediction": career,
            "final_assessment": f"综合评级{level}，总分{total:.1f}分",
            "evaluation_stats": self.evaluation_stats
        }
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """获取评估统计信息"""
        return {
            **self.evaluation_stats,
            "action_history_length": len(self.action_history),
            "evaluator_enabled": self.evaluator is not None
        }


# 改为工厂模式：每个环境创建独立实例
def get_dungeon_master(llm_client: Optional[Any] = None, 
                        enable_llm_evaluation: bool = True) -> DungeonMaster:
    """
    创建DungeonMaster实例（工厂模式，每个环境独立实例）
    
    参数:
        llm_client: LLM客户端实例，用于处理非线性数值判定
        enable_llm_evaluation: 是否启用LLM智能评分
        
    返回:
        DungeonMaster实例
    """
    # 每次调用都创建新实例，确保多环境互不干扰
    return DungeonMaster(llm_client, enable_llm_evaluation)
    