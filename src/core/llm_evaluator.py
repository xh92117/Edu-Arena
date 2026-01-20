"""
LLM智能评分系统

使用大模型对教育行为和孩子发展进行评估：
1. 行为效果评估 - 根据上下文评估单次行为的效果
2. 阶段性成就评估 - 每月/每年评估孩子的发展
3. 对话质量评估 - 评估父母的沟通方式
4. 最终成就评估 - 模拟结束时的综合评价
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger(__name__)


@dataclass
class ActionEvaluationResult:
    """行为评估结果"""
    # 效果修正系数（0.5-2.0，1.0为标准效果）
    effect_multiplier: float
    # 各维度的额外效果
    bonus_effects: Dict[str, float]
    # 评估理由
    reasoning: str
    # 是否触发特殊事件
    special_event: Optional[str] = None
    # 给父母的建议
    suggestion: Optional[str] = None


@dataclass
class DevelopmentEvaluationResult:
    """发展阶段评估结果"""
    # 总分（0-100）
    overall_score: float
    # 各维度评分
    dimension_scores: Dict[str, float]
    # 发展评语
    assessment: str
    # 优势和不足
    strengths: List[str]
    weaknesses: List[str]
    # 发展建议
    recommendations: List[str]


@dataclass
class FinalAchievementResult:
    """最终成就评估结果"""
    # 综合成就等级
    achievement_level: str  # S/A/B/C/D/F
    # 各领域成就
    academic_score: float      # 学业成就
    emotional_score: float     # 情商发展
    social_score: float        # 社交能力
    health_score: float        # 身心健康
    relationship_score: float  # 家庭关系
    creativity_score: float    # 创造力
    resilience_score: float    # 抗挫折能力
    # 人生预测
    career_prediction: str
    # 综合评语
    final_assessment: str


class LLMEvaluator:
    """
    LLM智能评分器
    
    使用大模型进行多维度评估
    """
    
    def __init__(self, llm_client, enable_detailed_eval: bool = True):
        """
        初始化评分器
        
        参数:
            llm_client: LLM客户端
            enable_detailed_eval: 是否启用详细评估（会增加API调用）
        """
        self.llm_client = llm_client
        self.enable_detailed_eval = enable_detailed_eval
        self.evaluation_history: List[Dict] = []
        
        # 获取模型名称（用于API调用）
        self.model_name = self._get_model_name()
    
    def _get_model_name(self) -> str:
        """获取LLM模型名称"""
        if not self.llm_client:
            return "gpt-3.5-turbo"
        
        # 尝试从客户端获取模型名称
        if hasattr(self.llm_client, 'model_name'):
            return self.llm_client.model_name
        if hasattr(self.llm_client, 'config'):
            # 尝试从配置中获取
            config = self.llm_client.config
            if hasattr(config, 'llm_model'):
                return config.llm_model
        
        return "gpt-3.5-turbo"  # 默认模型
    
    async def evaluate_action(self,
                               action: Dict[str, Any],
                               child_state: Any,
                               family_state: Any,
                               context: Dict[str, Any] = None) -> ActionEvaluationResult:
        """
        评估单次行为的效果
        
        参数:
            action: 行为字典（action_type, dialogue, cost）
            child_state: 孩子当前状态
            family_state: 家庭当前状态
            context: 上下文信息（历史行为、当前事件等）
            
        返回:
            行为评估结果
        """
        if not self.llm_client or not self.enable_detailed_eval:
            return self._fallback_action_evaluation(action, child_state, family_state)
        
        # 构建评估提示
        prompt = self._build_action_evaluation_prompt(action, child_state, family_state, context)
        
        try:
            messages = [
                {"role": "system", "content": self._get_action_evaluator_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            response = await self.llm_client.chat_completion(
                messages, 
                model=self.model_name,
                temperature=0.3,
                max_tokens=500
            )
            
            content = response["choices"][0]["message"]["content"]
            return self._parse_action_evaluation(content, action, child_state)
            
        except Exception as e:
            logger.warning(f"LLM行为评估失败: {e}，使用降级方案")
            return self._fallback_action_evaluation(action, child_state, family_state)
    
    def _get_action_evaluator_system_prompt(self) -> str:
        """获取行为评估的系统提示"""
        return """你是一位资深的儿童教育专家和心理学家，负责评估家长教育行为的效果。

你的评估需要考虑：
1. 孩子的当前状态（压力、情绪、发展阶段）
2. 行为与孩子年龄的适配性
3. 对话的情感质量和沟通方式
4. 行为的长期影响
5. 家庭的经济状况和行为成本

请返回JSON格式的评估结果：
{
    "effect_multiplier": 0.5-2.0之间的数值，1.0为标准效果，
    "bonus_effects": {
        "knowledge": 额外知识加成,
        "stress": 额外压力影响（正为增加，负为减少）,
        "relationship": 额外关系影响,
        "creativity": 额外创造力影响,
        "confidence": 额外自信影响
    },
    "reasoning": "评估理由（50字内）",
    "suggestion": "给家长的建议（如有）"
}

只返回JSON，不要有其他内容。"""
    
    def _build_action_evaluation_prompt(self,
                                         action: Dict[str, Any],
                                         child_state: Any,
                                         family_state: Any,
                                         context: Dict[str, Any] = None) -> str:
        """构建行为评估提示"""
        age = child_state.calculate_age(family_state.current_date)
        
        # 获取敏感期信息
        active_sensitivities = {}
        if hasattr(child_state, 'development_sensitivity'):
            active_sensitivities = child_state.development_sensitivity.get_active_sensitivities()
        
        prompt = f"""## 当前情况

**孩子状态**：
- 年龄：{age:.1f}岁
- 压力值：{child_state.stress:.1f}/100
- 知识储备：{child_state.knowledge:.1f}/100
- 情绪状态：{getattr(child_state, 'emotional_state', '未知')}
- 当前敏感期：{', '.join(active_sensitivities.keys()) if active_sensitivities else '无明显敏感期'}

**家庭经济**：
- 家庭存款：{family_state.family_savings:.0f}元

**本次行为**：
- 行为类型：{action.get('action_type', '未知')}
- 花费金额：{action.get('cost', 0)}元
- 对话内容："{action.get('dialogue', '无')}"

"""
        
        if context:
            if context.get('recent_actions'):
                prompt += f"**最近行为**：{', '.join(context['recent_actions'][-5:])}\n"
            if context.get('current_event'):
                prompt += f"**当前事件**：{context['current_event']}\n"
        
        prompt += "\n请评估这次教育行为的效果。"
        
        return prompt
    
    def _parse_action_evaluation(self,
                                  content: str,
                                  action: Dict[str, Any],
                                  child_state: Any) -> ActionEvaluationResult:
        """解析行为评估结果"""
        import re
        
        try:
            # 尝试提取JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                return ActionEvaluationResult(
                    effect_multiplier=max(0.5, min(2.0, data.get("effect_multiplier", 1.0))),
                    bonus_effects=data.get("bonus_effects", {}),
                    reasoning=data.get("reasoning", "LLM评估"),
                    suggestion=data.get("suggestion")
                )
        except Exception as e:
            logger.warning(f"解析LLM评估结果失败: {e}")
        
        # 解析失败，返回默认结果
        return ActionEvaluationResult(
            effect_multiplier=1.0,
            bonus_effects={},
            reasoning="无法解析LLM响应，使用默认效果"
        )
    
    def _fallback_action_evaluation(self,
                                     action: Dict[str, Any],
                                     child_state: Any,
                                     family_state: Any) -> ActionEvaluationResult:
        """降级行为评估（基于规则）"""
        multiplier = 1.0
        bonus_effects = {}
        reasoning_parts = []
        
        action_type = action.get("action_type", "")
        dialogue = action.get("dialogue", "")
        
        # 规则1：高压力时减压行为效果加成
        if child_state.stress > 70:
            if action_type in ["陪伴", "游戏互动", "鼓励", "安抚陪伴"]:
                multiplier *= 1.3
                reasoning_parts.append("高压力时减压行为更有效")
            elif action_type in ["严格要求", "监督学习"]:
                multiplier *= 0.6
                bonus_effects["stress"] = 5  # 额外增加压力
                reasoning_parts.append("高压力时严格要求可能适得其反")
        
        # 规则2：敏感期加成
        if hasattr(child_state, 'development_sensitivity'):
            sensitivities = child_state.development_sensitivity
            if sensitivities.language > 0.7 and action_type in ["早期阅读", "启蒙教育"]:
                multiplier *= 1.4
                bonus_effects["knowledge"] = 1.0
                reasoning_parts.append("语言敏感期，语言类活动效果显著")
            if sensitivities.social > 0.7 and action_type in ["社交接触", "游戏互动"]:
                multiplier *= 1.3
                bonus_effects["social_skill"] = 1.0
                reasoning_parts.append("社交敏感期，社交活动效果好")
        
        # 规则3：对话情感一致性
        positive_words = ["爱", "棒", "好", "乖", "加油", "相信"]
        negative_words = ["不行", "笨", "差", "失望", "没用"]
        
        positive_count = sum(1 for word in positive_words if word in dialogue)
        negative_count = sum(1 for word in negative_words if word in dialogue)
        
        if action_type in ["鼓励", "陪伴"] and negative_count > positive_count:
            multiplier *= 0.7
            reasoning_parts.append("行为与对话不一致")
        elif action_type in ["严格要求"] and positive_count > negative_count:
            multiplier *= 1.2
            reasoning_parts.append("严格但有爱，效果更好")
        
        return ActionEvaluationResult(
            effect_multiplier=multiplier,
            bonus_effects=bonus_effects,
            reasoning="；".join(reasoning_parts) if reasoning_parts else "标准效果"
        )
    
    async def evaluate_development_stage(self,
                                          child_state: Any,
                                          family_state: Any,
                                          history: List[Dict] = None) -> DevelopmentEvaluationResult:
        """
        阶段性发展评估（每月/每年调用）
        
        参数:
            child_state: 孩子状态
            family_state: 家庭状态
            history: 历史行为记录
            
        返回:
            发展评估结果
        """
        if not self.llm_client:
            return self._fallback_development_evaluation(child_state, family_state)
        
        prompt = self._build_development_evaluation_prompt(child_state, family_state, history)
        
        try:
            messages = [
                {"role": "system", "content": self._get_development_evaluator_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            response = await self.llm_client.chat_completion(
                messages,
                model=self.model_name,
                temperature=0.5,
                max_tokens=800
            )
            
            content = response["choices"][0]["message"]["content"]
            return self._parse_development_evaluation(content)
            
        except Exception as e:
            logger.warning(f"LLM发展评估失败: {e}")
            return self._fallback_development_evaluation(child_state, family_state)
    
    def _get_development_evaluator_system_prompt(self) -> str:
        """发展评估系统提示"""
        return """你是一位权威的儿童发展评估专家。请根据孩子的各项指标和成长历史，给出专业的发展评估。

请返回JSON格式：
{
    "overall_score": 0-100的综合分数,
    "dimension_scores": {
        "学业": 分数,
        "情商": 分数,
        "社交": 分数,
        "健康": 分数,
        "家庭关系": 分数,
        "创造力": 分数
    },
    "assessment": "综合评语（100字内）",
    "strengths": ["优势1", "优势2"],
    "weaknesses": ["不足1", "不足2"],
    "recommendations": ["建议1", "建议2"]
}"""
    
    def _build_development_evaluation_prompt(self,
                                               child_state: Any,
                                               family_state: Any,
                                               history: List[Dict] = None) -> str:
        """构建发展评估提示"""
        age = child_state.calculate_age(family_state.current_date)
        
        prompt = f"""## 孩子发展状况

**基本信息**：
- 年龄：{age:.1f}岁
- 智商：{child_state.iq}

**核心指标**：
- 知识储备：{child_state.knowledge:.1f}/100
- 压力水平：{child_state.stress:.1f}/100
- 身体健康：{child_state.physical_health:.1f}/100
- 安全感：{getattr(child_state, 'security_feeling', 70):.1f}/100
- 自信心：{getattr(child_state, 'self_confidence', 60):.1f}/100
- 社交能力：{getattr(child_state, 'social_skill', 50):.1f}/100
- 创造力：{getattr(child_state, 'creativity', 50):.1f}/100
- 抗挫折能力：{getattr(child_state, 'resilience', 50):.1f}/100

**家庭关系**：
- 与父亲：{child_state.father_relationship:.1f}/100
- 与母亲：{child_state.mother_relationship:.1f}/100
- 与祖父母：{(child_state.grandfather_relationship + child_state.grandmother_relationship)/2:.1f}/100

**情绪状态**：{getattr(child_state, 'emotional_state', '平静')}

请对这个孩子的发展状况进行评估。"""
        
        return prompt
    
    def _parse_development_evaluation(self, content: str) -> DevelopmentEvaluationResult:
        """解析发展评估结果"""
        import re
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                return DevelopmentEvaluationResult(
                    overall_score=data.get("overall_score", 60),
                    dimension_scores=data.get("dimension_scores", {}),
                    assessment=data.get("assessment", ""),
                    strengths=data.get("strengths", []),
                    weaknesses=data.get("weaknesses", []),
                    recommendations=data.get("recommendations", [])
                )
        except Exception as e:
            logger.warning(f"解析发展评估失败: {e}")
        
        return self._fallback_development_evaluation(None, None)
    
    def _fallback_development_evaluation(self,
                                          child_state: Any,
                                          family_state: Any) -> DevelopmentEvaluationResult:
        """降级发展评估"""
        if child_state is None:
            return DevelopmentEvaluationResult(
                overall_score=60,
                dimension_scores={},
                assessment="无法获取评估数据",
                strengths=[],
                weaknesses=[],
                recommendations=[]
            )
        
        # 基于规则计算分数
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
        
        return DevelopmentEvaluationResult(
            overall_score=overall,
            dimension_scores=scores,
            assessment=f"孩子整体发展{'良好' if overall > 70 else '一般' if overall > 50 else '需要关注'}",
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=self._generate_recommendations(scores)
        )
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if scores.get("学业", 50) < 50:
            recommendations.append("建议增加启蒙教育和阅读活动")
        if scores.get("情商", 50) < 50:
            recommendations.append("需要更多陪伴和情感交流")
        if scores.get("社交", 50) < 50:
            recommendations.append("鼓励参与社交活动，多接触同龄人")
        if scores.get("健康", 50) < 70:
            recommendations.append("增加户外活动，注意身体锻炼")
        if scores.get("家庭关系", 50) < 60:
            recommendations.append("改善沟通方式，增加亲子互动")
        
        return recommendations[:3] if recommendations else ["继续保持良好的教育方式"]
    
    async def evaluate_final_achievement(self,
                                          child_state: Any,
                                          family_state: Any,
                                          full_history: List[Dict] = None) -> FinalAchievementResult:
        """
        最终成就评估（模拟结束时调用）
        
        参数:
            child_state: 孩子最终状态
            family_state: 家庭最终状态
            full_history: 完整历史记录
            
        返回:
            最终成就评估结果
        """
        # 计算各项分数
        age = child_state.calculate_age(family_state.current_date)
        
        academic_score = child_state.knowledge
        emotional_score = (
            (100 - child_state.stress) * 0.3 +
            getattr(child_state, 'security_feeling', 70) * 0.4 +
            getattr(child_state, 'self_confidence', 60) * 0.3
        )
        social_score = getattr(child_state, 'social_skill', 50)
        health_score = child_state.physical_health
        relationship_score = (
            child_state.father_relationship * 0.3 +
            child_state.mother_relationship * 0.3 +
            child_state.grandfather_relationship * 0.2 +
            child_state.grandmother_relationship * 0.2
        )
        creativity_score = getattr(child_state, 'creativity', 50)
        resilience_score = getattr(child_state, 'resilience', 50)
        
        # 计算综合分数
        total_score = (
            academic_score * 0.25 +
            emotional_score * 0.20 +
            social_score * 0.15 +
            health_score * 0.10 +
            relationship_score * 0.15 +
            creativity_score * 0.10 +
            resilience_score * 0.05
        )
        
        # 确定等级
        if total_score >= 90:
            level = "S"
            career = "成为领域专家或企业高管的潜力很大"
        elif total_score >= 80:
            level = "A"
            career = "有望成为优秀的专业人才"
        elif total_score >= 70:
            level = "B"
            career = "能够找到稳定的工作，生活幸福"
        elif total_score >= 60:
            level = "C"
            career = "平稳发展，但可能需要更多努力"
        elif total_score >= 50:
            level = "D"
            career = "面临一些挑战，需要持续改进"
        else:
            level = "F"
            career = "发展遇到困难，需要大幅改变教育方式"
        
        # 生成评语
        assessment = self._generate_final_assessment(
            level, academic_score, emotional_score, 
            social_score, relationship_score
        )
        
        return FinalAchievementResult(
            achievement_level=level,
            academic_score=academic_score,
            emotional_score=emotional_score,
            social_score=social_score,
            health_score=health_score,
            relationship_score=relationship_score,
            creativity_score=creativity_score,
            resilience_score=resilience_score,
            career_prediction=career,
            final_assessment=assessment
        )
    
    def _generate_final_assessment(self,
                                    level: str,
                                    academic: float,
                                    emotional: float,
                                    social: float,
                                    relationship: float) -> str:
        """生成最终评语"""
        parts = []
        
        if level in ["S", "A"]:
            parts.append("这是一个非常成功的教育案例。")
        elif level in ["B", "C"]:
            parts.append("孩子的成长总体上是积极的。")
        else:
            parts.append("这个教育过程中存在一些需要反思的地方。")
        
        if academic > 70:
            parts.append(f"学业发展优秀（{academic:.0f}分），")
        elif academic < 50:
            parts.append(f"学业方面需要加强（{academic:.0f}分），")
        
        if emotional > 70:
            parts.append("情绪稳定且自信，")
        elif emotional < 50:
            parts.append("情绪管理需要关注，")
        
        if relationship > 80:
            parts.append("与家人关系非常融洽。")
        elif relationship < 60:
            parts.append("家庭关系有待改善。")
        else:
            parts.append("家庭关系良好。")
        
        return "".join(parts)


# 工厂函数
def create_llm_evaluator(llm_client, enable_detailed: bool = True) -> LLMEvaluator:
    """创建LLM评估器"""
    return LLMEvaluator(llm_client, enable_detailed)
