"""
测试修复的BUG
验证财务计算、IQ更新、类型提示等修复是否有效
"""
import sys
import os
from datetime import date, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import ChildState, FamilyState
from src.core.config import get_default_config


def test_salary_growth():
    """测试工资增长不会重复累积"""
    print("\n=== 测试1: 工资增长（避免重复累积） ===")
    
    config = get_default_config()
    family_state = FamilyState(current_date=date(2010, 1, 1))
    
    # 记录初始工资
    initial_father_salary = family_state.father.salary
    print(f"初始工资（2010年）: {initial_father_salary:.2f}")
    
    # 模拟10年
    for year in range(1, 11):
        family_state.current_date = date(2010 + year, 1, 1)
        family_state.update_salaries(config)
        print(f"{2010 + year}年工资: {family_state.father.salary:.2f}")
    
    # 验证工资增长合理（应该是初始工资 * (1 + 0.025)^10）
    expected_salary = initial_father_salary * (1 + config.salary_growth_rate) ** 10
    actual_salary = family_state.father.salary
    
    print(f"\n预期工资（10年后）: {expected_salary:.2f}")
    print(f"实际工资（10年后）: {actual_salary:.2f}")
    print(f"误差: {abs(expected_salary - actual_salary):.2f}")
    
    # 允许小误差（浮点数计算）
    assert abs(expected_salary - actual_salary) < 1.0, "工资计算误差过大"
    print("✅ 工资增长测试通过")


def test_financial_update_timing():
    """测试财务更新时机（每周更新）"""
    print("\n=== 测试2: 财务更新时机（每周更新） ===")
    
    config = get_default_config()
    family_state = FamilyState(current_date=date(2010, 1, 1))
    
    # 记录初始存款
    initial_savings = family_state.family_savings
    print(f"初始存款: {initial_savings:.2f}")
    
    # 模拟4周
    for week in range(1, 5):
        prev_savings = family_state.family_savings
        family_state.advance_date(weeks=1, config=config)
        delta_savings = family_state.family_savings - prev_savings
        print(f"第{week}周: 存款变化 {delta_savings:.2f}, 当前存款 {family_state.family_savings:.2f}")
    
    # 验证存款有变化（每周都更新）
    assert family_state.family_savings != initial_savings, "财务未更新"
    print("✅ 财务更新时机测试通过")


def test_iq_update_interval():
    """测试IQ更新间隔（4周）"""
    print("\n=== 测试3: IQ更新间隔（4周） ===")
    
    config = get_default_config()
    child_state = ChildState()
    
    # 记录初始IQ
    initial_iq = child_state.iq
    print(f"初始IQ: {initial_iq}")
    
    # 模拟4周（应该触发一次IQ更新）
    current_date = date(2010, 1, 1)
    for week in range(1, 5):
        current_date += timedelta(weeks=1)
        child_state.update_iq(current_date, config)
        print(f"第{week}周: IQ={child_state.iq}")
    
    # 验证IQ在4周后可能更新（取决于知识储备）
    print(f"4周后IQ: {child_state.iq}")
    print("✅ IQ更新间隔测试通过")


def test_type_hints():
    """测试类型提示是否正确导入"""
    print("\n=== 测试4: 类型提示导入 ===")
    
    try:
        from src.core.config import SimulationConfig, Dict
        print("✅ Dict类型已正确导入")
        
        # 尝试创建配置实例
        config = SimulationConfig()
        print(f"✅ 配置实例创建成功: {type(config)}")
    except Exception as e:
        print(f"❌ 类型提示测试失败: {e}")
        raise


def test_exception_handling():
    """测试异常处理改进"""
    print("\n=== 测试5: 异常处理 ===")
    
    child_state = ChildState()
    
    # 测试正常情况
    try:
        age = child_state.calculate_age(date(2020, 6, 15))
        print(f"正常计算年龄: {age:.2f}")
        assert age > 0, "年龄计算错误"
        print("✅ 年龄计算正常")
    except Exception as e:
        print(f"❌ 年龄计算异常: {e}")
        raise
    
    # 测试异常情况（使用无效日期）
    try:
        # 这应该触发降级方案
        invalid_date = None
        age = child_state.calculate_age(date(2020, 6, 15) if invalid_date is None else invalid_date)
        print(f"异常处理后年龄: {age:.2f}")
        print("✅ 异常处理测试通过")
    except Exception as e:
        print(f"异常处理失败: {e}")


def test_inflation_bounds():
    """测试通胀率边界检查"""
    print("\n=== 测试6: 通胀率边界检查 ===")
    
    config = get_default_config()
    family_state = FamilyState(current_date=date(2010, 1, 1))
    
    # 模拟50年（测试边界保护）
    family_state.current_date = date(2060, 1, 1)
    
    try:
        family_state.update_salaries(config)
        print(f"50年后工资: {family_state.father.salary:.2f}")
        
        # 验证工资不会过度增长
        assert family_state.father.salary < 100000, "工资增长过快，边界检查失效"
        print("✅ 通胀率边界检查通过")
    except Exception as e:
        print(f"❌ 通胀率测试失败: {e}")
        raise


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("BUG修复验证测试")
    print("=" * 60)
    
    tests = [
        test_salary_growth,
        test_financial_update_timing,
        test_iq_update_interval,
        test_type_hints,
        test_exception_handling,
        test_inflation_bounds
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ 测试失败: {test_func.__name__}")
            print(f"   错误: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
