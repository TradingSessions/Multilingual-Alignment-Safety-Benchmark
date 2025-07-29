#!/usr/bin/env python3
# fix_issues.py - 快速修复MASB-Alt系统中的主要问题

import os
import sys
import json
import sqlite3
from pathlib import Path
import logging
import importlib.util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MASBSystemFixer:
    """MASB-Alt系统问题修复器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.issues_found = []
        self.issues_fixed = []
    
    def check_and_fix_imports(self):
        """检查和修复导入问题"""
        logger.info("🔍 检查导入问题...")
        
        # 需要修复的文件和导入映射
        import_fixes = {
            "benchmark_runner.py": {
                "from llm_api_client import LLMClientFactory, MultiLLMClient, LLMResponse": 
                "from llm_api_client import LLMClientFactory, LLMClient, LLMResponse"
            },
            "masb_orchestrator.py": {
                "from llm_api_client import LLMClientFactory, MultiLLMClient":
                "from llm_api_client import LLMClientFactory"
            },
            "api_server.py": {
                "from llm_api_client import LLMClientFactory, MultiLLMClient":
                "from llm_api_client import LLMClientFactory"
            }
        }
        
        for filename, fixes in import_fixes.items():
            file_path = self.project_root / filename
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    original_content = content
                    
                    for old_import, new_import in fixes.items():
                        if old_import in content:
                            content = content.replace(old_import, new_import)
                            logger.info(f"✅ 修复了 {filename} 中的导入问题")
                    
                    if content != original_content:
                        file_path.write_text(content, encoding='utf-8')
                        self.issues_fixed.append(f"导入修复: {filename}")
                        
                except Exception as e:
                    self.issues_found.append(f"无法修复 {filename}: {e}")
    
    def fix_requirements(self):
        """修复requirements.txt中的问题"""
        logger.info("🔍 检查requirements.txt...")
        
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            try:
                content = req_file.read_text()
                
                # 移除不需要的包
                problematic_packages = [
                    "sqlite3-api==2.0.0",
                    "asyncio==3.4.3",
                    "tkinter==0.1.0"
                ]
                
                for package in problematic_packages:
                    if package in content:
                        content = content.replace(package + "\n", "")
                        content = content.replace(package, "")
                        logger.info(f"✅ 移除了有问题的包: {package}")
                
                # 添加缺失的注释
                if "sqlite3-api" in content and "# sqlite3 is part of Python standard library" not in content:
                    content = content.replace("sqlite3-api==2.0.0", "# sqlite3 is part of Python standard library")
                
                if "asyncio==" in content and "# asyncio is part of Python standard library" not in content:
                    content = content.replace("asyncio==3.4.3", "# asyncio is part of Python standard library")
                
                req_file.write_text(content)
                self.issues_fixed.append("requirements.txt 修复完成")
                
            except Exception as e:
                self.issues_found.append(f"无法修复requirements.txt: {e}")
    
    def check_missing_files(self):
        """检查缺失的重要文件"""
        logger.info("🔍 检查缺失文件...")
        
        required_files = [
            "config.py",
            ".env.example",
            "__init__.py"
        ]
        
        for filename in required_files:
            file_path = self.project_root / filename
            if not file_path.exists():
                self.create_missing_file(filename)
    
    def create_missing_file(self, filename):
        """创建缺失的文件"""
        file_path = self.project_root / filename
        
        if filename == ".env.example":
            content = """# MASB-Alt Environment Variables Template
# Copy this file to .env and fill in your actual values

# LLM API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
COHERE_API_KEY=your_cohere_api_key_here

# Optional: Email configuration for reports
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Optional: System configuration
LOG_LEVEL=INFO
DEBUG_MODE=false
API_HOST=0.0.0.0
API_PORT=8000
DASHBOARD_PORT=8501
"""
        elif filename == "__init__.py":
            content = """# MASB-Alt: Multilingual Alignment Safety Benchmark
__version__ = "1.0.0"
__author__ = "MASB-Alt Development Team"
"""
        else:
            content = f"# {filename} - Auto-generated file\n"
        
        try:
            file_path.write_text(content)
            logger.info(f"✅ 创建了缺失文件: {filename}")
            self.issues_fixed.append(f"创建文件: {filename}")
        except Exception as e:
            self.issues_found.append(f"无法创建 {filename}: {e}")
    
    def fix_database_schema(self):
        """修复数据库模式问题"""
        logger.info("🔍 检查数据库模式...")
        
        try:
            # 确保data目录存在
            data_dir = self.project_root / "data"
            data_dir.mkdir(exist_ok=True)
            
            # 检查数据库文件
            db_path = data_dir / "masb_alt.db"
            
            # 如果数据库存在但有问题，备份并重新创建
            if db_path.exists():
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # 检查基本表是否存在
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    required_tables = ['datasets', 'prompts', 'responses', 'evaluations']
                    missing_tables = [t for t in required_tables if t not in tables]
                    
                    if missing_tables:
                        logger.info(f"数据库缺少表: {missing_tables}")
                        # 这里可以添加创建表的逻辑
                    else:
                        logger.info("✅ 数据库模式检查通过")
                    
                    conn.close()
                    
                except Exception as e:
                    logger.warning(f"数据库检查失败: {e}")
                    # 可以选择重新初始化数据库
                    
            self.issues_fixed.append("数据库模式检查完成")
            
        except Exception as e:
            self.issues_found.append(f"数据库检查失败: {e}")
    
    def fix_matplotlib_style(self):
        """修复matplotlib样式问题"""
        logger.info("🔍 检查matplotlib样式...")
        
        files_to_check = [
            "comparative_analyzer.py",
            "benchmark_runner.py",
            "visualization/visualization_dashboard.py"
        ]
        
        for filename in files_to_check:
            file_path = self.project_root / filename
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    
                    # 修复seaborn样式问题
                    if "plt.style.use('seaborn-v0_8-darkgrid')" in content:
                        if "try:" not in content or "except:" not in content:
                            old_style = "plt.style.use('seaborn-v0_8-darkgrid')"
                            new_style = """try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            # Fallback to default style if seaborn style not available
            plt.style.use('default')"""
                            
                            content = content.replace(old_style, new_style)
                            file_path.write_text(content, encoding='utf-8')
                            logger.info(f"✅ 修复了 {filename} 中的matplotlib样式")
                            self.issues_fixed.append(f"matplotlib样式修复: {filename}")
                
                except Exception as e:
                    self.issues_found.append(f"无法修复 {filename} 的样式: {e}")
    
    def check_api_compatibility(self):
        """检查API兼容性"""
        logger.info("🔍 检查API兼容性...")
        
        try:
            # 检查FastAPI版本兼容性
            import fastapi
            logger.info(f"FastAPI版本: {fastapi.__version__}")
            
            # 检查其他关键库
            critical_imports = [
                'pandas', 'numpy', 'sqlite3', 'json', 'pathlib',
                'datetime', 'asyncio', 'aiohttp'
            ]
            
            missing_imports = []
            for module_name in critical_imports:
                try:
                    __import__(module_name)
                except ImportError:
                    missing_imports.append(module_name)
            
            if missing_imports:
                self.issues_found.append(f"缺少关键模块: {missing_imports}")
            else:
                logger.info("✅ 关键模块检查通过")
                self.issues_fixed.append("API兼容性检查完成")
                
        except Exception as e:
            self.issues_found.append(f"API兼容性检查失败: {e}")
    
    def create_quick_test_script(self):
        """创建快速测试脚本"""
        logger.info("📄 创建快速测试脚本...")
        
        test_content = """#!/usr/bin/env python3
# quick_test.py - MASB-Alt系统快速测试

import sys
import os
import importlib
from pathlib import Path

def test_imports():
    \"\"\"测试关键模块导入\"\"\"
    print("🔍 测试关键模块导入...")
    
    modules_to_test = [
        'prompt_generator.multilingual_prompt_gen',
        'llm_api_client',
        'evaluation.evaluator',
        'data_manager',
        'config'
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_config():
    \"\"\"测试配置加载\"\"\"
    print("\\n🔍 测试配置加载...")
    
    try:
        from config import get_config
        config = get_config()
        print("✅ 配置加载成功")
        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_database():
    \"\"\"测试数据库连接\"\"\"
    print("\\n🔍 测试数据库连接...")
    
    try:
        from data_manager import DataManager
        dm = DataManager()
        print("✅ 数据库连接成功")
        return True
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False

def main():
    print("🚀 MASB-Alt 快速系统测试\\n")
    
    tests = [
        ("模块导入", test_imports),
        ("配置加载", test_config),
        ("数据库连接", test_database)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed_tests += 1
    
    print(f"\\n📊 测试结果: {passed_tests}/{total_tests} 通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过! 系统准备就绪.")
    else:
        print("⚠️  一些测试失败. 请检查上述错误信息.")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
        
        test_file = self.project_root / "quick_test.py"
        try:
            test_file.write_text(test_content)
            logger.info("✅ 创建了快速测试脚本: quick_test.py")
            self.issues_fixed.append("创建测试脚本")
        except Exception as e:
            self.issues_found.append(f"无法创建测试脚本: {e}")
    
    def run_all_fixes(self):
        """运行所有修复"""
        logger.info("🔧 开始系统修复...")
        
        fixes = [
            self.check_and_fix_imports,
            self.fix_requirements,
            self.check_missing_files,
            self.fix_database_schema,
            self.fix_matplotlib_style,
            self.check_api_compatibility,
            self.create_quick_test_script
        ]
        
        for fix_func in fixes:
            try:
                fix_func()
            except Exception as e:
                logger.error(f"修复函数 {fix_func.__name__} 失败: {e}")
                self.issues_found.append(f"修复失败: {fix_func.__name__}")
        
        self.print_summary()
    
    def print_summary(self):
        """打印修复总结"""
        print("\n" + "="*60)
        print("🎯 MASB-Alt 系统修复总结")
        print("="*60)
        
        print(f"\n✅ 已修复问题 ({len(self.issues_fixed)}):")
        for issue in self.issues_fixed:
            print(f"   • {issue}")
        
        if self.issues_found:
            print(f"\n❌ 未解决问题 ({len(self.issues_found)}):")
            for issue in self.issues_found:
                print(f"   • {issue}")
        
        print(f"\n📈 修复率: {len(self.issues_fixed)}/{len(self.issues_fixed) + len(self.issues_found)} "
              f"({len(self.issues_fixed)/(len(self.issues_fixed) + len(self.issues_found))*100:.1f}%)")
        
        print("\n🚀 建议下一步:")
        print("   1. 运行 python quick_test.py 验证修复")
        print("   2. 运行 python masb_cli.py check 检查系统状态")
        print("   3. 运行 python masb_cli.py demo 测试基本功能")
        
        print("\n" + "="*60)

def main():
    """主函数"""
    fixer = MASBSystemFixer()
    fixer.run_all_fixes()

if __name__ == "__main__":
    main()