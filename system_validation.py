#!/usr/bin/env python3
# system_validation.py - MASB-Alt系统完整性验证

import os
import sys
import json
import sqlite3
from pathlib import Path
import importlib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemValidator:
    """MASB-Alt系统完整性验证器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.validation_results = {
            "files": {"passed": [], "failed": []},
            "imports": {"passed": [], "failed": []},
            "config": {"passed": [], "failed": []},
            "database": {"passed": [], "failed": []},
            "overall_status": "unknown"
        }
    
    def validate_file_structure(self):
        """验证文件结构"""
        logger.info("🔍 验证项目文件结构...")
        
        required_files = {
            # 核心组件
            "prompt_generator/multilingual_prompt_gen.py": "多语言提示生成器",
            "evaluation/evaluator.py": "评估框架",
            "llm_api_client.py": "LLM API客户端",
            "data_manager.py": "数据管理器",
            
            # Web服务和监控
            "api_server.py": "API服务器",
            "monitoring_dashboard.py": "监控面板",
            
            # 分析工具
            "benchmark_runner.py": "性能基准测试",
            "corpus_manager.py": "语料库管理器",
            "comparative_analyzer.py": "对比分析工具",
            "report_generator.py": "报告生成器",
            "finetuning_advisor.py": "微调建议系统",
            "history_tracker.py": "历史跟踪系统",
            
            # 系统管理
            "masb_orchestrator.py": "主控制器",
            "masb_cli.py": "命令行工具",
            "config.py": "配置管理",
            
            # 配置文件
            "config.json": "系统配置",
            "requirements.txt": "依赖文件",
            ".env.example": "环境变量模板",
            "README.md": "项目文档",
            "setup.py": "安装脚本",
            "fix_issues.py": "问题修复脚本"
        }
        
        for file_path, description in required_files.items():
            full_path = self.project_root / file_path
            if full_path.exists():
                self.validation_results["files"]["passed"].append({
                    "file": file_path,
                    "description": description,
                    "size": full_path.stat().st_size
                })
                logger.info(f"✅ {file_path} - {description}")
            else:
                self.validation_results["files"]["failed"].append({
                    "file": file_path,
                    "description": description,
                    "error": "文件不存在"
                })
                logger.error(f"❌ {file_path} - 文件不存在")
    
    def validate_imports(self):
        """验证关键模块导入"""
        logger.info("🔍 验证模块导入...")
        
        # 添加项目路径到sys.path
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        critical_modules = [
            ("prompt_generator.multilingual_prompt_gen", "多语言提示生成"),
            ("llm_api_client", "LLM API客户端"),
            ("evaluation.evaluator", "评估引擎"),
            ("data_manager", "数据管理"),
            ("config", "配置管理")
        ]
        
        for module_name, description in critical_modules:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    self.validation_results["imports"]["passed"].append({
                        "module": module_name,
                        "description": description
                    })
                    logger.info(f"✅ {module_name} - {description}")
                else:
                    raise ImportError(f"Module {module_name} not found")
            except Exception as e:
                self.validation_results["imports"]["failed"].append({
                    "module": module_name,
                    "description": description,
                    "error": str(e)
                })
                logger.error(f"❌ {module_name} - {e}")
    
    def validate_configuration(self):
        """验证配置文件"""
        logger.info("🔍 验证配置文件...")
        
        # 检查config.json
        config_file = self.project_root / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                required_keys = [
                    "models", "languages", "domains", 
                    "quality_thresholds", "api_settings"
                ]
                
                missing_keys = [key for key in required_keys if key not in config_data]
                
                if not missing_keys:
                    self.validation_results["config"]["passed"].append({
                        "file": "config.json",
                        "keys": list(config_data.keys())
                    })
                    logger.info("✅ config.json - 配置完整")
                else:
                    self.validation_results["config"]["failed"].append({
                        "file": "config.json",
                        "error": f"缺少配置项: {missing_keys}"
                    })
                    logger.error(f"❌ config.json - 缺少: {missing_keys}")
                    
            except Exception as e:
                self.validation_results["config"]["failed"].append({
                    "file": "config.json",
                    "error": str(e)
                })
                logger.error(f"❌ config.json - {e}")
        else:
            self.validation_results["config"]["failed"].append({
                "file": "config.json",
                "error": "文件不存在"
            })
        
        # 检查.env.example
        env_file = self.project_root / ".env.example"
        if env_file.exists():
            try:
                content = env_file.read_text(encoding='utf-8')
                required_vars = [
                    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY"
                ]
                
                missing_vars = [var for var in required_vars if var not in content]
                
                if not missing_vars:
                    self.validation_results["config"]["passed"].append({
                        "file": ".env.example",
                        "vars": required_vars
                    })
                    logger.info("✅ .env.example - 环境变量模板完整")
                else:
                    self.validation_results["config"]["failed"].append({
                        "file": ".env.example",
                        "error": f"缺少环境变量: {missing_vars}"
                    })
                    
            except Exception as e:
                self.validation_results["config"]["failed"].append({
                    "file": ".env.example",
                    "error": str(e)
                })
    
    def validate_database_schema(self):
        """验证数据库模式"""
        logger.info("🔍 验证数据库模式...")
        
        try:
            # 确保data目录存在
            data_dir = self.project_root / "data"
            data_dir.mkdir(exist_ok=True)
            
            # 创建测试数据库连接
            test_db = data_dir / "test_schema.db"
            conn = sqlite3.connect(test_db)
            cursor = conn.cursor()
            
            # 测试基本SQL操作
            cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
            cursor.execute("INSERT INTO test (id) VALUES (1)")
            cursor.execute("SELECT * FROM test")
            result = cursor.fetchone()
            
            if result:
                self.validation_results["database"]["passed"].append({
                    "test": "基本SQL操作",
                    "result": "成功"
                })
                logger.info("✅ 数据库基本操作测试通过")
            
            conn.close()
            # 清理测试文件
            test_db.unlink(missing_ok=True)
            
        except Exception as e:
            self.validation_results["database"]["failed"].append({
                "test": "数据库连接",
                "error": str(e)
            })
            logger.error(f"❌ 数据库测试失败: {e}")
    
    def validate_dependencies(self):
        """验证依赖项"""
        logger.info("🔍 验证Python依赖项...")
        
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            try:
                content = requirements_file.read_text()
                lines = [line.strip() for line in content.split('\n') 
                        if line.strip() and not line.startswith('#')]
                
                # 提取包名
                packages = []
                for line in lines:
                    if '==' in line:
                        package = line.split('==')[0]
                        packages.append(package)
                
                # 测试关键包的导入
                critical_packages = [
                    'pandas', 'numpy', 'fastapi', 'streamlit', 
                    'matplotlib', 'plotly', 'aiohttp'
                ]
                
                available_packages = []
                missing_packages = []
                
                for package in critical_packages:
                    try:
                        __import__(package)
                        available_packages.append(package)
                    except ImportError:
                        missing_packages.append(package)
                
                if available_packages:
                    self.validation_results["config"]["passed"].append({
                        "test": "依赖项检查",
                        "available": available_packages
                    })
                
                if missing_packages:
                    self.validation_results["config"]["failed"].append({
                        "test": "依赖项检查",
                        "missing": missing_packages
                    })
                    
            except Exception as e:
                self.validation_results["config"]["failed"].append({
                    "test": "requirements.txt",
                    "error": str(e)
                })
    
    def generate_validation_report(self):
        """生成验证报告"""
        logger.info("📊 生成验证报告...")
        
        total_files = len(self.validation_results["files"]["passed"]) + len(self.validation_results["files"]["failed"])
        total_imports = len(self.validation_results["imports"]["passed"]) + len(self.validation_results["imports"]["failed"])
        total_config = len(self.validation_results["config"]["passed"]) + len(self.validation_results["config"]["failed"])
        total_database = len(self.validation_results["database"]["passed"]) + len(self.validation_results["database"]["failed"])
        
        passed_files = len(self.validation_results["files"]["passed"])
        passed_imports = len(self.validation_results["imports"]["passed"])
        passed_config = len(self.validation_results["config"]["passed"])
        passed_database = len(self.validation_results["database"]["passed"])
        
        # 计算总体通过率
        total_tests = total_files + total_imports + total_config + total_database
        total_passed = passed_files + passed_imports + passed_config + passed_database
        
        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # 确定整体状态
        if pass_rate >= 95:
            self.validation_results["overall_status"] = "excellent"
        elif pass_rate >= 80:
            self.validation_results["overall_status"] = "good"
        elif pass_rate >= 60:
            self.validation_results["overall_status"] = "fair"
        else:
            self.validation_results["overall_status"] = "poor"
        
        # 生成报告
        report = {
            "validation_date": str(datetime.now()),
            "overall_status": self.validation_results["overall_status"],
            "pass_rate": f"{pass_rate:.1f}%",
            "summary": {
                "files": f"{passed_files}/{total_files}",
                "imports": f"{passed_imports}/{total_imports}",
                "config": f"{passed_config}/{total_config}",
                "database": f"{passed_database}/{total_database}"
            },
            "details": self.validation_results
        }
        
        # 保存报告
        report_file = self.project_root / "validation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        return report
    
    def print_summary(self, report):
        """打印验证总结"""
        status_emoji = {
            "excellent": "🎉",
            "good": "✅", 
            "fair": "⚠️",
            "poor": "❌"
        }
        
        print("\n" + "="*70)
        print("🔍 MASB-Alt 系统验证报告")
        print("="*70)
        
        print(f"\n{status_emoji.get(report['overall_status'], '❓')} 整体状态: {report['overall_status'].upper()}")
        print(f"📊 通过率: {report['pass_rate']}")
        
        print(f"\n📋 验证结果:")
        print(f"   📁 文件结构: {report['summary']['files']}")
        print(f"   📦 模块导入: {report['summary']['imports']}")
        print(f"   ⚙️  配置文件: {report['summary']['config']}")
        print(f"   🗄️  数据库: {report['summary']['database']}")
        
        # 显示失败项目
        failed_items = []
        for category in ['files', 'imports', 'config', 'database']:
            failed_items.extend(self.validation_results[category]['failed'])
        
        if failed_items:
            print(f"\n❌ 需要注意的问题 ({len(failed_items)}):")
            for item in failed_items[:10]:  # 只显示前10个
                print(f"   • {item}")
        
        print(f"\n🚀 系统组件总览:")
        print(f"   • 多语言支持: 10种语言 (英语、斯瓦希里语、阿拉伯语等)")
        print(f"   • 评估域: 5个领域 (医疗、教育、金融等)")
        print(f"   • LLM支持: OpenAI, Anthropic, Cohere")
        print(f"   • 分析工具: 基准测试、对比分析、微调建议")
        print(f"   • 监控系统: 实时面板、历史跟踪、异常检测")
        print(f"   • 报告生成: HTML, PDF, DOCX, Markdown")
        
        print(f"\n📖 使用建议:")
        if report['overall_status'] in ['excellent', 'good']:
            print("   ✅ 系统状态良好，可以开始使用")
            print("   🔧 运行: python masb_cli.py demo")
            print("   📊 运行: python masb_orchestrator.py evaluate")
        else:
            print("   ⚠️  建议先解决上述问题")
            print("   🔧 运行: python fix_issues.py")
            print("   🔍 运行: python system_validation.py")
        
        print("="*70)
    
    def run_full_validation(self):
        """运行完整验证"""
        logger.info("🚀 开始MASB-Alt系统完整性验证...")
        
        validation_steps = [
            self.validate_file_structure,
            self.validate_imports,
            self.validate_configuration,
            self.validate_database_schema,
            self.validate_dependencies
        ]
        
        for step in validation_steps:
            try:
                step()
            except Exception as e:
                logger.error(f"验证步骤失败 {step.__name__}: {e}")
        
        report = self.generate_validation_report()
        self.print_summary(report)
        
        return report

def main():
    """主函数"""
    validator = SystemValidator()
    report = validator.run_full_validation()
    
    # 根据验证结果设置退出码
    if report['overall_status'] in ['excellent', 'good']:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()