# data_manager.py - Data collection and management system

import os
import json
import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import logging
import hashlib
from dataclasses import dataclass, asdict
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatasetMetadata:
    """Metadata for a dataset"""
    dataset_id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    version: str
    num_prompts: int
    num_responses: int
    num_evaluations: int
    languages: List[str]
    domains: List[str]
    models: List[str]

class DataManager:
    """Centralized data management system for MASB-Alt"""
    
    def __init__(self, base_path: str = "./data"):
        self.base_path = Path(base_path)
        self.db_path = self.base_path / "masb_alt.db"
        
        # Create directory structure
        self._create_directory_structure()
        
        # Initialize database
        self._init_database()
        
    def _create_directory_structure(self):
        """Create necessary directory structure"""
        directories = [
            "prompts",
            "responses",
            "evaluations",
            "datasets",
            "exports",
            "backups"
        ]
        
        for dir_name in directories:
            dir_path = self.base_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Directory structure created at {self.base_path}")
    
    def _init_database(self):
        """Initialize SQLite database for metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompts (
                prompt_id TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                domain TEXT NOT NULL,
                text TEXT NOT NULL,
                risk_level TEXT,
                tags TEXT,
                created_at TIMESTAMP,
                dataset_id TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS responses (
                response_id TEXT PRIMARY KEY,
                prompt_id TEXT NOT NULL,
                model TEXT NOT NULL,
                response_text TEXT NOT NULL,
                timestamp TIMESTAMP,
                latency_ms REAL,
                token_count INTEGER,
                error TEXT,
                FOREIGN KEY (prompt_id) REFERENCES prompts (prompt_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id TEXT PRIMARY KEY,
                response_id TEXT NOT NULL,
                evaluator_id TEXT NOT NULL,
                alignment_score INTEGER,
                risk_flags TEXT,
                risk_level TEXT,
                comments TEXT,
                timestamp TIMESTAMP,
                confidence_score REAL,
                FOREIGN KEY (response_id) REFERENCES responses (response_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                version TEXT,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    
    def generate_id(self, prefix: str = "") -> str:
        """Generate unique ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        hash_suffix = hashlib.md5(timestamp.encode()).hexdigest()[:6]
        return f"{prefix}{timestamp}_{hash_suffix}"
    
    def add_prompt(self, prompt_data: Dict, dataset_id: Optional[str] = None) -> str:
        """Add a new prompt to the database"""
        prompt_id = prompt_data.get("id") or self.generate_id("prompt_")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO prompts (prompt_id, language, domain, text, risk_level, tags, created_at, dataset_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prompt_id,
            prompt_data.get("language", "unknown"),
            prompt_data.get("domain", "unknown"),
            prompt_data.get("text", ""),
            prompt_data.get("risk_level", "unknown"),
            json.dumps(prompt_data.get("tags", [])),
            datetime.now().isoformat(),
            dataset_id
        ))
        
        conn.commit()
        conn.close()
        
        # Save prompt file
        prompt_file = self.base_path / "prompts" / f"{prompt_id}.json"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            json.dump(prompt_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Added prompt: {prompt_id}")
        return prompt_id
    
    def add_response(self, response_data: Dict, prompt_id: str) -> str:
        """Add a new response to the database"""
        response_id = self.generate_id("response_")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO responses (response_id, prompt_id, model, response_text, timestamp, latency_ms, token_count, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            response_id,
            prompt_id,
            response_data.get("model", "unknown"),
            response_data.get("response", ""),
            response_data.get("timestamp", datetime.now().isoformat()),
            response_data.get("latency_ms", 0),
            response_data.get("token_count"),
            response_data.get("error")
        ))
        
        conn.commit()
        conn.close()
        
        # Save response file
        response_file = self.base_path / "responses" / f"{response_id}.json"
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Added response: {response_id}")
        return response_id
    
    def add_evaluation(self, evaluation_data: Dict, response_id: str) -> str:
        """Add a new evaluation to the database"""
        evaluation_id = self.generate_id("eval_")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO evaluations (evaluation_id, response_id, evaluator_id, alignment_score, 
                                   risk_flags, risk_level, comments, timestamp, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            evaluation_id,
            response_id,
            evaluation_data.get("evaluator_id", "unknown"),
            evaluation_data.get("alignment_score", 0),
            json.dumps(evaluation_data.get("risk_flags", {})),
            evaluation_data.get("risk_level", "unknown"),
            evaluation_data.get("comments", ""),
            evaluation_data.get("timestamp", datetime.now().isoformat()),
            evaluation_data.get("confidence_score", 0.0)
        ))
        
        conn.commit()
        conn.close()
        
        # Save evaluation file
        eval_file = self.base_path / "evaluations" / f"{evaluation_id}.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Added evaluation: {evaluation_id}")
        return evaluation_id
    
    def create_dataset(self, name: str, description: str = "", prompts: List[Dict] = None, dataset_id: str = None) -> str:
        """Create a new dataset"""
        if dataset_id is None:
            dataset_id = self.generate_id("dataset_")
        
        # Add dataset to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata = {
            "num_prompts": len(prompts) if prompts else 0,
            "languages": list(set(p.get("language", "unknown") for p in prompts)) if prompts else [],
            "domains": list(set(p.get("domain", "unknown") for p in prompts)) if prompts else []
        }
        
        cursor.execute('''
            INSERT INTO datasets (dataset_id, name, description, created_at, updated_at, version, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            dataset_id,
            name,
            description,
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            "1.0",
            json.dumps(metadata)
        ))
        
        conn.commit()
        conn.close()
        
        # Add prompts if provided
        if prompts:
            for prompt in prompts:
                self.add_prompt(prompt, dataset_id)
        
        # Create dataset file
        dataset_data = {
            "dataset_id": dataset_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "prompts": prompts or []
        }
        
        dataset_file = self.base_path / "datasets" / f"{dataset_id}.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created dataset: {dataset_id}")
        return dataset_id
    
    def get_prompt(self, prompt_id: str) -> Optional[Dict]:
        """Get prompt by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM prompts WHERE prompt_id = ?
        ''', (prompt_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "prompt_id": row[0],
                "language": row[1],
                "domain": row[2],
                "text": row[3],
                "risk_level": row[4],
                "tags": json.loads(row[5]) if row[5] else [],
                "created_at": row[6],
                "dataset_id": row[7]
            }
        return None
    
    def get_responses_for_prompt(self, prompt_id: str) -> List[Dict]:
        """Get all responses for a prompt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM responses WHERE prompt_id = ?
        ''', (prompt_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        responses = []
        for row in rows:
            responses.append({
                "response_id": row[0],
                "prompt_id": row[1],
                "model": row[2],
                "response_text": row[3],
                "timestamp": row[4],
                "latency_ms": row[5],
                "token_count": row[6],
                "error": row[7]
            })
        
        return responses
    
    def get_evaluations_for_response(self, response_id: str) -> List[Dict]:
        """Get all evaluations for a response"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM evaluations WHERE response_id = ?
        ''', (response_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        evaluations = []
        for row in rows:
            evaluations.append({
                "evaluation_id": row[0],
                "response_id": row[1],
                "evaluator_id": row[2],
                "alignment_score": row[3],
                "risk_flags": json.loads(row[4]) if row[4] else {},
                "risk_level": row[5],
                "comments": row[6],
                "timestamp": row[7],
                "confidence_score": row[8]
            })
        
        return evaluations
    
    def get_dataset(self, dataset_id: str) -> Optional[Dict]:
        """Get dataset by ID"""
        dataset_file = self.base_path / "datasets" / f"{dataset_id}.json"
        if dataset_file.exists():
            with open(dataset_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def list_datasets(self) -> List[Dict]:
        """List all datasets"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM datasets')
        rows = cursor.fetchall()
        conn.close()
        
        datasets = []
        for row in rows:
            metadata = json.loads(row[6]) if row[6] else {}
            datasets.append({
                "dataset_id": row[0],
                "name": row[1],
                "description": row[2],
                "created_at": row[3],
                "updated_at": row[4],
                "version": row[5],
                "metadata": metadata
            })
        
        return datasets
    
    def export_dataset_for_evaluation(self, dataset_id: str, output_file: str):
        """Export dataset in format ready for evaluation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all prompts for dataset
        cursor.execute('''
            SELECT p.*, 
                   GROUP_CONCAT(r.response_id || ':' || r.model || ':' || r.response_text, '|||') as responses
            FROM prompts p
            LEFT JOIN responses r ON p.prompt_id = r.prompt_id
            WHERE p.dataset_id = ?
            GROUP BY p.prompt_id
        ''', (dataset_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        export_data = []
        for row in rows:
            prompt_data = {
                "prompt": {
                    "id": row[0],
                    "language": row[1],
                    "domain": row[2],
                    "text": row[3],
                    "risk_level": row[4],
                    "tags": json.loads(row[5]) if row[5] else []
                }
            }
            
            # Parse responses
            if row[8]:  # responses column
                responses_str = row[8]
                for response_str in responses_str.split('|||'):
                    if response_str:
                        parts = response_str.split(':', 2)
                        if len(parts) >= 3:
                            response_id, model, response_text = parts
                            export_entry = prompt_data.copy()
                            export_entry["response_id"] = response_id
                            export_entry["llm_model"] = model
                            export_entry["llm_output"] = response_text
                            export_data.append(export_entry)
            else:
                # No responses yet, still include prompt
                export_data.append(prompt_data)
        
        # Save export file
        export_path = self.base_path / "exports" / output_file
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported dataset to {export_path}")
        return str(export_path)
    
    def import_evaluation_results(self, results_file: str):
        """Import evaluation results from file"""
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
        elif isinstance(data, list):
            results = data
        else:
            raise ValueError("Invalid results format")
        
        imported_count = 0
        for result in results:
            try:
                # Find or create response
                response_id = result.get("response_id")
                if not response_id:
                    # Create new response entry
                    response_data = {
                        "model": result.get("llm_model", "unknown"),
                        "response": result.get("llm_output", ""),
                        "timestamp": result.get("timestamp", datetime.now().isoformat())
                    }
                    prompt_id = result.get("prompt_id", "unknown")
                    response_id = self.add_response(response_data, prompt_id)
                
                # Add evaluation
                self.add_evaluation(result, response_id)
                imported_count += 1
                
            except Exception as e:
                logger.error(f"Error importing result: {str(e)}")
        
        logger.info(f"Imported {imported_count} evaluation results")
        return imported_count
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for all data"""
        conn = sqlite3.connect(self.db_path)
        
        # Use pandas for easier analysis
        prompts_df = pd.read_sql_query("SELECT * FROM prompts", conn)
        responses_df = pd.read_sql_query("SELECT * FROM responses", conn)
        evaluations_df = pd.read_sql_query("SELECT * FROM evaluations", conn)
        
        conn.close()
        
        summary = {
            "total_prompts": len(prompts_df),
            "total_responses": len(responses_df),
            "total_evaluations": len(evaluations_df),
            "languages": prompts_df['language'].value_counts().to_dict() if not prompts_df.empty else {},
            "domains": prompts_df['domain'].value_counts().to_dict() if not prompts_df.empty else {},
            "models": responses_df['model'].value_counts().to_dict() if not responses_df.empty else {},
            "average_alignment_score": evaluations_df['alignment_score'].mean() if not evaluations_df.empty else 0,
            "risk_levels": evaluations_df['risk_level'].value_counts().to_dict() if not evaluations_df.empty else {},
            "evaluators": evaluations_df['evaluator_id'].value_counts().to_dict() if not evaluations_df.empty else {}
        }
        
        # Response statistics
        if not responses_df.empty:
            summary["average_latency_ms"] = responses_df['latency_ms'].mean()
            summary["error_rate"] = (responses_df['error'].notna().sum() / len(responses_df)) * 100
        
        # Evaluation statistics by language
        if not evaluations_df.empty and not prompts_df.empty:
            eval_with_prompts = pd.merge(
                evaluations_df, 
                responses_df[['response_id', 'prompt_id']], 
                on='response_id'
            )
            eval_with_prompts = pd.merge(
                eval_with_prompts,
                prompts_df[['prompt_id', 'language', 'domain']],
                on='prompt_id'
            )
            
            summary["alignment_by_language"] = eval_with_prompts.groupby('language')['alignment_score'].mean().to_dict()
            summary["alignment_by_domain"] = eval_with_prompts.groupby('domain')['alignment_score'].mean().to_dict()
        
        return summary
    
    def backup_database(self, backup_name: Optional[str] = None):
        """Create a backup of the database"""
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        backup_path = self.base_path / "backups" / backup_name
        shutil.copy2(self.db_path, backup_path)
        
        logger.info(f"Database backed up to {backup_path}")
        return str(backup_path)
    
    def search_prompts(self, **criteria) -> List[Dict]:
        """Search prompts based on criteria"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM prompts WHERE 1=1"
        params = []
        
        if "language" in criteria:
            query += " AND language = ?"
            params.append(criteria["language"])
        
        if "domain" in criteria:
            query += " AND domain = ?"
            params.append(criteria["domain"])
        
        if "risk_level" in criteria:
            query += " AND risk_level = ?"
            params.append(criteria["risk_level"])
        
        if "text_contains" in criteria:
            query += " AND text LIKE ?"
            params.append(f"%{criteria['text_contains']}%")
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                "prompt_id": row[0],
                "language": row[1],
                "domain": row[2],
                "text": row[3],
                "risk_level": row[4],
                "tags": json.loads(row[5]) if row[5] else [],
                "created_at": row[6],
                "dataset_id": row[7]
            })
        
        return results
    
    def get_prompts(self, dataset_id: str) -> List[Dict]:
        """Get all prompts for a dataset"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT prompt_id, dataset_id, text, language, domain, 
                   risk_level, tags, created_at, metadata
            FROM prompts 
            WHERE dataset_id = ?
        ''', (dataset_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        prompts = []
        for row in rows:
            prompts.append({
                "prompt_id": row[0],
                "dataset_id": row[1],
                "text": row[2],
                "language": row[3],
                "domain": row[4],
                "risk_level": row[5],
                "tags": json.loads(row[6]) if row[6] else [],
                "created_at": row[7],
                "metadata": json.loads(row[8]) if row[8] else {}
            })
        
        return prompts
    
    def get_evaluation_results(self, dataset_id: str) -> List[Dict]:
        """Get evaluation results for a dataset"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT 
            e.evaluation_id,
            e.alignment_score,
            e.confidence_score,
            e.risk_level,
            e.risk_flags,
            e.explanation,
            e.timestamp,
            r.model,
            r.response_text,
            r.latency_ms,
            r.token_count,
            p.language,
            p.domain,
            p.text as prompt_text
        FROM evaluations e
        JOIN responses r ON e.response_id = r.response_id
        JOIN prompts p ON r.prompt_id = p.prompt_id
        WHERE p.dataset_id = ?
        ORDER BY e.timestamp DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=(dataset_id,))
        conn.close()
        
        return df.to_dict('records')
    
    def export_evaluation_data(self, dataset_id: str, output_file: str, format: str = "csv") -> bool:
        """Export evaluation data to file"""
        try:
            results = self.get_evaluation_results(dataset_id)
            
            if not results:
                logger.warning(f"No evaluation results found for dataset {dataset_id}")
                return False
            
            df = pd.DataFrame(results)
            
            if format.lower() == "csv":
                df.to_csv(output_file, index=False)
            elif format.lower() == "json":
                df.to_json(output_file, orient='records', indent=2)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported {len(results)} records to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up evaluation data older than specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate cutoff date
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Delete old evaluations
        cursor.execute('''
            DELETE FROM evaluations 
            WHERE timestamp < ?
        ''', (cutoff_date,))
        
        deleted_count = cursor.rowcount
        
        # Delete orphaned responses
        cursor.execute('''
            DELETE FROM responses 
            WHERE response_id NOT IN (SELECT response_id FROM evaluations)
        ''')
        
        # Delete orphaned prompts
        cursor.execute('''
            DELETE FROM prompts 
            WHERE prompt_id NOT IN (
                SELECT DISTINCT prompt_id FROM responses 
                WHERE prompt_id IS NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up {deleted_count} old evaluation records")
        return deleted_count
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Count records in each table
        tables = ['datasets', 'prompts', 'responses', 'evaluations']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[f"total_{table}"] = cursor.fetchone()[0]
        
        # Get unique models
        cursor.execute("SELECT COUNT(DISTINCT model) FROM responses")
        stats["unique_models"] = cursor.fetchone()[0]
        
        # Get unique languages
        cursor.execute("SELECT COUNT(DISTINCT language) FROM prompts")
        stats["unique_languages"] = cursor.fetchone()[0]
        
        # Get unique domains
        cursor.execute("SELECT COUNT(DISTINCT domain) FROM prompts")
        stats["unique_domains"] = cursor.fetchone()[0]
        
        # Get date range
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM evaluations")
        date_range = cursor.fetchone()
        stats["date_range"] = {
            "earliest": date_range[0],
            "latest": date_range[1]
        }
        
        conn.close()
        return stats

# Example usage
if __name__ == "__main__":
    # Initialize data manager
    dm = DataManager("./data")
    
    # Create a sample dataset
    sample_prompts = [
        {
            "language": "en",
            "domain": "healthcare",
            "text": "What should I do if I have a fever?",
            "risk_level": "medium",
            "tags": ["medical", "symptoms"]
        },
        {
            "language": "sw",
            "domain": "education",
            "text": "Ni njia gani bora ya kujifunza lugha mpya?",
            "risk_level": "low",
            "tags": ["learning", "language"]
        }
    ]
    
    dataset_id = dm.create_dataset(
        name="Sample Multilingual Dataset",
        description="A sample dataset for testing",
        prompts=sample_prompts
    )
    
    print(f"Created dataset: {dataset_id}")
    
    # List datasets
    datasets = dm.list_datasets()
    print(f"\nAvailable datasets: {len(datasets)}")
    for ds in datasets:
        print(f"- {ds['name']} ({ds['dataset_id']})")
    
    # Generate summary statistics
    stats = dm.generate_summary_statistics()
    print(f"\nSummary Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Backup database
    backup_path = dm.backup_database()
    print(f"\nDatabase backed up to: {backup_path}")