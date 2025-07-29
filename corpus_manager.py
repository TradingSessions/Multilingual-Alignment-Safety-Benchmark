# corpus_manager.py - Multilingual corpus management system

import os
import json
import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import hashlib
import logging
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict
import re
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CorpusEntry:
    """Single corpus entry"""
    entry_id: str
    text: str
    language: str
    domain: str
    source: str
    metadata: Dict[str, Any]
    quality_score: float
    verified: bool
    created_at: datetime
    updated_at: datetime
    
@dataclass
class TranslationPair:
    """Translation pair between two languages"""
    pair_id: str
    source_language: str
    target_language: str
    source_text: str
    target_text: str
    alignment_score: float
    verified: bool
    metadata: Dict[str, Any]

class CorpusManager:
    """Comprehensive multilingual corpus management system"""
    
    def __init__(self, base_path: str = "./corpus"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.base_path / "corpus.db"
        self._init_database()
        
        # Language detection patterns
        self.language_patterns = {
            "en": r"[a-zA-Z\s.,!?]+",
            "ar": r"[\u0600-\u06FF\s.,!?]+",
            "hi": r"[\u0900-\u097F\s.,!?]+",
            "sw": r"[a-zA-Z\s.,!?]+",  # Swahili uses Latin script
            "vi": r"[a-zA-Zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ\s.,!?]+",
            "ug": r"[\u0626-\u0628\u062A-\u063A\u0641-\u0642\u0644-\u0648\u064A\u067E\u0686\u0698\u06AD\u06AF\u06BE\u06C6-\u06C8\u06CB\u06D0\u06D5\s.,!?]+"
        }
        
        # Quality metrics
        self.quality_metrics = {
            "min_length": 10,
            "max_length": 5000,
            "min_words": 3,
            "max_repetition_ratio": 0.3,
            "min_unique_words_ratio": 0.4
        }
    
    def _init_database(self):
        """Initialize SQLite database for corpus management"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Corpus entries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS corpus_entries (
                entry_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                language TEXT NOT NULL,
                domain TEXT NOT NULL,
                source TEXT NOT NULL,
                metadata TEXT,
                quality_score REAL,
                verified INTEGER DEFAULT 0,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                text_hash TEXT UNIQUE
            )
        ''')
        
        # Translation pairs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS translation_pairs (
                pair_id TEXT PRIMARY KEY,
                source_language TEXT NOT NULL,
                target_language TEXT NOT NULL,
                source_text TEXT NOT NULL,
                target_text TEXT NOT NULL,
                source_entry_id TEXT,
                target_entry_id TEXT,
                alignment_score REAL,
                verified INTEGER DEFAULT 0,
                metadata TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (source_entry_id) REFERENCES corpus_entries (entry_id),
                FOREIGN KEY (target_entry_id) REFERENCES corpus_entries (entry_id)
            )
        ''')
        
        # Corpus statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS corpus_stats (
                stat_id TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                domain TEXT NOT NULL,
                entry_count INTEGER,
                avg_length REAL,
                avg_quality_score REAL,
                verified_count INTEGER,
                last_updated TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_language ON corpus_entries (language)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON corpus_entries (domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_quality ON corpus_entries (quality_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_translation_langs ON translation_pairs (source_language, target_language)')
        
        conn.commit()
        conn.close()
        
        logger.info("Corpus database initialized")
    
    def add_entry(self, 
                  text: str,
                  language: str,
                  domain: str,
                  source: str,
                  metadata: Optional[Dict] = None,
                  verified: bool = False) -> Optional[str]:
        """Add a new corpus entry"""
        # Clean and validate text
        text = self._clean_text(text)
        if not self._validate_text(text, language):
            logger.warning(f"Text validation failed for language {language}")
            return None
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(text, language)
        
        # Generate entry ID and hash
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        entry_id = f"{language}_{domain}_{text_hash[:8]}"
        
        # Check for duplicates
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT entry_id FROM corpus_entries WHERE text_hash = ?', (text_hash,))
        if cursor.fetchone():
            logger.info(f"Duplicate entry detected for {entry_id}")
            conn.close()
            return None
        
        # Insert entry
        try:
            cursor.execute('''
                INSERT INTO corpus_entries 
                (entry_id, text, language, domain, source, metadata, quality_score, 
                 verified, created_at, updated_at, text_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry_id, text, language, domain, source,
                json.dumps(metadata or {}), quality_score,
                int(verified), datetime.now(), datetime.now(), text_hash
            ))
            
            conn.commit()
            logger.info(f"Added corpus entry: {entry_id}")
            
        except Exception as e:
            logger.error(f"Error adding corpus entry: {e}")
            entry_id = None
        finally:
            conn.close()
        
        # Update statistics
        self._update_statistics(language, domain)
        
        return entry_id
    
    def add_translation_pair(self,
                            source_text: str,
                            target_text: str,
                            source_language: str,
                            target_language: str,
                            domain: str,
                            source: str = "manual",
                            verified: bool = False) -> Optional[str]:
        """Add a translation pair"""
        # Add source and target as corpus entries
        source_id = self.add_entry(source_text, source_language, domain, source, verified=verified)
        target_id = self.add_entry(target_text, target_language, domain, source, verified=verified)
        
        if not source_id or not target_id:
            return None
        
        # Calculate alignment score
        alignment_score = self._calculate_alignment_score(source_text, target_text)
        
        # Create pair ID
        pair_id = f"{source_language}_{target_language}_{hashlib.md5(f'{source_id}{target_id}'.encode()).hexdigest()[:8]}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO translation_pairs
                (pair_id, source_language, target_language, source_text, target_text,
                 source_entry_id, target_entry_id, alignment_score, verified, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pair_id, source_language, target_language, source_text, target_text,
                source_id, target_id, alignment_score, int(verified),
                json.dumps({"source": source}), datetime.now()
            ))
            
            conn.commit()
            logger.info(f"Added translation pair: {pair_id}")
            
        except Exception as e:
            logger.error(f"Error adding translation pair: {e}")
            pair_id = None
        finally:
            conn.close()
        
        return pair_id
    
    def import_corpus_file(self,
                          file_path: str,
                          language: str,
                          domain: str,
                          source: str,
                          file_format: str = "jsonl") -> int:
        """Import corpus from file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        imported_count = 0
        
        if file_format == "jsonl":
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Importing {language} corpus"):
                    try:
                        data = json.loads(line.strip())
                        text = data.get("text", "")
                        metadata = {k: v for k, v in data.items() if k != "text"}
                        
                        if self.add_entry(text, language, domain, source, metadata):
                            imported_count += 1
                    except Exception as e:
                        logger.error(f"Error importing line: {e}")
        
        elif file_format == "txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Importing {language} corpus"):
                    text = line.strip()
                    if text and self.add_entry(text, language, domain, source):
                        imported_count += 1
        
        elif file_format == "csv":
            df = pd.read_csv(file_path)
            text_column = "text" if "text" in df.columns else df.columns[0]
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Importing {language} corpus"):
                text = str(row[text_column])
                metadata = row.to_dict()
                metadata.pop(text_column, None)
                
                if self.add_entry(text, language, domain, source, metadata):
                    imported_count += 1
        
        logger.info(f"Imported {imported_count} entries from {file_path}")
        return imported_count
    
    def search_corpus(self,
                     query: str,
                     language: Optional[str] = None,
                     domain: Optional[str] = None,
                     min_quality_score: float = 0.0,
                     verified_only: bool = False,
                     limit: int = 100) -> List[Dict]:
        """Search corpus with various filters"""
        conn = sqlite3.connect(self.db_path)
        
        # Build query
        sql = "SELECT * FROM corpus_entries WHERE 1=1"
        params = []
        
        if query:
            sql += " AND text LIKE ?"
            params.append(f"%{query}%")
        
        if language:
            sql += " AND language = ?"
            params.append(language)
        
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        
        sql += " AND quality_score >= ?"
        params.append(min_quality_score)
        
        if verified_only:
            sql += " AND verified = 1"
        
        sql += f" ORDER BY quality_score DESC LIMIT {limit}"
        
        cursor = conn.cursor()
        cursor.execute(sql, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "entry_id": row[0],
                "text": row[1],
                "language": row[2],
                "domain": row[3],
                "source": row[4],
                "metadata": json.loads(row[5]) if row[5] else {},
                "quality_score": row[6],
                "verified": bool(row[7]),
                "created_at": row[8]
            })
        
        conn.close()
        return results
    
    def get_translation_pairs(self,
                             source_language: str,
                             target_language: str,
                             domain: Optional[str] = None,
                             min_alignment_score: float = 0.0,
                             limit: int = 100) -> List[Dict]:
        """Get translation pairs for language pair"""
        conn = sqlite3.connect(self.db_path)
        
        sql = '''
            SELECT tp.*, ce1.domain
            FROM translation_pairs tp
            JOIN corpus_entries ce1 ON tp.source_entry_id = ce1.entry_id
            WHERE tp.source_language = ? AND tp.target_language = ?
            AND tp.alignment_score >= ?
        '''
        params = [source_language, target_language, min_alignment_score]
        
        if domain:
            sql += " AND ce1.domain = ?"
            params.append(domain)
        
        sql += f" ORDER BY tp.alignment_score DESC LIMIT {limit}"
        
        cursor = conn.cursor()
        cursor.execute(sql, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "pair_id": row[0],
                "source_language": row[1],
                "target_language": row[2],
                "source_text": row[3],
                "target_text": row[4],
                "alignment_score": row[7],
                "verified": bool(row[8]),
                "domain": row[11]
            })
        
        conn.close()
        return results
    
    def export_corpus(self,
                     output_file: str,
                     language: Optional[str] = None,
                     domain: Optional[str] = None,
                     format: str = "jsonl",
                     verified_only: bool = False) -> int:
        """Export corpus to file"""
        # Get entries
        entries = self.search_corpus(
            query="",
            language=language,
            domain=domain,
            verified_only=verified_only,
            limit=1000000  # Large limit
        )
        
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        elif format == "csv":
            df = pd.DataFrame(entries)
            df.to_csv(output_path, index=False)
        
        elif format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(entry['text'] + '\n')
        
        logger.info(f"Exported {len(entries)} entries to {output_path}")
        return len(entries)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Trim
        text = text.strip()
        
        return text
    
    def _validate_text(self, text: str, language: str) -> bool:
        """Validate text for given language"""
        # Check length
        if len(text) < self.quality_metrics["min_length"]:
            return False
        if len(text) > self.quality_metrics["max_length"]:
            return False
        
        # Check word count
        words = text.split()
        if len(words) < self.quality_metrics["min_words"]:
            return False
        
        # Check language pattern (basic check)
        if language in self.language_patterns:
            pattern = self.language_patterns[language]
            if not re.search(pattern, text):
                return False
        
        return True
    
    def _calculate_quality_score(self, text: str, language: str) -> float:
        """Calculate quality score for text"""
        score = 1.0
        
        # Length score
        optimal_length = 200
        length_diff = abs(len(text) - optimal_length)
        length_score = max(0, 1 - length_diff / 1000)
        score *= (0.3 + 0.7 * length_score)
        
        # Word diversity score
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score *= (0.5 + 0.5 * unique_ratio)
        
        # Repetition check
        if len(words) > 3:
            repetitions = sum(1 for i in range(len(words)-2) 
                            if words[i] == words[i+1] == words[i+2])
            repetition_ratio = repetitions / (len(words) - 2)
            score *= (1 - repetition_ratio)
        
        # Language-specific adjustments
        if language in ["ar", "ug"]:  # RTL languages
            # Check for proper RTL markers
            if '\u200F' in text or '\u200E' in text:
                score *= 1.1
        
        return min(1.0, max(0.0, score))
    
    def _calculate_alignment_score(self, source_text: str, target_text: str) -> float:
        """Calculate alignment score between two texts"""
        # Simple heuristic based on length ratio
        source_len = len(source_text)
        target_len = len(target_text)
        
        if source_len == 0 or target_len == 0:
            return 0.0
        
        # Length ratio score
        length_ratio = min(source_len, target_len) / max(source_len, target_len)
        
        # Word count ratio
        source_words = len(source_text.split())
        target_words = len(target_text.split())
        
        if source_words == 0 or target_words == 0:
            word_ratio = 0.0
        else:
            word_ratio = min(source_words, target_words) / max(source_words, target_words)
        
        # Combined score
        score = 0.7 * length_ratio + 0.3 * word_ratio
        
        return min(1.0, max(0.0, score))
    
    def _update_statistics(self, language: str, domain: str):
        """Update corpus statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as entry_count,
                AVG(LENGTH(text)) as avg_length,
                AVG(quality_score) as avg_quality,
                SUM(verified) as verified_count
            FROM corpus_entries
            WHERE language = ? AND domain = ?
        ''', (language, domain))
        
        stats = cursor.fetchone()
        
        if stats and stats[0] > 0:
            stat_id = f"{language}_{domain}"
            
            cursor.execute('''
                INSERT OR REPLACE INTO corpus_stats
                (stat_id, language, domain, entry_count, avg_length, 
                 avg_quality_score, verified_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                stat_id, language, domain, stats[0], stats[1],
                stats[2], stats[3], datetime.now()
            ))
            
            conn.commit()
        
        conn.close()
    
    def get_statistics(self) -> pd.DataFrame:
        """Get corpus statistics"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM corpus_stats", conn)
        conn.close()
        return df
    
    def verify_entry(self, entry_id: str) -> bool:
        """Mark entry as verified"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE corpus_entries 
            SET verified = 1, updated_at = ?
            WHERE entry_id = ?
        ''', (datetime.now(), entry_id))
        
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return affected > 0
    
    def create_augmented_corpus(self,
                               base_language: str,
                               target_languages: List[str],
                               domain: str,
                               augmentation_factor: int = 3) -> Dict[str, int]:
        """Create augmented corpus by generating variations"""
        results = {}
        
        # Get base entries
        base_entries = self.search_corpus(
            query="",
            language=base_language,
            domain=domain,
            limit=10000
        )
        
        for target_lang in target_languages:
            augmented_count = 0
            
            for entry in tqdm(base_entries, desc=f"Augmenting for {target_lang}"):
                # Here you would integrate with translation API
                # For demo, we'll create placeholder
                for i in range(augmentation_factor):
                    augmented_text = f"[{target_lang}] {entry['text']} (variant {i+1})"
                    
                    if self.add_entry(
                        text=augmented_text,
                        language=target_lang,
                        domain=domain,
                        source="augmented",
                        metadata={"base_entry_id": entry['entry_id'], "variant": i+1}
                    ):
                        augmented_count += 1
            
            results[target_lang] = augmented_count
        
        return results

# Example usage and CLI
def main():
    """Command-line interface for corpus manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multilingual Corpus Manager")
    parser.add_argument("action", choices=["import", "export", "search", "stats", "verify"])
    parser.add_argument("--file", help="File path for import/export")
    parser.add_argument("--language", help="Language code")
    parser.add_argument("--domain", help="Domain")
    parser.add_argument("--format", choices=["jsonl", "csv", "txt"], default="jsonl")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--entry-id", help="Entry ID for verification")
    
    args = parser.parse_args()
    
    manager = CorpusManager()
    
    if args.action == "import":
        if not args.file or not args.language or not args.domain:
            print("Error: --file, --language, and --domain required for import")
            return
        
        count = manager.import_corpus_file(
            args.file, args.language, args.domain, 
            source="cli_import", file_format=args.format
        )
        print(f"Imported {count} entries")
    
    elif args.action == "export":
        if not args.file:
            print("Error: --file required for export")
            return
        
        count = manager.export_corpus(
            args.file, language=args.language, 
            domain=args.domain, format=args.format
        )
        print(f"Exported {count} entries to {args.file}")
    
    elif args.action == "search":
        results = manager.search_corpus(
            query=args.query or "",
            language=args.language,
            domain=args.domain
        )
        
        for result in results[:10]:
            print(f"\n[{result['entry_id']}] {result['language']} - {result['domain']}")
            print(f"Text: {result['text'][:100]}...")
            print(f"Quality: {result['quality_score']:.2f}, Verified: {result['verified']}")
    
    elif args.action == "stats":
        stats = manager.get_statistics()
        print("\nCorpus Statistics:")
        print(stats.to_string())
    
    elif args.action == "verify":
        if not args.entry_id:
            print("Error: --entry-id required for verification")
            return
        
        if manager.verify_entry(args.entry_id):
            print(f"Entry {args.entry_id} verified")
        else:
            print(f"Entry {args.entry_id} not found")

if __name__ == "__main__":
    main()