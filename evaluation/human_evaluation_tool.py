# human_evaluation_tool.py - Interactive tool for human evaluation

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from evaluator import (
    MultilingualAlignmentEvaluator, 
    EvaluationResult, 
    AlignmentScore, 
    RiskFlags, 
    RiskLevel
)

class HumanEvaluationGUI:
    """GUI tool for human evaluation of LLM responses"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Multilingual Alignment Human Evaluation Tool")
        self.root.geometry("1200x800")
        
        # Initialize evaluator
        self.evaluator = MultilingualAlignmentEvaluator()
        
        # Data management
        self.current_data = []
        self.current_index = 0
        self.evaluation_results = []
        self.evaluator_name = ""
        
        # Setup GUI
        self._setup_ui()
        self._bind_shortcuts()
        
    def _setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(header_frame, text="Evaluator Name:").grid(row=0, column=0, padx=(0, 5))
        self.evaluator_name_var = tk.StringVar()
        ttk.Entry(header_frame, textvariable=self.evaluator_name_var, width=30).grid(row=0, column=1, padx=(0, 20))
        
        ttk.Button(header_frame, text="Load Data", command=self._load_data).grid(row=0, column=2, padx=5)
        ttk.Button(header_frame, text="Save Progress", command=self._save_progress).grid(row=0, column=3, padx=5)
        ttk.Button(header_frame, text="Export Results", command=self._export_results).grid(row=0, column=4, padx=5)
        
        # Progress indicator
        self.progress_var = tk.StringVar(value="No data loaded")
        ttk.Label(header_frame, textvariable=self.progress_var).grid(row=0, column=5, padx=(20, 0))
        
        # Content area
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Prompt and Response
        left_panel = ttk.Frame(content_frame)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        content_frame.columnconfigure(0, weight=2)
        content_frame.rowconfigure(0, weight=1)
        
        # Prompt info
        prompt_info_frame = ttk.LabelFrame(left_panel, text="Prompt Information", padding="10")
        prompt_info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.prompt_id_var = tk.StringVar()
        self.language_var = tk.StringVar()
        self.domain_var = tk.StringVar()
        self.model_var = tk.StringVar()
        
        ttk.Label(prompt_info_frame, text="ID:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(prompt_info_frame, textvariable=self.prompt_id_var).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(prompt_info_frame, text="Language:").grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
        ttk.Label(prompt_info_frame, textvariable=self.language_var).grid(row=0, column=3, sticky=tk.W)
        
        ttk.Label(prompt_info_frame, text="Domain:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(prompt_info_frame, textvariable=self.domain_var).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(prompt_info_frame, text="Model:").grid(row=1, column=2, sticky=tk.W, padx=(20, 5))
        ttk.Label(prompt_info_frame, textvariable=self.model_var).grid(row=1, column=3, sticky=tk.W)
        
        # Prompt text
        prompt_frame = ttk.LabelFrame(left_panel, text="Prompt", padding="10")
        prompt_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        left_panel.rowconfigure(1, weight=1)
        
        self.prompt_text = tk.Text(prompt_frame, height=5, wrap=tk.WORD)
        self.prompt_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        prompt_frame.columnconfigure(0, weight=1)
        prompt_frame.rowconfigure(0, weight=1)
        
        prompt_scroll = ttk.Scrollbar(prompt_frame, orient=tk.VERTICAL, command=self.prompt_text.yview)
        prompt_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.prompt_text.configure(yscrollcommand=prompt_scroll.set)
        
        # Response text
        response_frame = ttk.LabelFrame(left_panel, text="LLM Response", padding="10")
        response_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        left_panel.rowconfigure(2, weight=2)
        
        self.response_text = tk.Text(response_frame, height=10, wrap=tk.WORD)
        self.response_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        response_frame.columnconfigure(0, weight=1)
        response_frame.rowconfigure(0, weight=1)
        
        response_scroll = ttk.Scrollbar(response_frame, orient=tk.VERTICAL, command=self.response_text.yview)
        response_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.response_text.configure(yscrollcommand=response_scroll.set)
        
        # Right panel - Evaluation
        right_panel = ttk.Frame(content_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(1, weight=1)
        
        # Alignment score
        score_frame = ttk.LabelFrame(right_panel, text="Alignment Score", padding="10")
        score_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.score_var = tk.IntVar(value=3)
        scores = [
            ("1 - Poor", 1),
            ("2 - Below Average", 2),
            ("3 - Average", 3),
            ("4 - Good", 4),
            ("5 - Excellent", 5)
        ]
        
        for i, (label, value) in enumerate(scores):
            ttk.Radiobutton(score_frame, text=label, variable=self.score_var, 
                           value=value).grid(row=i, column=0, sticky=tk.W, pady=2)
        
        # Risk flags
        risk_frame = ttk.LabelFrame(right_panel, text="Risk Flags", padding="10")
        risk_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.risk_vars = {
            "hallucination": tk.BooleanVar(),
            "unsafe_medical_advice": tk.BooleanVar(),
            "culturally_insensitive": tk.BooleanVar(),
            "non_compliant": tk.BooleanVar(),
            "harmful_content": tk.BooleanVar(),
            "misleading_information": tk.BooleanVar(),
            "inappropriate_tone": tk.BooleanVar(),
            "privacy_violation": tk.BooleanVar()
        }
        
        risk_labels = {
            "hallucination": "Hallucination",
            "unsafe_medical_advice": "Unsafe Medical Advice",
            "culturally_insensitive": "Culturally Insensitive",
            "non_compliant": "Non-compliant",
            "harmful_content": "Harmful Content",
            "misleading_information": "Misleading Information",
            "inappropriate_tone": "Inappropriate Tone",
            "privacy_violation": "Privacy Violation"
        }
        
        for i, (key, var) in enumerate(self.risk_vars.items()):
            ttk.Checkbutton(risk_frame, text=risk_labels[key], 
                           variable=var).grid(row=i, column=0, sticky=tk.W, pady=2)
        
        # Comments
        comments_frame = ttk.LabelFrame(right_panel, text="Comments", padding="10")
        comments_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        right_panel.rowconfigure(2, weight=1)
        
        self.comments_text = tk.Text(comments_frame, height=5, wrap=tk.WORD)
        self.comments_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        comments_frame.columnconfigure(0, weight=1)
        comments_frame.rowconfigure(0, weight=1)
        
        # Auto-evaluation suggestion
        auto_eval_frame = ttk.LabelFrame(right_panel, text="Auto-Evaluation Suggestion", padding="10")
        auto_eval_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        self.auto_eval_var = tk.StringVar(value="Click 'Get Suggestion' to see auto-evaluation")
        ttk.Label(auto_eval_frame, textvariable=self.auto_eval_var, 
                 wraplength=300).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(auto_eval_frame, text="Get Suggestion", 
                  command=self._get_auto_suggestion).grid(row=1, column=0, pady=(5, 0))
        
        # Navigation
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(nav_frame, text="<< Previous", command=self._previous_item).grid(row=0, column=0, padx=5)
        ttk.Button(nav_frame, text="Save & Next >>", command=self._save_and_next).grid(row=0, column=1, padx=5)
        ttk.Button(nav_frame, text="Skip", command=self._skip_item).grid(row=0, column=2, padx=5)
        
        # Keyboard shortcuts info
        ttk.Label(nav_frame, text="Shortcuts: Ctrl+S (Save & Next), Ctrl+P (Previous), Ctrl+K (Skip)", 
                 font=('TkDefaultFont', 9, 'italic')).grid(row=0, column=3, padx=(20, 0))
    
    def _bind_shortcuts(self):
        """Bind keyboard shortcuts"""
        self.root.bind('<Control-s>', lambda e: self._save_and_next())
        self.root.bind('<Control-p>', lambda e: self._previous_item())
        self.root.bind('<Control-k>', lambda e: self._skip_item())
        self.root.bind('<Control-g>', lambda e: self._get_auto_suggestion())
    
    def _load_data(self):
        """Load evaluation data from file"""
        filename = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Expected format: list of {prompt, llm_output, llm_model}
            if isinstance(data, list):
                self.current_data = data
            elif isinstance(data, dict) and 'data' in data:
                self.current_data = data['data']
            else:
                messagebox.showerror("Error", "Invalid data format")
                return
            
            self.current_index = 0
            self.evaluation_results = []
            self._display_current_item()
            self._update_progress()
            
            messagebox.showinfo("Success", f"Loaded {len(self.current_data)} items")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def _display_current_item(self):
        """Display current item for evaluation"""
        if not self.current_data or self.current_index >= len(self.current_data):
            return
        
        item = self.current_data[self.current_index]
        
        # Update prompt info
        prompt = item.get('prompt', {})
        self.prompt_id_var.set(prompt.get('id', 'N/A'))
        self.language_var.set(prompt.get('language', 'N/A'))
        self.domain_var.set(prompt.get('domain', 'N/A'))
        self.model_var.set(item.get('llm_model', 'N/A'))
        
        # Update texts
        self.prompt_text.delete(1.0, tk.END)
        self.prompt_text.insert(1.0, prompt.get('text', ''))
        
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(1.0, item.get('llm_output', ''))
        
        # Reset evaluation fields
        self.score_var.set(3)
        for var in self.risk_vars.values():
            var.set(False)
        self.comments_text.delete(1.0, tk.END)
        self.auto_eval_var.set("Click 'Get Suggestion' to see auto-evaluation")
    
    def _update_progress(self):
        """Update progress indicator"""
        if not self.current_data:
            self.progress_var.set("No data loaded")
        else:
            evaluated = len(self.evaluation_results)
            total = len(self.current_data)
            self.progress_var.set(f"Progress: {evaluated}/{total} ({evaluated/total*100:.1f}%)")
    
    def _get_auto_suggestion(self):
        """Get auto-evaluation suggestion"""
        if not self.current_data or self.current_index >= len(self.current_data):
            return
        
        item = self.current_data[self.current_index]
        
        # Run auto-evaluation
        try:
            result = self.evaluator.evaluate_response(
                prompt=item.get('prompt', {}),
                llm_output=item.get('llm_output', ''),
                llm_model=item.get('llm_model', 'unknown'),
                evaluator_id="auto"
            )
            
            # Display suggestion
            suggestion = f"Score: {result.alignment_score.value}\n"
            suggestion += f"Risk Level: {result.risk_level.value}\n"
            suggestion += f"Flags: {', '.join(result.risk_flags.get_flagged_risks()) or 'None'}\n"
            suggestion += f"Comments: {result.comments}"
            
            self.auto_eval_var.set(suggestion)
            
        except Exception as e:
            self.auto_eval_var.set(f"Error: {str(e)}")
    
    def _save_current_evaluation(self):
        """Save current evaluation"""
        if not self.current_data or self.current_index >= len(self.current_data):
            return False
        
        if not self.evaluator_name_var.get():
            messagebox.showwarning("Warning", "Please enter your evaluator name")
            return False
        
        item = self.current_data[self.current_index]
        
        # Create risk flags
        risk_flags = RiskFlags()
        for key, var in self.risk_vars.items():
            setattr(risk_flags, key, var.get())
        
        # Determine risk level
        risk_level = self.evaluator._determine_risk_level(
            risk_flags, 
            item.get('prompt', {}).get('domain', 'general')
        )
        
        # Create evaluation result
        result = EvaluationResult(
            prompt_id=item.get('prompt', {}).get('id', 'unknown'),
            language=item.get('prompt', {}).get('language', 'unknown'),
            domain=item.get('prompt', {}).get('domain', 'unknown'),
            prompt_text=item.get('prompt', {}).get('text', ''),
            llm_output=item.get('llm_output', ''),
            llm_model=item.get('llm_model', 'unknown'),
            alignment_score=AlignmentScore(self.score_var.get()),
            risk_flags=risk_flags,
            risk_level=risk_level,
            comments=self.comments_text.get(1.0, tk.END).strip(),
            evaluator_id=self.evaluator_name_var.get(),
            timestamp=datetime.now(),
            confidence_score=0.9  # Human evaluation has high confidence
        )
        
        self.evaluation_results.append(result)
        return True
    
    def _save_and_next(self):
        """Save current evaluation and move to next"""
        if self._save_current_evaluation():
            self.current_index += 1
            if self.current_index < len(self.current_data):
                self._display_current_item()
            else:
                messagebox.showinfo("Complete", "All items have been evaluated!")
            self._update_progress()
    
    def _previous_item(self):
        """Go to previous item"""
        if self.current_index > 0:
            self.current_index -= 1
            self._display_current_item()
    
    def _skip_item(self):
        """Skip current item"""
        self.current_index += 1
        if self.current_index < len(self.current_data):
            self._display_current_item()
        else:
            messagebox.showinfo("Complete", "Reached end of data")
        self._update_progress()
    
    def _save_progress(self):
        """Save evaluation progress"""
        if not self.evaluation_results:
            messagebox.showwarning("Warning", "No evaluations to save")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save progress",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            self.evaluator.save_results(self.evaluation_results, filename)
            messagebox.showinfo("Success", f"Saved {len(self.evaluation_results)} evaluations")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def _export_results(self):
        """Export evaluation results with summary"""
        if not self.evaluation_results:
            messagebox.showwarning("Warning", "No evaluations to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Generate summary
            summary = self.evaluator.generate_summary_report(self.evaluation_results)
            
            # Create export data
            export_data = {
                "metadata": {
                    "evaluator": self.evaluator_name_var.get(),
                    "total_items": len(self.current_data),
                    "evaluated_items": len(self.evaluation_results),
                    "export_timestamp": datetime.now().isoformat()
                },
                "summary": summary,
                "detailed_results": [result.to_dict() for result in self.evaluation_results]
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo("Success", f"Exported results to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")

def main():
    """Main entry point"""
    root = tk.Tk()
    app = HumanEvaluationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()