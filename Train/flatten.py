from datetime import datetime
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import time


class LegalCaseProcessor:
    """Processes legal case JSON files and converts them to JSONL training format."""
    
    def __init__(self, input_files: List[str], output_file: str, completed_log: str):
        self.input_files = [Path(f) for f in input_files]
        self.output_file = Path(output_file)
        self.completed_log = Path(completed_log)
        
        # Initialize state tracking
        self.state_file = self.output_file.parent / "processing_state.json"
        self.state = self._load_state()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_state(self) -> Dict[str, Any]:
        """Load existing processing state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def flatten_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Convert one hybrid case into a chat-style training sample."""
        case_id = case.get("case_id", "UNKNOWN")
        instruction = case.get("instruction", "")
        question = case.get("question", "")
        analysis = case.get("analysis", {})
        metadata = case.get("metadata", {})
        
        # Safely extract nested data
        case_summary = analysis.get("case_summary", {})
        court_reasoning = analysis.get("court_reasoning", {})
        
        # Build structured response
        assistant_answer = self._build_assistant_response(
            case_summary, court_reasoning, metadata
        )
        
        return {
            "case_id": case_id,
            "messages": [
                {"role": "user", "content": f"{instruction}\n\n{question}".strip()},
                {"role": "assistant", "content": assistant_answer}
            ]
        }
    
    def _build_assistant_response(self, case_summary: Dict, court_reasoning: Dict, 
                                 metadata: Dict) -> str:
        """Build the assistant response string."""
        sections = []
        
        if any(case_summary.get(k) for k in ['facts', 'legal_issue', 'law_applied', 'judgment']):
            sections.append(
                "**Case Summary**\n"
                f"- Facts: {case_summary.get('facts', 'Not provided')}\n"
                f"- Legal Issue: {case_summary.get('legal_issue', 'Not provided')}\n"
                f"- Law Applied: {case_summary.get('law_applied', 'Not provided')}\n"
                f"- Judgment: {case_summary.get('judgment', 'Not provided')}"
            )
        
        if any(court_reasoning.get(k) for k in ['reasoning', 'decision']):
            sections.append(
                "**Court Reasoning**\n"
                f"- Reasoning: {court_reasoning.get('reasoning', 'Not provided')}\n"
                f"- Decision: {court_reasoning.get('decision', 'Not provided')}"
            )
        
        if any(metadata.get(k) for k in ['category', 'case_type', 'law']):
            sections.append(
                "**Metadata**\n"
                f"- Category: {metadata.get('category', 'Not specified')}\n"
                f"- Case Type: {metadata.get('case_type', 'Not specified')}\n"
                f"- Law: {metadata.get('law', 'Not specified')}"
            )
        
        return "\n\n".join(sections)
    
    def save_processing_state(self, file_path: Path, cases_processed: int, 
                             total_cases: int, start_time: float) -> None:
        """Save detailed processing state to a separate tracking file."""
        processing_time = time.time() - start_time
        
        self.state[str(file_path)] = {
            "cases_processed": cases_processed,
            "total_cases": total_cases,
            "processing_time_seconds": round(processing_time, 2),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "success_rate": round((cases_processed / total_cases * 100), 2) if total_cases > 0 else 0
        }
        
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except IOError as e:
            self.logger.warning(f"Could not save processing state: {e}")
    
    def get_total_cases_count(self) -> int:
        """Get total number of cases across all unprocessed files for progress bar."""
        total = 0
        for file_path in self.input_files:
            if self.already_completed(file_path):
                continue
            if not file_path.exists():
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    total += len(data)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Could not count cases in {file_path}: {e}")
        return total
    
    def load_processed_cases(self) -> set:
        """Load set of already processed case IDs to avoid duplicates."""
        processed_cases = set()
        
        # Check all potential output files
        output_files = [
            self.output_file,
            self.output_file.parent / "train_civil.jsonl",
            self.output_file.parent / "train_criminal.jsonl"
        ]
        
        for output_file in output_files:
            if output_file.exists():
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            try:
                                case_data = json.loads(line.strip())
                                case_id = case_data.get("case_id")
                                if case_id:
                                    processed_cases.add(case_id)
                            except json.JSONDecodeError:
                                self.logger.warning(f"Invalid JSON at line {line_num} in {output_file}")
                except IOError as e:
                    self.logger.warning(f"Could not read existing output file {output_file}: {e}")
        
        self.logger.info(f"Found {len(processed_cases)} already processed cases")
        return processed_cases
    
    def already_completed(self, file_path: Path) -> bool:
        """Check if file has already been processed (file-level)."""
        if not self.completed_log.exists():
            return False
        try:
            with open(self.completed_log, "r", encoding="utf-8") as f:
                completed = {line.strip() for line in f}
            return str(file_path) in completed
        except IOError as e:
            self.logger.warning(f"Could not read completed log: {e}")
            return False
    
    def mark_completed(self, file_path: Path) -> None:
        """Mark file as processed."""
        try:
            with open(self.completed_log, "a", encoding="utf-8") as f:
                f.write(str(file_path) + "\n")
        except IOError as e:
            self.logger.error(f"Could not write to completed log: {e}")
    
    def process_file(self, file_path: Path, pbar: Optional[tqdm] = None, 
                    processed_cases: Optional[set] = None) -> tuple[int, int]:
        """Process a single JSON file and return (cases_processed, cases_skipped)."""
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return 0, 0
        
        start_time = time.time()
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return 0, 0
        
        if not isinstance(data, list):
            self.logger.error(f"Expected list in {file_path}, got {type(data)}")
            return 0, 0
        
        cases_processed, cases_skipped = 0, 0
        total_cases = len(data)
        
        file_pbar = tqdm(
            total=total_cases,
            desc=f"Processing {file_path.name}",
            unit="cases",
            leave=False,
            disable=pbar is None
        )
        
        try:
            # Decide output file per input
            if "civil" in file_path.stem.lower():
                output_path = self.output_file.parent / "train_civil.jsonl"
            elif "criminal" in file_path.stem.lower():
                output_path = self.output_file.parent / "train_criminal.jsonl"
            else:
                output_path = self.output_file  # fallback

            with open(output_path, "a", encoding="utf-8") as fout:
                for i, case in enumerate(data, start=1):
                    try:
                        case_id = case.get("case_id", f"UNKNOWN_{i}")
                        
                        if processed_cases and case_id in processed_cases:
                            cases_skipped += 1
                        else:
                            flat_case = self.flatten_case(case)
                            fout.write(json.dumps(flat_case, ensure_ascii=False) + "\n")
                            cases_processed += 1
                            if processed_cases is not None:
                                processed_cases.add(case_id)
                        
                        file_pbar.update(1)
                        if pbar:
                            pbar.update(1)
                        
                        if i % 100 == 0:
                            file_pbar.set_postfix({
                                'processed': cases_processed,
                                'skipped': cases_skipped
                            })
                    
                    except Exception as e:
                        self.logger.warning(
                            f"Error processing case {i} ({case.get('case_id', 'UNKNOWN')}) in {file_path}: {e}"
                        )
                        file_pbar.update(1)
                        if pbar:
                            pbar.update(1)
                        continue
        except IOError as e:
            self.logger.error(f"Error writing to output file: {e}")
        finally:
            file_pbar.close()
        
        self.save_processing_state(file_path, cases_processed, total_cases, start_time)
        return cases_processed, cases_skipped
    
    def process_all(self) -> None:
        """Process all input files with progress tracking."""
        processed_cases = self.load_processed_cases()
        total_cases = self.get_total_cases_count()
        
        if total_cases == 0:
            self.logger.info("No new cases to process!")
            return
        
        main_pbar = tqdm(
            total=total_cases,
            desc="Overall Progress",
            unit="cases",
            colour="green"
        )
        
        total_written, total_skipped, files_processed = 0, 0, 0
        
        try:
            for file_path in self.input_files:
                if self.already_completed(file_path):
                    self.logger.info(f"✅ Skipping {file_path.name} (already completed)")
                    continue
                
                self.logger.info(f"📂 Processing {file_path.name} ...")
                cases_processed, cases_skipped = self.process_file(file_path, main_pbar, processed_cases)
                
                if cases_processed > 0 or cases_skipped > 0:
                    self.mark_completed(file_path)
                    total_written += cases_processed
                    total_skipped += cases_skipped
                    files_processed += 1
                    self.logger.info(
                        f"✅ Finished {file_path.name} - "
                        f"Processed: {cases_processed}, Skipped: {cases_skipped}"
                    )
                else:
                    self.logger.warning(f"❌ No cases processed from {file_path.name}")
        finally:
            main_pbar.close()
        
        self.logger.info(f"\n🎯 Processing Complete!")
        self.logger.info(f"Files processed: {files_processed}")
        self.logger.info(f"New training examples: {total_written}")
        self.logger.info(f"Duplicates skipped: {total_skipped}")
        self.logger.info(f"Total examples in dataset: {len(processed_cases)}")
        
        self._save_final_summary(files_processed, total_written, total_skipped, len(processed_cases))
    
    def _save_final_summary(self, files_processed: int, total_written: int, 
                           total_skipped: int, total_examples: int) -> None:
        """Save final processing summary."""
        summary_file = self.output_file.parent / "processing_summary.json"
        summary = {
            "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files_processed": files_processed,
            "new_examples_added": total_written,
            "duplicates_skipped": total_skipped,
            "total_examples_in_dataset": total_examples,
            "output_file": str(self.output_file),
            "input_files": [str(f) for f in self.input_files]
        }
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            self.logger.info(f"📊 Summary saved to {summary_file}")
        except IOError as e:
            self.logger.warning(f"Could not save summary: {e}")


def main():
    """Main function to run the legal case processor."""
    input_files = [
        r"C:\Users\Kushal\Desktop\new\Data_Verify\verified_civil.json",
        r"C:\Users\Kushal\Desktop\new\Data_Verify\verified_criminal.json"
    ]
    output_file = r"C:\Users\Kushal\Desktop\new\Train\train.jsonl"
    completed_log = r"C:\Users\Kushal\Desktop\new\Train\completed_files.txt"
    
    processor = LegalCaseProcessor(input_files, output_file, completed_log)
    processor.process_all()


if __name__ == "__main__":
    main()