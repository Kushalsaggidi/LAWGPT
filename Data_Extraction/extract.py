import os
import json
import asyncio
import re
import time
import psutil
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque

import pdfplumber
import aiohttp
import aiofiles
import aiofiles.os
from tqdm.asyncio import tqdm

# ====================== CONFIGURATION ======================
@dataclass
class Config:
    # Paths
    base_dir: Path = Path(r"C:\Users\Kushal\Desktop\new\Data_Extraction\data\court_36_29_jan")
    civil_output: Path = Path("civil_cases.json")
    criminal_output: Path = Path("criminal_cases.json") 
    progress_file: Path = Path("processing_progress.json")
    
    # API Configuration
    api_keys: List[str] = field(default_factory=lambda: [
        'AIzaSyByofJwHz-ULKTZP3TboSnDSgxgTIrA9OM',
        'AIzaSyCz41qZibYl1rca9Ye7_pkKaaBlQnms6ME',
        'AIzaSyA27TjinZixsBBl1ZkY7dRV9dK5Hf-ZzbA',
        'AIzaSyBFEIHo4GKXri8a3rNAaQKq0EwRSrwjwYs',
        'AIzaSyCAEEIT9lc5WxLAyq_N4SaKouAggntDxC4'
    ])
    model_variants: List[str] = field(default_factory=lambda: [
        "gemini-2.5-flash-lite", 
        "gemini-2.5-flash", 
        "gemini-2.5-pro"
    ])
    
    # Processing Limits - STREAMING APPROACH
    max_concurrency: int = 4
    pdf_concurrency: int = 3  # Further reduced for stability
    batch_size: int = 5  # Process in small batches
    batch_max_chars: int = 15000
    per_item_text_cap: int = 4000  # Reduced to handle more files
    pdf_timeout: int = 10  # Shorter timeout
    
    # Streaming settings
    file_chunk_size: int = 50  # Process files in chunks
    save_every: int = 10  # Save more frequently
    
    # Performance
    retry_cycles: int = 2  # Fewer retries for faster processing
    backoff_base: float = 1.2
    backoff_max: float = 5.0
    
    # Resource Limits
    max_memory_gb: float = 4.0  # Lower memory limit
    max_cpu_percent: float = 75.0

config = Config()

# ====================== METRICS & MONITORING ======================
@dataclass
class ProcessingMetrics:
    start_time: datetime = field(default_factory=datetime.now)
    total_discovered: int = 0
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    pdf_extracted: int = 0
    pdf_failed: int = 0
    api_calls: int = 0
    api_errors: int = 0
    current_chunk: int = 0
    
    def success_rate(self) -> float:
        return self.successful / max(self.total_processed, 1)
    
    def processing_rate(self) -> float:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return self.total_processed / max(elapsed / 60, 0.01)  # per minute

metrics = ProcessingMetrics()

# ====================== UTILITIES ======================
async def atomic_write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    async with aiofiles.open(tmp, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(data, ensure_ascii=False, indent=2))
    await aiofiles.os.rename(tmp, path)

async def safe_load_json(path: Path, default: Any = None):
    if not await aiofiles.os.path.exists(path):
        return default or {}
    
    try:
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return json.loads(content) if content.strip() else (default or {})
    except Exception:
        return default or {}

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.encode("utf-8", "ignore").decode()
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ====================== STREAMING FILE PROCESSOR ======================
class StreamingFileProcessor:
    def __init__(self):
        self.pdf_semaphore = asyncio.Semaphore(config.pdf_concurrency)
        self.api_semaphore = asyncio.Semaphore(config.max_concurrency)
        self.pdf_cache = {}
        self.failed_pdfs = set()
    
    async def discover_files_stream(self) -> AsyncGenerator[Path, None]:
        """Stream files one by one instead of loading all"""
        skip_files = {config.civil_output.name, config.criminal_output.name, config.progress_file.name}
        
        if not config.base_dir.exists():
            print(f"❌ Base directory does not exist: {config.base_dir}")
            return
        
        count = 0
        for root, _, filenames in os.walk(config.base_dir):
            for filename in filenames:
                if (filename.lower().endswith('.json') and 
                    filename not in skip_files):
                    count += 1
                    if count % 100 == 0:
                        print(f"📁 Discovered {count} files...")
                    yield Path(root) / filename
        
        metrics.total_discovered = count
        print(f"✅ Total files discovered: {count}")
    
    def _extract_pdf_sync(self, pdf_path: str) -> str:
        """Synchronous PDF extraction with size limits"""
        if not pdf_path or not os.path.exists(pdf_path):
            return ""
        
        try:
            # Check file size first
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            if file_size_mb > 50:  # Skip very large files
                print(f"⚠️ Skipping large PDF ({file_size_mb:.1f}MB): {os.path.basename(pdf_path)}")
                return "Large file skipped"
            
            pages = []
            with pdfplumber.open(pdf_path) as pdf:
                # Limit pages to process
                max_pages = min(len(pdf.pages), 20)  # Only first 20 pages
                
                for i, page in enumerate(pdf.pages[:max_pages]):
                    try:
                        text = page.extract_text()
                        if text:
                            pages.append(text)
                        # Early break if we have enough content
                        if len('\n'.join(pages)) > config.per_item_text_cap:
                            break
                    except Exception:
                        continue
            
            return "\n\n".join(pages)
        except Exception:
            return ""
    
    async def extract_pdf_text(self, meta: dict, case_id: str) -> str:
        """Extract PDF text with caching and error handling"""
        pdf_link = meta.get("pdf_link") or meta.get("pdf_path", "")
        if not pdf_link:
            return "No PDF link"
        
        # Generate PDF path
        if pdf_link.startswith("http"):
            pdf_path = config.base_dir / Path(pdf_link).name
        else:
            pdf_path = config.base_dir / pdf_link if not Path(pdf_link).is_absolute() else Path(pdf_link)
        
        pdf_path_str = str(pdf_path)
        
        # Check cache and failed list
        if pdf_path_str in self.pdf_cache:
            return self.pdf_cache[pdf_path_str]
        
        if pdf_path_str in self.failed_pdfs:
            return "Previously failed"
        
        if not pdf_path.exists():
            self.failed_pdfs.add(pdf_path_str)
            return "File not found"
        
        async with self.pdf_semaphore:
            try:
                text = await asyncio.wait_for(
                    asyncio.to_thread(self._extract_pdf_sync, pdf_path_str),
                    timeout=config.pdf_timeout
                )
                
                if text and text.strip():
                    text = clean_text(text)
                    if len(text) > config.per_item_text_cap:
                        text = text[:config.per_item_text_cap] + "\n[Truncated]"
                    
                    self.pdf_cache[pdf_path_str] = text
                    metrics.pdf_extracted += 1
                    return text
                else:
                    self.failed_pdfs.add(pdf_path_str)
                    metrics.pdf_failed += 1
                    return "No text extracted"
                    
            except asyncio.TimeoutError:
                self.failed_pdfs.add(pdf_path_str)
                metrics.pdf_failed += 1
                return "PDF timeout"
            except Exception:
                self.failed_pdfs.add(pdf_path_str)
                metrics.pdf_failed += 1
                return "Extraction error"

# ====================== API PROCESSOR ======================
class APIProcessor:
    def __init__(self):
        self.session = None
        self.api_keys = config.api_keys
        self.current_key_index = 0
    
    async def initialize(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
    
    async def close(self):
        if self.session:
            await self.session.close()
    
    def get_next_api_key(self):
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return key
    
    async def process_case(self, case_data: dict) -> Tuple[Optional[dict], Optional[str]]:
        """Process a single case through the API"""
        api_key = self.get_next_api_key()
        model = config.model_variants[0]  # Use fastest model
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        prompt = f"""Analyze this Indian legal case and return JSON with:
"case_id", "instruction", "question", "facts", "charge", "law", "law_content", "judgment", "category", "case_type"

Case Type Rules:
- Criminal: FIR, IPC, CrPC, NDPS, POCSO, bail, accused
- Civil: Property, contract, injunction, writ, compensation

Case Data:
{json.dumps(case_data, ensure_ascii=False)}"""
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }
        
        try:
            metrics.api_calls += 1
            async with self.session.post(url, json=payload) as resp:
                if resp.status != 200:
                    metrics.api_errors += 1
                    return None, f"HTTP {resp.status}"
                
                data = await resp.json(content_type=None)
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                
                try:
                    result = json.loads(text)
                    if isinstance(result, list) and result:
                        result = result[0]
                    
                    # Ensure required fields
                    result["case_id"] = case_data.get("case_id", "unknown")
                    result.setdefault("instruction", "Parse this legal judgment")
                    result.setdefault("question", "What was the court's decision?")
                    
                    return result, None
                    
                except json.JSONDecodeError:
                    metrics.api_errors += 1
                    return None, "Invalid JSON response"
                
        except Exception as e:
            metrics.api_errors += 1
            return None, f"API error: {str(e)[:100]}"

# ====================== RESULT MANAGER ======================
class StreamingResultManager:
    def __init__(self):
        self.civil_cases = []
        self.criminal_cases = []
        self.progress = {}
        self.save_counter = 0
    
    async def initialize(self):
        self.civil_cases = await safe_load_json(config.civil_output, [])
        self.criminal_cases = await safe_load_json(config.criminal_output, [])
        self.progress = await safe_load_json(config.progress_file, {})
        print(f"📊 Loaded progress: {len(self.progress)} entries")
    
    async def add_result(self, case_id: str, result: dict, error: Optional[str]):
        if error:
            self.progress[case_id] = {"status": "failed", "error": error}
            metrics.failed += 1
        else:
            case_type = result.get("case_type", "").lower()
            if "criminal" in case_type or "crim" in case_type:
                self.criminal_cases.append(result)
            else:
                self.civil_cases.append(result)
            
            self.progress[case_id] = {"status": "success"}
            metrics.successful += 1
        
        metrics.total_processed += 1
        self.save_counter += 1
        
        if self.save_counter >= config.save_every:
            await self.save_all()
            self.save_counter = 0
    
    async def save_all(self):
        await asyncio.gather(
            atomic_write_json(config.civil_output, self.civil_cases),
            atomic_write_json(config.criminal_output, self.criminal_cases),
            atomic_write_json(config.progress_file, self.progress)
        )
        print(f"💾 Saved: {len(self.civil_cases)} civil, {len(self.criminal_cases)} criminal cases")

# ====================== MAIN STREAMING PIPELINE ======================
async def process_file_chunk(file_processor: StreamingFileProcessor, 
                           api_processor: APIProcessor,
                           result_manager: StreamingResultManager,
                           files_chunk: List[Path],
                           pbar: tqdm):
    """Process a chunk of files"""
    
    # Process each file in the chunk
    tasks = []
    for file_path in files_chunk:
        case_id = file_path.name
        
        # Skip if already processed
        if case_id in result_manager.progress:
            status = result_manager.progress[case_id].get("status")
            if status == "success":
                metrics.skipped += 1
                pbar.update(1)
                continue
        
        # Create processing task
        task = asyncio.create_task(process_single_file(
            file_processor, api_processor, result_manager, file_path, case_id
        ))
        tasks.append(task)
    
    # Wait for all tasks in chunk to complete
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
        
    # Update progress bar
    pbar.update(len(files_chunk) - len(tasks))  # Update for skipped files

async def process_single_file(file_processor: StreamingFileProcessor,
                            api_processor: APIProcessor,
                            result_manager: StreamingResultManager,
                            file_path: Path,
                            case_id: str):
    """Process a single file completely"""
    try:
        # Load metadata
        meta = await safe_load_json(file_path, {})
        if not meta:
            await result_manager.add_result(case_id, {}, "Empty metadata")
            return
        
        # Extract PDF text
        text = await file_processor.extract_pdf_text(meta, case_id)
        
        # Prepare case data
        case_data = {
            "case_id": case_id,
            "metadata": meta,
            "text": text
        }
        
        # Process through API
        result, error = await api_processor.process_case(case_data)
        
        # Save result
        await result_manager.add_result(case_id, result or {}, error)
        
    except Exception as e:
        await result_manager.add_result(case_id, {}, f"Processing error: {str(e)[:100]}")

async def print_progress():
    """Print progress periodically"""
    while True:
        await asyncio.sleep(30)  # Every 30 seconds
        elapsed = (datetime.now() - metrics.start_time).total_seconds()
        rate = metrics.processing_rate()
        
        print(f"""
📊 PROGRESS UPDATE:
   Processed: {metrics.total_processed:,} | Success: {metrics.successful:,} | Failed: {metrics.failed:,}
   Rate: {rate:.1f}/min | PDF: {metrics.pdf_extracted:,}✅/{metrics.pdf_failed:,}❌ | API: {metrics.api_calls:,}
   Runtime: {elapsed/60:.1f}min | Memory: {psutil.virtual_memory().used/(1024**3):.1f}GB
""")

async def main():
    print("🚀 Legal Document Processor v3.0 (STREAMING)")
    print("="*60)
    
    # Initialize components
    file_processor = StreamingFileProcessor()
    api_processor = APIProcessor()
    result_manager = StreamingResultManager()
    
    await api_processor.initialize()
    await result_manager.initialize()
    
    try:
        # Start progress reporter
        progress_task = asyncio.create_task(print_progress())
        
        # Stream process files
        files_chunk = []
        total_files = 0
        
        print("🔄 Starting streaming file processing...")
        
        # Create progress bar (we'll update the total as we go)
        pbar = tqdm(desc="Processing", unit="files", dynamic_ncols=True)
        
        async for file_path in file_processor.discover_files_stream():
            files_chunk.append(file_path)
            total_files += 1
            
            # Process chunk when it's full
            if len(files_chunk) >= config.file_chunk_size:
                metrics.current_chunk += 1
                print(f"📦 Processing chunk {metrics.current_chunk} ({len(files_chunk)} files)")
                
                await process_file_chunk(file_processor, api_processor, result_manager, files_chunk, pbar)
                files_chunk = []
                
                # Brief pause between chunks
                await asyncio.sleep(1)
        
        # Process remaining files
        if files_chunk:
            metrics.current_chunk += 1
            print(f"📦 Processing final chunk ({len(files_chunk)} files)")
            await process_file_chunk(file_processor, api_processor, result_manager, files_chunk, pbar)
        
        # Update progress bar total
        pbar.total = total_files
        pbar.refresh()
        
        # Final save
        await result_manager.save_all()
        pbar.close()
        progress_task.cancel()
        
        # Final stats
        elapsed = (datetime.now() - metrics.start_time).total_seconds()
        print(f"""
✅ PROCESSING COMPLETE!
📊 Total Files: {metrics.total_discovered:,}
✅ Successful: {metrics.successful:,} ({metrics.success_rate():.1%})
❌ Failed: {metrics.failed:,}
⏭️ Skipped: {metrics.skipped:,}
📄 PDFs: {metrics.pdf_extracted:,}✅/{metrics.pdf_failed:,}❌
🔗 API Calls: {metrics.api_calls:,}
⏱️ Runtime: {elapsed/60:.1f} minutes
📈 Rate: {metrics.processing_rate():.1f} files/minute

📊 Final Results:
   Civil Cases: {len(result_manager.civil_cases):,}
   Criminal Cases: {len(result_manager.criminal_cases):,}
""")
        
    finally:
        await api_processor.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()