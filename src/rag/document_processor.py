import os
import re
from typing import List, Dict, Tuple, Optional
import tiktoken
from pathlib import Path


try:
    import fitz  
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


class DocumentProcessor:
    
    def __init__(
        self,
        chunk_size: int = 250,
        chunk_overlap: int = 50,
        encoding: str = "cl100k_base"  
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(encoding)
        
        print(f"✓ Document Processor initialized")
        print(f"  Chunk size: {chunk_size} tokens")
        print(f"  Overlap: {chunk_overlap} tokens")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        text = ""
        metadata = {
            "file_name": os.path.basename(pdf_path),
            "file_path": pdf_path,
            "n_pages": 0
        }
        
        
        if HAS_PYMUPDF:
            try:
                doc = fitz.open(pdf_path)
                metadata["n_pages"] = len(doc)
                
                for page in doc:
                    text += page.get_text()
                
                doc.close()
                print(f"  Extracted with PyMuPDF: {metadata['n_pages']} pages")
                return text, metadata
            except Exception as e:
                print(f"  PyMuPDF failed: {e}, trying pdfplumber...")
        
        
        if HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    metadata["n_pages"] = len(pdf.pages)
                    
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                
                print(f"  Extracted with pdfplumber: {metadata['n_pages']} pages")
                return text, metadata
            except Exception as e:
                raise RuntimeError(f"Failed to extract PDF with all methods: {e}")
        
        raise RuntimeError("No PDF library available. Install PyMuPDF or pdfplumber.")
    
    def chunk_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            
            start_idx += self.chunk_size - self.chunk_overlap
            
            
            if end_idx >= len(tokens):
                break
        
        return chunks
    
    def extract_metadata_from_text(self, text: str) -> Dict:
        metadata = {}
        
        
        temp_patterns = [
            r'(\d+)\s*[°]?C',  
            r'(\d+)\s*degrees?\s*celsius',
            r'temperature.*?(\d+)\s*[°]?C',
        ]
        temps = []
        for pattern in temp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            temps.extend([float(m) for m in matches])
        
        if temps:
            metadata["temperature_C"] = {
                "min": min(temps),
                "max": max(temps),
                "values": list(set(temps))
            }
        
        
        pressure_patterns = [
            r'(\d+\.?\d*)\s*atm',
            r'(\d+\.?\d*)\s*bar',
            r'pressure.*?(\d+\.?\d*)\s*atm',
        ]
        pressures = []
        for pattern in pressure_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            pressures.extend([float(m) for m in matches])
        
        if pressures:
            metadata["pressure_atm"] = {
                "min": min(pressures),
                "max": max(pressures),
                "values": list(set(pressures))
            }
        
        
        catalyst_keywords = [
            "Pt/C", "platinum carbon", "Pt catalyst",
            "carbon felt", "graphite", "carbon fiber"
        ]
        found_catalysts = []
        for keyword in catalyst_keywords:
            if re.search(keyword, text, re.IGNORECASE):
                found_catalysts.append(keyword)
        
        if found_catalysts:
            metadata["catalysts"] = list(set(found_catalysts))
        
        
        i0_patterns = [
            r'i[_\s]?0.*?(\d+\.?\d*[eE][-+]?\d+)',  
            r'exchange\s+current.*?(\d+\.?\d*[eE][-+]?\d+)',
        ]
        i0_values = []
        for pattern in i0_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            i0_values.extend([float(m) for m in matches])
        
        if i0_values:
            metadata["i0_A_cm2"] = {
                "min": min(i0_values),
                "max": max(i0_values),
                "values": list(set(i0_values))
            }
        
        
        if re.search(r'PEMFC|proton.exchange.membrane|PEM fuel cell', text, re.IGNORECASE):
            metadata["system"] = "PEMFC"
        elif re.search(r'VRFB|vanadium.redox|flow battery', text, re.IGNORECASE):
            metadata["system"] = "VRFB"
        elif re.search(r'SOFC|solid.oxide', text, re.IGNORECASE):
            metadata["system"] = "SOFC"
        
        return metadata
    
    def process_pdf(
        self,
        pdf_path: str,
        extract_metadata: bool = True
    ) -> List[Dict]:
        print(f"\nProcessing: {os.path.basename(pdf_path)}")
        
        
        text, file_metadata = self.extract_text_from_pdf(pdf_path)
        
        
        doc_metadata = {}
        if extract_metadata:
            doc_metadata = self.extract_metadata_from_text(text)
        
        
        chunks = self.chunk_text(text)
        print(f"  Created {len(chunks)} chunks")
        
        
        chunk_docs = []
        for idx, chunk in enumerate(chunks):
            chunk_metadata = {
                **file_metadata,
                **doc_metadata,
                "chunk_id": idx,
                "n_tokens": len(self.tokenizer.encode(chunk))
            }
            
            chunk_docs.append({
                "text": chunk,
                "metadata": chunk_metadata,
                "doc_id": f"{Path(pdf_path).stem}_chunk_{idx}"
            })
        
        return chunk_docs
    
    def process_directory(
        self,
        directory: str,
        pattern: str = "*.pdf",
        extract_metadata: bool = True
    ) -> List[Dict]:
        pdf_files = list(Path(directory).glob(pattern))
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found in {directory}")
        
        print(f"Found {len(pdf_files)} PDF files in {directory}")
        
        all_chunks = []
        for pdf_path in pdf_files:
            try:
                chunks = self.process_pdf(str(pdf_path), extract_metadata)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"  ✗ Error processing {pdf_path.name}: {e}")
                continue
        
        print(f"\n✓ Processed {len(pdf_files)} PDFs → {len(all_chunks)} chunks")
        
        return all_chunks


def main():
    print("Testing Document Processor...")
    
    
    processor = DocumentProcessor(
        chunk_size=250,
        chunk_overlap=50
    )
    
    
    sample_text = """
    Proton Exchange Membrane Fuel Cell (PEMFC) Performance Study
    
    Abstract: This paper investigates PEMFC performance at various temperatures.
    The fuel cell operates at 80°C with a platinum/carbon (Pt/C) catalyst.
    The exchange current density i0 was measured at 1.5e-7 A/cm² at standard
    pressure of 1 atm. Operating conditions ranged from 60°C to 90°C.
    
    Results show that voltage efficiency decreases at higher current densities
    due to activation losses. The membrane resistance was 0.15 Ω·cm².
    """
    
    
    print("\nTesting text chunking...")
    chunks = processor.chunk_text(sample_text)
    print(f"  Created {len(chunks)} chunks from sample text")
    
    
    print("\nTesting metadata extraction...")
    metadata = processor.extract_metadata_from_text(sample_text)
    print(f"  Extracted metadata: {metadata}")
    
    
    literature_dir = "data/literature"
    if os.path.exists(literature_dir):
        try:
            docs = processor.process_directory(literature_dir)
            print(f"\n✓ Processed {len(docs)} chunks from PDFs")
        except FileNotFoundError:
            print(f"\nNo PDFs found in {literature_dir}")
            print("  Place PDF files in data/literature/ directory")
    else:
        print(f"\n📁 Create {literature_dir}/ and add PDFs to process them")
    
    print("\n✓ Document Processor tests complete")


if __name__ == "__main__":
    main()
