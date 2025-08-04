import requests
import fitz  # PyMuPDF
import re
from functools import lru_cache
from typing import Optional, List

@lru_cache(maxsize=32)
def get_pdf_text_from_url(url: str) -> Optional[str]:
    """
    Downloads a PDF from a URL and extracts clean text content.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()

        doc = fitz.open(stream=response.content, filetype="pdf")
        
        full_text = []
        for page in doc:
            page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES)
            
            # 1. Normalize all whitespace to single spaces
            cleaned_text = re.sub(r'\s+', ' ', page_text).strip()
            
            # 2. Re-join words that are hyphenated across lines
            cleaned_text = re.sub(r'-\s+', '', cleaned_text)
            
            # 3. Targeted removal of known, repetitive footers. This is much safer.
            footer_patterns = [
                r"National Insurance Co\. Ltd\.\s+Premises No\. 18-0374, Plot no\. CBD-81,\s+New Town, Kolkata - \d+",
                r"Page \d+ of \d+",
                r"National Parivar Mediclaim Plus Policy\s+UIN: \w+"
            ]
            for pattern in footer_patterns:
                cleaned_text = re.sub(pattern, '', cleaned_text)

            full_text.append(cleaned_text)

        doc.close()
        final_text = ' '.join(full_text)
        return final_text if len(final_text) > 100 else None

    except requests.exceptions.RequestException as e:
        print(f"PDF download error from URL {url}: {e}")
        return None
    except Exception as e:
        print(f"PDF text extraction error for URL {url}: {e}")
        return None

def semantic_chunker(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    if not sentences:
        return []

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            overlap_text = " ".join(current_chunk.split()[-int(overlap/5):])
            current_chunk = overlap_text + " " + sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk) > 100]
