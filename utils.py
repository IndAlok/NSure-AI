# utils.py
"""
This module contains utility functions for the RAG pipeline, primarily for
fetching and parsing documents from external sources.
"""
import requests
import fitz  # PyMuPDF library

def get_pdf_text_from_url(url: str) -> str:
    """
    Downloads a PDF from a given URL, extracts its text content in memory,
    and returns it as a single string. This approach is efficient as it
    avoids writing to disk.

    Args:
        url (str): The public URL of the PDF document.

    Returns:
        str: The concatenated text content of all pages in the PDF.
             Returns an empty string if fetching or parsing fails.
    
    Raises:
        requests.exceptions.RequestException: If the URL is invalid or the
                                              network request fails.
        Exception: For errors encountered during PDF parsing.
    """
    print(f"Fetching PDF from URL: {url}")
    try:
        # Use a timeout to prevent hanging on unresponsive servers
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

        # Read PDF content directly from the response bytes
        pdf_bytes = response.content
        
        text_content = ""
        # Open the PDF from bytes using fitz
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            print(f"Successfully opened PDF. It has {len(doc)} pages.")
            # Iterate through each page and extract text
            for page_num, page in enumerate(doc):
                text_content += page.get_text()
            print("Text extraction complete.")
        
        return text_content
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to fetch PDF from URL. {e}")
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred during PDF parsing. {e}")
        raise