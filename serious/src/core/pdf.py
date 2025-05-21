# src/core/pdf.py

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        """
        Initializes the PDFProcessor with a text splitter.
        Args:
            chunk_size (int): The maximum size of each text chunk.
            chunk_overlap (int): The overlap between consecutive chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
    def process(self, file_path_or_buffer):
        """
        Processes a PDF file, extracts text, and splits it into chunks.
        M1-optimized considerations involve efficient library usage.
        Args:
            file_path_or_buffer: Path to the PDF file or a file-like object.
        Returns:
            list: A list of text chunks.
        """
        try:
            pdf = PdfReader(file_path_or_buffer)
            extracted_texts = []
            for page in pdf.pages:
                page_text = page.extract_text() # Call only once
                if page_text: # Check the result
                    extracted_texts.append(page_text)
            text = '\n'.join(extracted_texts)
        except Exception as e:
            # Handle potential pypdf errors (e.g., encrypted PDF, malformed file)
            print(f"Error reading PDF: {e}")
            return []
            
        # Memory-conscious chunking
        return self.splitter.split_text(text)
