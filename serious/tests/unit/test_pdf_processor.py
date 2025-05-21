# tests/unit/test_pdf_processor.py

import pytest
import os
from pypdf import PdfWriter, PdfReader # For creating a dummy PDF for testing
from unittest.mock import MagicMock, patch # For mocking

# Add src to Python path
import sys
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root_dir)

from src.core.pdf import PDFProcessor

@pytest.fixture(scope="module")
def dummy_pdf_content_path():
    # Create a temporary dummy PDF for testing
    test_dir = os.path.join(project_root_dir, "tests/test_data")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    pdf_path = os.path.join(test_dir, "dummy_content_test.pdf")
    
    writer = PdfWriter()
    # Page 1
    writer.add_blank_page(width=612, height=792)
    # Adding text via pypdf is tricky; it's about how PdfReader's extract_text works.
    # extract_text mainly gets text from content streams. We can't easily write arbitrary text streams here.
    # For this test, we'll create a PDF that pypdf *can* read something from, even if minimal.
    # A better approach is to have a small, static PDF with known text.

    # Let's try a different way: Save a PDF from a known source or create one with a tool
    # that actually embeds text, then use it here.
    # For now, we rely on PdfReader's behavior with a blank page.
    writer.add_metadata({"/Title": "Page 1 Title"}) # Metadata often not extracted by extract_text

    # Page 2 - another blank page
    writer.add_blank_page(width=612, height=792)
    writer.add_metadata({"/Title": "Page 2 Title"})

    with open(pdf_path, "wb") as f:
        writer.write(f)
    yield pdf_path
    # os.remove(pdf_path) # Clean up

def test_pdf_processor_initialization():
    processor = PDFProcessor(chunk_size=500, chunk_overlap=50)
    assert processor.splitter._chunk_size == 500 # Changed to _chunk_size
    assert processor.splitter._chunk_overlap == 50 # Changed to _chunk_overlap

# We will mock PdfReader to control what extract_text returns
def test_pdf_processor_process_mocked_extraction(mocker):
    processor = PDFProcessor()
    
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "This is page one. It has some text."
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page two is here with more words."
    mock_page3 = MagicMock() # Page that returns None for text
    mock_page3.extract_text.return_value = None
    
    mock_pdf_reader_instance = MagicMock()
    mock_pdf_reader_instance.pages = [mock_page1, mock_page2, mock_page3]
    
    mocker.patch('src.core.pdf.PdfReader', return_value=mock_pdf_reader_instance)
    
    # Use a dummy path, PdfReader is mocked
    chunks = processor.process("dummy_path.pdf")
    
    assert len(chunks) > 0
    full_text = "This is page one. It has some text.\nPage two is here with more words."
    # Check if the splitter got the concatenated text (excluding None page)
    # This depends on the splitter behavior, but we know input text.
    reconstructed_from_chunks_approx = "".join(chunks) # Simplistic check
    assert "page one" in reconstructed_from_chunks_approx
    assert "Page two" in reconstructed_from_chunks_approx
    mock_page1.extract_text.assert_called_once()
    mock_page2.extract_text.assert_called_once()
    mock_page3.extract_text.assert_called_once()


def test_pdf_processor_process_extraction_error(mocker, capsys):
    processor = PDFProcessor()
    
    # Mock PdfReader to raise an exception
    mocker.patch('src.core.pdf.PdfReader', side_effect=Exception("PDF Read Error"))
    
    chunks = processor.process("any_path.pdf")
    assert chunks == [] # Should return empty list on error
    captured = capsys.readouterr()
    assert "Error reading PDF: PDF Read Error" in captured.out

def test_pdf_processor_process_real_dummy_pdf(dummy_pdf_content_path):
    # This test uses the actual dummy PDF. pypdf's extract_text on blank pages
    # might return empty strings or specific minimal content.
    processor = PDFProcessor()
    chunks = processor.process(dummy_pdf_content_path)
    assert isinstance(chunks, list)
    # For blank pages, extract_text often returns '\n' or empty string.
    # If chunks = ['', ''], then splitter might make it just [''] or []
    # This test ensures it runs without error and returns a list.
    # If your dummy_pdf_content_path actually has real text, this test becomes more robust.
    # For now, we are mostly testing the flow.
    # For better test, use a PDF with known text content.

def test_pdf_processor_process_nonexistent_pdf(capsys): # Already somewhat covered
    processor = PDFProcessor()
    # The PdfReader in pypdf will raise FileNotFoundError
    # The try-except in PDFProcessor.process catches this
    chunks = processor.process("nonexistent_for_sure.pdf")
    assert chunks == []
    captured = capsys.readouterr()
    # Check that our specific error message for PdfReader failure is printed
    assert "Error reading PDF:" in captured.out # The actual error might vary

def test_pdf_processor_chunking_logic(): # Already good
    processor = PDFProcessor(chunk_size=10, chunk_overlap=2)
    long_text = "This is a test sentence for chunking."
    chunks = processor.splitter.split_text(long_text)
    assert len(chunks) > 1
    if len(chunks) > 1:
      assert chunks[0] != long_text
