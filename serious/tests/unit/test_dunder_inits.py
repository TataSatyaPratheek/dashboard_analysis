# tests/unit/test_dunder_inits.py

def test_core_imports():
    from src.core import PDFProcessor
    from src.core import EmbeddingGenerator
    from src.core import FAISSIndexer
    from src.core import CommunityDetector
    assert PDFProcessor is not None
    assert EmbeddingGenerator is not None
    assert FAISSIndexer is not None
    assert CommunityDetector is not None

def test_utils_imports():
    from src.utils import load_config
    assert load_config is not None

