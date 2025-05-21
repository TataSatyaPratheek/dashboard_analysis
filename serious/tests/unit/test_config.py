# tests/unit/test_config.py
import pytest
import yaml
import os

# Add src to Python path
import sys
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root_dir)

from src.utils.config import Config, _deep_update, load_config

# --- Test Config class ---
def test_config_class_initialization():
    data = {"key1": "value1", "nested": {"key2": 123}}
    cfg = Config(data)
    assert cfg.key1 == "value1"
    assert isinstance(cfg.nested, Config)
    assert cfg.nested.key2 == 123

def test_config_class_attribute_access():
    cfg = Config({"a": 1})
    assert cfg.a == 1
    with pytest.raises(AttributeError):
        _ = cfg.b

# --- Test _deep_update ---
def test_deep_update_new_keys():
    base = {"a": 1}
    update = {"b": 2}
    _deep_update(base, update)
    assert base == {"a": 1, "b": 2}

def test_deep_update_override_keys():
    base = {"a": 1, "b": 2}
    update = {"b": 3, "c": 4}
    _deep_update(base, update)
    assert base == {"a": 1, "b": 3, "c": 4}

def test_deep_update_nested_dicts():
    base = {"a": 1, "b": {"x": 10, "y": 20}}
    update = {"b": {"y": 25, "z": 30}, "c": 3}
    _deep_update(base, update)
    assert base == {"a": 1, "b": {"x": 10, "y": 25, "z": 30}, "c": 3}

def test_deep_update_update_value_to_dict():
    base = {"a": 1}
    update = {"a": {"nested": True}}
    _deep_update(base, update)
    assert base == {"a": {"nested": True}}

def test_deep_update_base_value_to_dict():
    base = {"a": {"nested": True}}
    update = {"a": 1}
    _deep_update(base, update)
    assert base == {"a": 1}

# --- Test load_config ---
@pytest.fixture
def temp_config_file(tmp_path):
    def _create_config(filename, content):
        file_path = tmp_path / filename
        with open(file_path, "w") as f:
            yaml.dump(content, f)
        return str(file_path)
    return _create_config

def test_load_config_defaults_only():
    # Test with no config files provided, relying on hardcoded defaults
    # This part depends on how load_config is structured to handle no paths
    # Assuming load_config() with no args uses its internal defaults
    cfg_obj = load_config() # Will print warning "Config file ... not found"
    assert cfg_obj.app_name == "SEO Dashboard Analyzer" # Check a default
    assert cfg_obj.faiss_indexer.k_search == 10 # Check nested default

def test_load_config_single_file(temp_config_file):
    content = {"app_name": "Test App", "faiss_indexer": {"k_search": 7}}
    config_path = temp_config_file("test_cfg1.yaml", content)
    
    cfg_obj = load_config(config_path)
    assert cfg_obj.app_name == "Test App"
    assert cfg_obj.faiss_indexer.k_search == 7
    assert cfg_obj.pdf_processor.chunk_size == 1000 # From defaults

def test_load_config_multiple_files_override(temp_config_file):
    base_content = {"app_name": "Base App", "feature_x": True, "faiss_indexer": {"m": 8}}
    override_content = {"app_name": "Override App", "feature_y": False, "faiss_indexer": {"m": 16, "nbits": 6}}
    
    base_path = temp_config_file("base_test.yaml", base_content)
    override_path = temp_config_file("override_test.yaml", override_content)
    
    cfg_obj = load_config(base_path, override_path)
    assert cfg_obj.app_name == "Override App" # Overridden
    assert cfg_obj.feature_x is True          # From base
    assert cfg_obj.feature_y is False         # From override
    assert cfg_obj.faiss_indexer.m == 16      # Overridden nested
    assert cfg_obj.faiss_indexer.nbits == 6   # New nested in override
    assert cfg_obj.pdf_processor.chunk_size == 1000 # From defaults

def test_load_config_file_not_found(capsys):
    cfg_obj = load_config("nonexistent_config.yaml")
    captured = capsys.readouterr()
    assert "Warning: Config file nonexistent_config.yaml not found. Skipping." in captured.out
    assert cfg_obj.app_name == "SEO Dashboard Analyzer" # Should use defaults

def test_load_config_empty_yaml_file(temp_config_file, capsys):
    config_path = temp_config_file("empty.yaml", {}) # Empty dict for valid YAML
    cfg_obj = load_config(config_path) # Should load defaults after "empty" file
    captured = capsys.readouterr()
    assert f"Successfully loaded and merged config: {config_path}" in captured.out
    assert cfg_obj.app_name == "SEO Dashboard Analyzer"

def test_load_config_malformed_yaml_file(temp_config_file, capsys):
    config_path = temp_config_file("malformed.yaml", "key: value: another") # Invalid YAML
    cfg_obj = load_config(config_path)
    captured = capsys.readouterr()
    assert f"Error loading or merging config file {config_path}" in captured.out
    assert cfg_obj.app_name == "SEO Dashboard Analyzer" # Defaults
