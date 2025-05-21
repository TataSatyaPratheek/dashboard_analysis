# src/utils/config.py

import yaml # You would need to add PyYAML to requirements.txt
import os

class Config:
    """ Simple configuration class """
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value)) # Recursively create Config objects for nested dicts
            else:
                setattr(self, key, value)

def _deep_update(base_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_config(*config_paths):
    """
    Loads configuration from multiple YAML files, with later files overriding earlier ones.
    """
    base_config_data = {
        'app_name': "SEO Dashboard Analyzer", # Added default app_name
        'pdf_processor': {'chunk_size': 1000, 'chunk_overlap': 100},
        'embedding_generator': {'model_name': 'all-MiniLM-L6-v2', 'batch_size': 32},
        'faiss_indexer': {'dim': 384, 'm': 8, 'nbits': 8, 'k_search': 10},
        'community_detector': {'n_neighbors': 15, 'use_weights': False},
        'streamlit_ui': {'title': "M1 SEO Analyzer", 'max_upload_size_mb': 200},
        'benchmark': {'num_test_vectors': 10000, 'num_query_vectors': 100, 'pdf_test_file': "data/sample_benchmark.pdf"},
        'openai': {'api_key': "YOUR_OPENAI_API_KEY_HERE", 'model': "gpt-4o-mini", 'temperature': 0.7, 'max_tokens': 250}
    }

    final_config_data = base_config_data.copy()

    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    override_data = yaml.safe_load(f)
                    if override_data: # Ensure file is not empty
                        final_config_data = _deep_update(final_config_data, override_data)
                print(f"Successfully loaded and merged config: {config_path}")
            except Exception as e:
                print(f"Error loading or merging config file {config_path}: {e}. Skipping this file.")
        else:
            print(f"Warning: Config file {config_path} not found. Skipping.")
            
    return Config(final_config_data)

# Example of how it might be used (not part of the file itself):
# if __name__ == '__main__':
#     # Create a dummy configs/base.yaml for testing
#     # import os
#     # if not os.path.exists("configs"): os.makedirs("configs")
#     # with open("configs/base.yaml", "w") as f:
#     #     f.write("faiss_indexer:\n  k_search: 5\n")
#
#     # app_config = load_config() # Old way
#     # print(app_config.faiss_indexer['k_search'])
#     # print(app_config.streamlit_ui['title'])
#     app_config = load_config("configs/base.yaml", "configs/m1_optim.yaml") # New way example
