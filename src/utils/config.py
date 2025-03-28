import yaml
import os

# Use os.path.join for cross-platform compatibility
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "config.yml")

def load_config():
    """Loads the YAML configuration file."""
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config['model']['model_path'] = os.path.join(base_dir, config['model']['model_path'])
    config['model']['tokenizer_path'] = os.path.join(base_dir, config['model']['tokenizer_path'])
    
    return config

# Load config once and make it accessible globally
config = load_config()