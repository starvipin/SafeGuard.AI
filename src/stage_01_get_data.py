import os
import shutil
import yaml

# Function to read the configuration file (params.yaml)
def read_params(config_path):
    with open(config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    try:
        config = read_params(config_path)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found")
        return
    except Exception as e:
        print(f"Error reading config: {e}")
        return
    
    # 1. Extract paths (from params.yaml)
    try:
        source_data_path = config["data_source"]["local_path"]
        raw_data_dir = config["data_source"]["raw_data_dir"]
    except KeyError as e:
        print(f"Error: Missing config key {e}")
        return
    
    # 2. Create data folder in the project (if it doesn't already exist)
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # Target file name
    target_path = os.path.join(raw_data_dir, "dataset.parquet")
    
    # 3. Include data in the Data Pipeline
    if os.path.exists(source_data_path):
        try:
            shutil.copy(source_data_path, target_path)
            print(f"Stage 01 Success: Data successfully ingested to '{target_path}'")
        except Exception as e:
            print(f"Error copying file: {e}")
    else:
        print(f"Error: Source file '{source_data_path}' not found. Did you run the extraction script?")

if __name__ == "__main__":
    # This function will trigger when the script is run
    get_data(config_path="params.yaml")