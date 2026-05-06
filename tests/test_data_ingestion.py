import pytest
import os
import tempfile
import shutil
import pandas as pd
from unittest.mock import patch

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stage_01_get_data import read_params, get_data

class TestReadParams:
    """Test parameter reading functionality"""

    def test_read_params_valid(self, temp_config):
        """Test reading valid config file."""
        config = read_params(temp_config)
        assert isinstance(config, dict)
        assert 'data_source' in config
        assert 'raw_data_dir' in config['data_source']

    def test_read_params_invalid_file(self):
        """Test reading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            read_params("non_existent.yaml")

class TestGetData:
    """Test data ingestion functionality"""

    def setup_method(self):
        """Setup temporary directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.raw_data_dir = os.path.join(self.temp_dir, "data", "raw_data")
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def teardown_method(self):
        """Cleanup temporary directories."""
        shutil.rmtree(self.temp_dir)

    def test_get_data_success(self, temp_config):
        """Test successful data ingestion."""
        # Create sample source data as parquet
        source_path = os.path.join(self.temp_dir, "sample_data.parquet")
        sample_data = pd.DataFrame({
            'text': ['test message 1', 'test message 2'],
            'label': [0, 1]
        })
        sample_data.to_parquet(source_path, index=False)

        # Modify config to use temp paths
        import yaml
        with open(temp_config, 'r') as f:
            config = yaml.safe_load(f)

        config['data_source']['local_path'] = source_path
        config['data_source']['raw_data_dir'] = self.raw_data_dir

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            test_config = f.name

        try:
            # Run get_data
            get_data(test_config)

            # Check if target file exists
            target_path = os.path.join(self.raw_data_dir, "dataset.parquet")
            assert os.path.exists(target_path)

            # Check content
            df = pd.read_parquet(target_path)
            assert len(df) == 2
            assert list(df.columns) == ['text', 'label']

        finally:
            os.unlink(test_config)

    def test_get_data_source_not_found(self, temp_config):
        """Test data ingestion when source file doesn't exist."""
        # Modify config to use non-existent source
        import yaml
        with open(temp_config, 'r') as f:
            config = yaml.safe_load(f)

        config['data_source']['local_path'] = "non_existent.csv"
        config['data_source']['raw_data_dir'] = self.raw_data_dir

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            test_config = f.name

        try:
            # Run get_data - should not raise error but print message
            get_data(test_config)

            # Target file should not exist
            target_path = os.path.join(self.raw_data_dir, "dataset.parquet")
            assert not os.path.exists(target_path)

        finally:
            os.unlink(test_config)

    @patch('os.makedirs')
    def test_get_data_creates_directories(self, mock_makedirs, temp_config):
        """Test that directories are created."""
        get_data(temp_config)
        mock_makedirs.assert_called()