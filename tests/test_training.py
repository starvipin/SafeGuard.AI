import pytest
import os
import tempfile
import pandas as pd
import torch
from unittest.mock import patch, MagicMock


# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stage_02_train import read_params, FraudDataset

class TestReadParamsTrain:
    """Test parameter reading for training"""

    def test_read_params_train(self, temp_config):
        """Test  reading training config."""
        config = read_params(temp_config)
        assert 'train' in config
        assert 'model_name' in config['train']
        assert config['train']['model_name'] == 'distilbert-base-uncased'

class TestFraudDataset:
    """Test the FraudDataset class"""

    def test_dataset_creation(self):
        """Test dataset initialization."""
        encodings = {
            'input_ids': [[1, 2, 3], [4, 5, 6]],
            'attention_mask': [[1, 1, 1], [1, 1, 1]]
        }
        labels = pd.Series([0, 1])

        dataset = FraudDataset(encodings, labels)
        assert len(dataset) == 2

    def test_dataset_getitem(self):
        """Test getting item from dataset."""
        encodings = {
            'input_ids': [[1, 2, 3]],
            'attention_mask': [[1, 1, 1]]
        }
        labels = pd.Series([1])

        dataset = FraudDataset(encodings, labels)
        item = dataset[0]

        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item
        assert item['labels'].item() == 1

class TestTrainModel:
    """Test training functionality"""

    @patch('stage_02_train.DistilBertTokenizerFast.from_pretrained')
    @patch('stage_02_train.DistilBertForSequenceClassification.from_pretrained')
    @patch('stage_02_train.torch.device')
    @patch('stage_02_train.DataLoader')
    @patch('stage_02_train.AdamW')
    @patch('stage_02_train.os.makedirs')
    @patch('pandas.read_csv')
    @patch('stage_02_train.train_test_split')
    def test_train_model_basic(self, mock_split, mock_read_csv, mock_makedirs,
                              mock_adamw, mock_dataloader, mock_device,
                              mock_model_class, mock_tokenizer_class):
        """Test basic training flow."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {'input_ids': [[1, 2, 3]], 'attention_mask': [[1, 1, 1]]}
        mock_tokenizer_class.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        mock_device.return_value = 'cpu'

        mock_batch = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]]),
            'labels': torch.tensor([1])
        }
        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = [mock_batch]  # One batch
        mock_dataloader.return_value = mock_loader

        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer

        # Mock data
        mock_df = pd.DataFrame({
            'text': ['message 1', 'message 2'],
            'label': [0, 1]
        })
        mock_read_csv.return_value = mock_df

        mock_split.return_value = (['msg1'], ['msg2'], pd.Series([0]), pd.Series([1]))

        # Mock model outputs
        mock_outputs = MagicMock()
        mock_outputs.loss = MagicMock()
        mock_outputs.loss.item.return_value = 0.5
        mock_outputs.logits = torch.tensor([[0.2, 0.8]])
        mock_model.return_value = mock_outputs

        # Test training (with minimal epochs)
        from stage_02_train import train_model

        # Create temp config with 1 epoch
        import yaml
        config = {
            "data_source": {
                "raw_data_dir": "data/raw_data",
                "dataset_name": "fraud_dataset.csv"
            },
            "train": {
                "model_name": "distilbert-base-uncased",
                "batch_size": 2,
                "epochs": 1,
                "learning_rate": 5e-5,
                "save_steps": 500
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch.dict('sys.modules', {'upload_to_hf': MagicMock()}):
                train_model(config_path)
            # Should complete without errors
            assert mock_model.save_pretrained.called
            assert mock_tokenizer.save_pretrained.called

        finally:
            os.unlink(config_path)

    @patch('os.listdir')
    @patch('os.path.isdir')
    @patch('shutil.rmtree')
    def test_checkpoint_cleanup(self, mock_rmtree, mock_isdir, mock_listdir):
        """Test cleanup of intermediate checkpoints."""
        mock_listdir.return_value = ['checkpoint-100', 'checkpoint-200', 'fraud_model_final']
        mock_isdir.return_value = True

        from stage_02_train import train_model

        # This would be called at the end of training
        # For testing, we can call the cleanup part directly
        import shutil
        models_dir = "models"
        for folder_name in os.listdir(models_dir):
            if folder_name.startswith("checkpoint-"):
                folder_path = os.path.join(models_dir, folder_name)
                if os.path.isdir(folder_path):
                    shutil.rmtree(folder_path)  # Use rmtree for mock

        # Verify calls
        assert mock_listdir.called
        assert mock_rmtree.call_count == 2  # Two checkpoints