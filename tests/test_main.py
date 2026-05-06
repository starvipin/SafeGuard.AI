import pytest
import sys
import os
from unittest.mock import patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import main

class TestMain:
    """Test main module functionality"""

    @patch('builtins.print')
    def test_main_function(self, mock_print):
        """Test the main function."""
        main()

        # Check that print was called with expected message
        mock_print.assert_called_once_with("Hello from safeguard-ai!")