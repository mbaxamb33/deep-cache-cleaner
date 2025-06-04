
# File: tests/test_cache_cleaner.py
"""Tests for cache cleaner functionality."""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from cache_cleaner.core import CacheCleaner, check_cache_sizes


class TestCacheCleaner:
    """Test the CacheCleaner class."""
    
    def test_init(self):
        """Test CacheCleaner initialization."""
        cleaner = CacheCleaner(verbose=True)
        assert cleaner.verbose is True
        
        cleaner = CacheCleaner(verbose=False)
        assert cleaner.verbose is False
    
    def test_print_verbose(self, capsys):
        """Test verbose printing."""
        cleaner = CacheCleaner(verbose=True)
        cleaner._print("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out
    
    def test_print_quiet(self, capsys):
        """Test quiet mode."""
        cleaner = CacheCleaner(verbose=False)
        cleaner._print("test message")
        captured = capsys.readouterr()
        assert captured.out == ""
    
    @patch('cache_cleaner.core.HAS_TORCH', False)
    def test_clear_gpu_cache_no_torch(self):
        """Test GPU cache clearing without torch."""
        cleaner = CacheCleaner(verbose=False)
        result = cleaner.clear_gpu_cache()
        assert result is False
    
    def test_clear_python_cache(self):
        """Test Python garbage collection."""
        cleaner = CacheCleaner(verbose=False)
        result = cleaner.clear_python_cache()
        assert isinstance(result, int)
        assert result >= 0
    
    def test_get_directory_size(self):
        """Test directory size calculation."""
        cleaner = CacheCleaner()
        
        # Create a temporary directory with a file
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")
            
            size = cleaner._get_directory_size(temp_dir)
            assert size > 0
    
    def test_check_temp_files(self):
        """Test temp files checking."""
        cleaner = CacheCleaner(verbose=False)
        result = cleaner.check_temp_files()
        assert isinstance(result, int)
        assert result >= 0


def test_check_cache_sizes():
    """Test cache size checking function."""
    sizes = check_cache_sizes(verbose=False)
    
    assert isinstance(sizes, dict)
    assert "Total" in sizes
    assert "HuggingFace" in sizes
    assert "Torch" in sizes
    
    # All values should be floats
    for key, value in sizes.items():
        assert isinstance(value, (int, float))
        assert value >= 0