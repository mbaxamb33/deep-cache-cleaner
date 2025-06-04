# File: src/cache_cleaner/__init__.py
"""
Cache Cleaner - Comprehensive cache clearing utility for PyTorch, HuggingFace, and other ML libraries.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import (
    clear_all_caches,
    aggressive_cache_clear,
    check_cache_sizes,
    CacheCleaner,
)

__all__ = [
    "clear_all_caches",
    "aggressive_cache_clear", 
    "check_cache_sizes",
    "CacheCleaner",
]