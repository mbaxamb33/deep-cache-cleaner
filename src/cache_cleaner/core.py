
# File: src/cache_cleaner/core.py
"""Core cache cleaning functionality."""

import gc
import os
import shutil
import tempfile
from typing import Dict, Optional, List
import warnings

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not found. GPU cache clearing will be disabled.")

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import sentence_transformers
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class CacheCleaner:
    """Main cache cleaner class with configurable options."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize CacheCleaner.
        
        Args:
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        
    def _print(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def clear_gpu_cache(self) -> bool:
        """Clear PyTorch GPU cache."""
        if not HAS_TORCH or not torch.cuda.is_available():
            self._print("  ⚠️  PyTorch CUDA not available")
            return False
            
        self._print("  ✓ Clearing CUDA cache...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        return True
    
    def clear_python_cache(self) -> int:
        """Clear Python garbage collection cache."""
        self._print("  ✓ Running garbage collection...")
        total_collected = 0
        for i in range(3):
            collected = gc.collect()
            total_collected += collected
            if i == 0 and self.verbose:
                self._print(f"    - Collected {collected} objects")
        return total_collected
    
    def clear_huggingface_cache(self) -> Optional[float]:
        """Clear HuggingFace transformers cache."""
        if not HAS_TRANSFORMERS:
            self._print("  ⚠️  HuggingFace transformers not available")
            return None
            
        try:
            self._print("  ✓ Clearing HuggingFace cache...")
            
            # Clear in-memory cache
            if hasattr(AutoModel, '_modules'):
                AutoModel._modules.clear()
            if hasattr(AutoTokenizer, '_tokenizers'):
                AutoTokenizer._tokenizers.clear()
            
            # Check cache directory size
            hf_cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
            if os.path.exists(hf_cache_dir):
                cache_size = self._get_directory_size(hf_cache_dir)
                self._print(f"    - HF cache size: {cache_size:.2f} GB")
                return cache_size
                
        except Exception as e:
            self._print(f"    - Warning: Could not clear HF cache: {e}")
        
        return None
    
    def clear_sentence_transformers_cache(self) -> Optional[float]:
        """Clear sentence-transformers cache."""
        try:
            self._print("  ✓ Checking sentence-transformers cache...")
            st_cache_dir = os.path.expanduser("~/.cache/sentence_transformers")
            if os.path.exists(st_cache_dir):
                cache_size = self._get_directory_size(st_cache_dir)
                self._print(f"    - ST cache size: {cache_size:.2f} GB")
                return cache_size
        except Exception as e:
            self._print(f"    - Warning: Could not check ST cache: {e}")
        
        return None
    
    def clear_python_pycache(self) -> bool:
        """Clear Python __pycache__ directories."""
        try:
            self._print("  ✓ Clearing Python cache...")
            import sys
            for module_name in list(sys.modules.keys()):
                module = sys.modules[module_name]
                if hasattr(module, '__cached__'):
                    delattr(module, '__cached__')
            return True
        except Exception as e:
            self._print(f"    - Warning: Could not clear Python cache: {e}")
            return False
    
    def check_temp_files(self) -> int:
        """Check temporary files count."""
        try:
            self._print("  ✓ Checking temporary files...")
            temp_dir = tempfile.gettempdir()
            temp_files = [f for f in os.listdir(temp_dir) if f.startswith('tmp')]
            count = len(temp_files) 
            self._print(f"    - Found {count} temp files")
            return count
        except Exception as e:
            self._print(f"    - Warning: Could not check temp files: {e}")
            return 0
    
    def display_gpu_status(self):
        """Display current GPU memory status."""
        if not HAS_TORCH or not torch.cuda.is_available():
            return
            
        self._print("  📊 Current GPU memory status:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            self._print(f"    - GPU {i}: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {total:.2f}GB total")
    
    def _get_directory_size(self, path: str) -> float:
        """Get directory size in GB."""
        return sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(path)
            for filename in filenames
        ) / (1024**3)
        
    def clear_all_caches(self):
        """Comprehensive cache clearing function."""
        self._print("🧹 Starting comprehensive cache cleanup...")
        
        # Clear GPU cache
        self.clear_gpu_cache()
        
        # Clear Python cache
        self.clear_python_cache()
        
        # Clear library caches
        self.clear_huggingface_cache()
        self.clear_sentence_transformers_cache()
        self.clear_python_pycache()
        
        # Check temp files
        self.check_temp_files()
        
        # Display GPU status
        self.display_gpu_status()
        
        self._print("🎉 Cache cleanup completed!")


def clear_all_caches(verbose: bool = True):
    """Comprehensive cache clearing function."""
    cleaner = CacheCleaner(verbose=verbose)
    cleaner.clear_all_caches()


def aggressive_cache_clear(auto_confirm: bool = False, verbose: bool = True):
    """Nuclear option - clear everything possible."""
    cleaner = CacheCleaner(verbose=verbose)
    
    if verbose:
        print("💣 AGGRESSIVE CACHE CLEARING - This will delete downloaded models!")
    
    if not auto_confirm:
        response = input("Are you sure? This will delete all cached models (y/N): ")
        if response.lower() != 'y':
            if verbose:
                print("Cancelled.")
            return
    
    if verbose:
        print("🔥 Starting aggressive cleanup...")
    
    # Standard cleanup first
    cleaner.clear_all_caches()
    
    # Delete cache directories
    cache_dirs = [
        ("HuggingFace", "~/.cache/huggingface"),
        ("Sentence Transformers", "~/.cache/sentence_transformers"),
        ("Torch", "~/.cache/torch"),
    ]
    
    for name, path in cache_dirs:
        try:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                if verbose:
                    print(f"  💥 Deleting {name} cache: {expanded_path}")
                shutil.rmtree(expanded_path)
        except Exception as e:
            if verbose:
                print(f"    - Error deleting {name} cache: {e}")
    
    # Clear environment variables related to caches
    cache_env_vars = ['HF_HOME', 'HF_HUB', 'TRANSFORMERS_CACHE', 'SENTENCE_TRANSFORMERS_HOME']
    for var in cache_env_vars:
        if var in os.environ:
            cache_path = os.environ[var]
            if os.path.exists(cache_path):
                if verbose:
                    print(f"  💥 Deleting {var} cache: {cache_path}")
                try:
                    shutil.rmtree(cache_path)
                except Exception as e:
                    if verbose:
                        print(f"    - Error deleting {var} cache: {e}")
    
    if verbose:
        print("💀 Aggressive cleanup completed! You'll need to re-download models.")


def check_cache_sizes(verbose: bool = True) -> Dict[str, float]:
    """Check sizes of various caches."""
    if verbose:
        print("📊 Cache Size Report:")
    
    cache_dirs = {
        "HuggingFace": "~/.cache/huggingface",
        "Sentence Transformers": "~/.cache/sentence_transformers", 
        "Torch": "~/.cache/torch",
        "Pip": "~/.cache/pip",
    }
    
    sizes = {}
    total_size = 0
    
    for name, path in cache_dirs.items():
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(expanded_path)
                for filename in filenames
            ) / (1024**3)  # GB
            sizes[name] = size
            total_size += size
            if verbose:
                print(f"  {name}: {size:.2f} GB ({expanded_path})")
        else:
            sizes[name] = 0
            if verbose:
                print(f"  {name}: Not found ({expanded_path})")
    
    sizes["Total"] = total_size
    
    if verbose:
        print(f"\n💾 Total cache size: {total_size:.2f} GB")
        
        # GPU memory info
        if HAS_TORCH and torch.cuda.is_available():
            print(f"\n🖥️  GPU Memory:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                free = total - allocated
                print(f"  GPU {i}: {allocated:.2f}GB used, {free:.2f}GB free, {total:.2f}GB total")
    
    return sizes

# File: src/cache_cleaner/cli.py
"""Command line interface for cache cleaner."""

import argparse
import sys
from .core import clear_all_caches, aggressive_cache_clear, check_cache_sizes


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive cache clearing utility for PyTorch, HuggingFace, and other ML libraries"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear all caches')
    clear_parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    
    # Aggressive clear command
    aggressive_parser = subparsers.add_parser('aggressive', help='Aggressively clear all caches (deletes downloaded models)')
    aggressive_parser.add_argument('--yes', '-y', action='store_true', help='Auto-confirm deletion')
    aggressive_parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check cache sizes')
    check_parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    verbose = not args.quiet
    
    try:
        if args.command == 'clear':
            clear_all_caches(verbose=verbose)
        elif args.command == 'aggressive':
            aggressive_cache_clear(auto_confirm=args.yes, verbose=verbose)
        elif args.command == 'check':
            check_cache_sizes(verbose=verbose)
    except KeyboardInterrupt:
        if verbose:
            print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        if verbose:
            print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()