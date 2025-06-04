
# File: README.md
# Deep Cache Cleaner

A comprehensive cache clearing utility for PyTorch, HuggingFace, and other ML libraries.

## Features

- 🧹 Clear PyTorch GPU memory cache
- 🤗 Clear HuggingFace transformers cache  
- 📝 Clear sentence-transformers cache
- 🗑️ Clear Python __pycache__ directories
- 📊 Check cache sizes and GPU memory usage
- 💣 Aggressive cleanup mode (deletes downloaded models)
- 🔧 Command-line interface
- 📦 Easy to use Python API

## Installation

```bash
pip install deep-cache-cleaner
```

### Optional Dependencies

For full functionality, install with optional dependencies:

```bash
# For HuggingFace transformers support
pip install deep-cache-cleaner[transformers]

# For sentence-transformers support  
pip install deep-cache-cleaner[sentence-transformers]

# For development
pip install deep-cache-cleaner[dev]
```

## Usage

### Command Line Interface

```bash
# Clear all caches
deep-cache-cleaner clear

# Check cache sizes
deep-cache-cleaner check  

# Aggressive cleanup (deletes downloaded models)
deep-cache-cleaner aggressive

# Auto-confirm aggressive cleanup
deep-cache-cleaner aggressive --yes

# Quiet mode
deep-cache-cleaner clear --quiet
```

### Python API

```python
import cache_cleaner

# Clear all caches
cache_cleaner.clear_all_caches()

# Check cache sizes
sizes = cache_cleaner.check_cache_sizes()
print(f"Total cache size: {sizes['Total']:.2f} GB")

# Aggressive cleanup
cache_cleaner.aggressive_cache_clear()

# Use the class interface for more control
cleaner = cache_cleaner.CacheCleaner(verbose=True)
cleaner.clear_all_caches()
```

## What Gets Cleared

### Standard Cleanup (`clear`)
- PyTorch GPU memory cache
- Python garbage collection
- In-memory HuggingFace model cache
- Python __pycache__ directories
- Reports on cache sizes and temp files

### Aggressive Cleanup (`aggressive`) 
⚠️ **Warning: This deletes downloaded models!**

Everything from standard cleanup, plus:
- HuggingFace cache directory (`~/.cache/huggingface`)
- Sentence-transformers cache (`~/.cache/sentence_transformers`) 
- Torch cache directory (`~/.cache/torch`)
- Cache directories from environment variables

## Requirements

- Python 3.8+
- PyTorch (for GPU cache clearing)
- Optional: transformers, sentence-transformers

## License

MIT License

## Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests.