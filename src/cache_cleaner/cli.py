
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
    
    verbose = not getattr(args, 'quiet', False)
    
    try:
        if args.command == 'clear':
            clear_all_caches(verbose=verbose)
        elif args.command == 'aggressive':
            aggressive_cache_clear(auto_confirm=getattr(args, 'yes', False), verbose=verbose)
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