#!/usr/bin/env python3
"""
Simple enhanced photo culler - just adds better technical analysis
"""
import click
from pathlib import Path
from batch import BatchCuller
from models import ProcessingMode
import json
import logging
import csv
from datetime import datetime


def setup_logging(verbose=False):
    """Setup logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


@click.command()
@click.argument('folder', type=click.Path(exists=True))
@click.option('--fast', is_flag=True, help='Use fast CV mode (no ON1 metadata)')
@click.option('--cache-dir', type=click.Path(), help='Directory for thumbnail and results cache')
@click.option('--csv-file', default='photo_culler_results.csv', help='CSV file to append results to')
@click.option('--move-deletes', is_flag=True, help='Move files marked for deletion')
@click.option('--use-ollama', is_flag=True, help='Use Ollama vision model instead of CLIP')
@click.option('--ollama-model', default='llava:13b', help='Ollama model to use')
@click.option('--verbose', is_flag=True, help='Show processing details')
@click.option('--detail', is_flag=True, help='Show individual file results during processing')
@click.option('--extensions', default='nef,cr2,arw,jpg,jpeg', help='File extensions to process')
@click.option('--override', is_flag=True, help='Override ALL existing keywords and descriptions')
@click.option('--learning', is_flag=True, help='Enable adaptive learning mode')
@click.option('--concurrent', default=2, help='Number of concurrent Ollama instances')
def simple_cull(folder, fast, cache_dir, csv_file, move_deletes, use_ollama, ollama_model, 
               verbose, detail, extensions, override, learning, concurrent):
    """
    Simple enhanced photo culler - your original workflow with better quality detection
    
    Just adds enhanced technical analysis for better blur/noise/aberration detection.
    No complicated features, just better results.
    """
    
    logger = setup_logging(verbose)
    
    # Your original logic, exactly the same
    mode = ProcessingMode.FAST if fast else ProcessingMode.ACCURATE
    
    folder = Path(folder)
    cache = Path(cache_dir) if cache_dir else None
    csv_path = Path(csv_file)
    
    ext_list = ['.' + ext.strip().lstrip('.') for ext in extensions.split(',')]
    
    print(f"📁 Folder: {folder}")
    print(f"🔧 Mode: {mode.value}" + (f" (Ollama: {ollama_model})" if use_ollama and not fast else ""))
    print(f"📊 CSV: {csv_path}")
    print(f"📝 ON1 Metadata: {'Yes' if not fast else 'No (fast mode)'}")
    print(f"🚀 Concurrent: {concurrent} Ollama instances")
    print("=" * 60)
    
    # Initialize culler (same as before)
    try:
        culler = BatchCuller(
            cache_dir=cache,
            mode=mode,
            max_workers=4,
            batch_size=8,
            use_ollama=use_ollama,
            ollama_model=ollama_model,
            learning_enabled=learning
        )
    except Exception as e:
        print(f"❌ Failed to initialize culler: {e}")
        return
    
    # Find files (same as before)
    files = []
    for ext in ext_list:
        files.extend(folder.glob(f'*{ext}'))
        files.extend(folder.glob(f'*{ext.upper()}'))
    
    files = sorted(files)
    
    if not files:
        print(f"❌ No files found with extensions: {ext_list}")
        return
    
    print(f"🔍 Found {len(files)} files to process\n")
    
    # Use your existing concurrent processing from culler_on1.py
    from culler_on1 import process_concurrent
    
    results, on1_updated = process_concurrent(
        files, culler, concurrent, 1, detail, fast, override, csv_path
    )
    
    # Save session (same as before)
    try:
        culler.save_session()
    except Exception as e:
        print(f"⚠️  Warning: Failed to save session data: {e}")
    
    # Summary (same as your original)
    print("=" * 60)
    print("🏁 CULLING COMPLETE")
    print("=" * 60)
    
    total = len(files)
    for decision, items in results.items():
        if items:
            percentage = len(items) / total * 100
            print(f"{decision:>8}: {len(items):>3} files ({percentage:4.1f}%)")
    
    if move_deletes and results['Delete']:
        trash_dir = folder / '_culled_deletes'
        trash_dir.mkdir(exist_ok=True)
        
        moved = 0
        for result in results['Delete']:
            if result.confidence > 0.7:
                try:
                    dest = trash_dir / result.filepath.name
                    result.filepath.rename(dest)
                    moved += 1
                except Exception as e:
                    print(f"Failed to move {result.filepath.name}: {e}")
        
        print(f"📦 Moved {moved} high-confidence deletions to {trash_dir}")
    
    print(f"📊 Results saved to: {csv_path}")
    if not fast and on1_updated > 0:
        print(f"📝 Updated {on1_updated} ON1 metadata files")


if __name__ == '__main__':
    simple_cull()