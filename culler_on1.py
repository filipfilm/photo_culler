#!/usr/bin/env python3
"""
All-in-one photo culler for ON1 Photo RAW - analyze, ON1 metadata, CSV, results
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


def read_on1_metadata(on1_file):
    """Read existing ON1 metadata file"""
    if not on1_file.exists():
        return None
    
    try:
        with open(on1_file, 'r') as f:
            return json.load(f)
    except:
        return None


def write_on1_metadata(filepath, decision, metrics, confidence, issues, override=False):
    """Write to ON1 .on1 sidecar file preserving existing metadata"""
    on1_file = filepath.with_suffix('.on1')
    
    # Read existing data or create new structure
    data = read_on1_metadata(on1_file)
    if not data:
        # Create a new ON1 structure
        data = {
            "version": "2.0",
            "photos": {
                str(filepath.name): {
                    "metadata": {}
                }
            }
        }
    
    # Find or create the photo entry
    photos = data.get('photos', {})
    
    # Look for existing photo entry by filename
    photo_id = str(filepath.name)
    if photo_id not in photos:
        # Try to find by any existing key (sometimes ON1 uses different naming)
        if photos:
            photo_id = list(photos.keys())[0]
        else:
            # Create new photo entry
            photos[photo_id] = {"metadata": {}}
    
    photo_data = photos[photo_id]
    
    # Get or create metadata section
    metadata = photo_data.setdefault('metadata', {})
    
    # Handle existing keywords based on override mode
    existing_keywords = metadata.get('Keywords', [])
    
    if override:
        # Override mode: only keep non-culler keywords, but replace everything
        existing_keywords = [kw for kw in existing_keywords 
                            if not kw.startswith('PhotoCuller:') 
                            and not kw.startswith('CullerConfidence:')
                            and not kw.startswith('CullerIssues:')
                            and not kw.startswith('CullerAnalysis:')
                            and not kw.startswith('CullerSuggestedRating:')
                            and not kw.startswith('AI:')]
        # In override mode, clear existing keywords and start fresh
        existing_keywords = []
    else:
        # Preserve mode: only remove old culler data, keep user keywords
        existing_keywords = [kw for kw in existing_keywords 
                            if not kw.startswith('PhotoCuller:') 
                            and not kw.startswith('CullerConfidence:')
                            and not kw.startswith('CullerIssues:')
                            and not kw.startswith('CullerAnalysis:')
                            and not kw.startswith('CullerSuggestedRating:')
                            and not kw.startswith('AI:')]
    
    # Add new culler keywords
    issues_str = ', '.join(issues) if issues else 'none'
    culler_keywords = [
        f'PhotoCuller:{decision}',
        f'CullerConfidence:{confidence:.2f}',
        f'CullerIssues:{issues_str}'
    ]
    
    # Add clean AI-generated keywords (no prefix, they're already good)
    if hasattr(metrics, 'keywords') and metrics.keywords:
        existing_keywords.extend(metrics.keywords[:8])  # Add directly to keywords
    
    # Handle description based on override mode
    if hasattr(metrics, 'description') and metrics.description:
        if override or 'Description' not in metadata:
            # Override existing description or add if none exists
            metadata['Description'] = metrics.description
        # In preserve mode, don't overwrite existing descriptions
    
    # Only suggest rating if none exists (rating 0 means no rating in ON1)
    existing_rating = metadata.get('Rating', 0)
    if existing_rating == 0:
        # Suggest rating based on overall score
        if metrics.overall_quality >= 0.8:
            suggested_rating = 5
        elif metrics.overall_quality >= 0.6:
            suggested_rating = 4
        elif metrics.overall_quality >= 0.4:
            suggested_rating = 3
        elif metrics.overall_quality >= 0.2:
            suggested_rating = 2
        else:
            suggested_rating = 1
        
        culler_keywords.append(f'CullerSuggestedRating:{suggested_rating}')
    
    # Store technical analysis in separate field (not keywords)
    analysis = {
        'decision': decision,
        'confidence': f'{confidence:.2f}',
        'issues': issues_str,
        'blur_score': f'{metrics.blur_score:.2f}',
        'exposure_score': f'{metrics.exposure_score:.2f}',
        'composition_score': f'{metrics.composition_score:.2f}',
        'overall_quality': f'{metrics.overall_quality:.2f}'
    }
    metadata['PhotoCullerAnalysis'] = analysis
    
    # Only add the basic culler tags to keywords (clean and searchable)
    existing_keywords.extend(culler_keywords)
    metadata['Keywords'] = existing_keywords
    
    # Update metadata date
    metadata['MetadataDate'] = datetime.now().strftime('%a %b %d %H:%M:%S %Y')
    metadata['MetadataDateOffset'] = 0
    
    # Write back
    try:
        with open(on1_file, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        return True
    except:
        return False


def append_to_csv(csv_file, filepath, decision, confidence, issues, metrics, processing_ms):
    """Append result to CSV file"""
    file_exists = csv_file.exists()
    
    with open(csv_file, 'a', newline='') as f:
        fieldnames = ['timestamp', 'filepath', 'filename', 'decision', 'confidence', 'issues', 
                     'blur_score', 'exposure_score', 'composition_score', 'overall_score', 'processing_ms']
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'timestamp': datetime.now().isoformat(),
            'filepath': str(filepath),
            'filename': filepath.name,
            'decision': decision,
            'confidence': confidence,
            'issues': ', '.join(issues) if issues else '',
            'blur_score': metrics.blur_score,
            'exposure_score': metrics.exposure_score,
            'composition_score': metrics.composition_score,
            'overall_score': metrics.overall_quality,
            'processing_ms': processing_ms
        })


def print_result(filepath, decision, confidence, issues, metrics):
    """Print result for this photo"""
    issues_str = ', '.join(issues) if issues else 'none'
    
    # Color code the decision
    if decision == 'Delete':
        decision_colored = f"🔴 {decision}"
    elif decision == 'Review':
        decision_colored = f"🟡 {decision}"
    else:
        decision_colored = f"🟢 {decision}"
    
    # Confidence bar
    conf_bars = int(confidence * 10)
    conf_visual = '█' * conf_bars + '░' * (10 - conf_bars)
    
    print(f"{decision_colored:<12} {filepath.name}")
    print(f"   Confidence: {confidence:.2f} [{conf_visual}]")
    if issues:
        print(f"   Issues: {issues_str}")
    print(f"   Scores: blur={metrics.blur_score:.2f} exposure={metrics.exposure_score:.2f} composition={metrics.composition_score:.2f}")
    
    # Show description if available
    if hasattr(metrics, 'description') and metrics.description:
        print(f"   📝 {metrics.description}")
    
    # Show keywords if available
    if hasattr(metrics, 'keywords') and metrics.keywords:
        keywords_str = ', '.join(metrics.keywords[:6])  # Show first 6 keywords
        print(f"   🏷️  {keywords_str}")
    
    print()


@click.command()
@click.argument('folder', type=click.Path(exists=True))
@click.option('--fast', is_flag=True, help='Use fast CV mode (no ON1 metadata)')
@click.option('--cache-dir', type=click.Path(), help='Directory for thumbnail and results cache')
@click.option('--csv-file', default='photo_culler_results.csv', help='CSV file to append results to')
@click.option('--move-deletes', is_flag=True, help='Move files marked for deletion')
@click.option('--use-ollama', is_flag=True, help='Use Ollama vision model instead of CLIP')
@click.option('--ollama-model', default='llava:13b', help='Ollama model to use')
@click.option('--verbose', is_flag=True, help='Show processing details')
@click.option('--extensions', default='nef,cr2,arw,jpg,jpeg', help='File extensions to process')
@click.option('--override', is_flag=True, help='Override existing keywords and descriptions (preserves ratings and edits)')
def cull_on1(folder, fast, cache_dir, csv_file, move_deletes, use_ollama, ollama_model, verbose, extensions, override):
    """
    All-in-one photo culler for ON1 Photo RAW:
    
    - Analyzes photos with AI vision (unless --fast)
    - Updates ON1 .on1 metadata files (preserves existing data)
    - Appends results to CSV file
    - Shows result for each photo processed
    - Moves deletion candidates if requested
    
    Examples:
    
        # Full AI analysis with Ollama + ON1 metadata
        python culler_on1.py /photos
        
        # Fast mode for quick triage (no metadata)
        python culler_on1.py /photos --fast
        
        # Move deletions and use custom CSV
        python culler_on1.py /photos --move-deletes --csv-file my_results.csv
    """
    
    logger = setup_logging(verbose)
    
    # Determine mode
    mode = ProcessingMode.FAST if fast else ProcessingMode.ACCURATE
    
    folder = Path(folder)
    cache = Path(cache_dir) if cache_dir else None
    csv_path = Path(csv_file)
    
    # Parse extensions
    ext_list = ['.' + ext.strip().lstrip('.') for ext in extensions.split(',')]
    
    print(f"📁 Folder: {folder}")
    print(f"🔧 Mode: {mode.value}" + (f" (Ollama: {ollama_model})" if use_ollama and not fast else ""))
    print(f"📊 CSV: {csv_path}")
    print(f"📝 ON1 Metadata: {'Yes' if not fast else 'No (fast mode)'}")
    print(f"📎 Extensions: {', '.join(ext_list)}")
    print("=" * 60)
    
    # Initialize culler
    try:
        culler = BatchCuller(
            cache_dir=cache, 
            mode=mode,
            max_workers=4,
            batch_size=8,
            use_ollama=use_ollama,
            ollama_model=ollama_model
        )
        
    except Exception as e:
        print(f"❌ Failed to initialize culler: {e}")
        return
    
    # Find all files
    files = []
    for ext in ext_list:
        files.extend(folder.glob(f'*{ext}'))
        files.extend(folder.glob(f'*{ext.upper()}'))
    
    files = sorted(files)
    
    if not files:
        print(f"❌ No files found with extensions: {ext_list}")
        return
    
    print(f"🔍 Found {len(files)} files to process\n")
    
    # Process each file
    results = {'Keep': [], 'Delete': [], 'Review': [], 'Failed': []}
    on1_updated = 0
    
    for i, filepath in enumerate(files, 1):
        print(f"[{i:3d}/{len(files)}] Processing {filepath.name}...")
        
        try:
            result = culler.process_image(filepath)
            
            if result:
                decision = result.decision
                confidence = result.confidence
                issues = result.issues
                metrics = result.metrics
                processing_ms = result.processing_ms
                
                # Add to results
                results[decision].append(result)
                
                # Print result
                print_result(filepath, decision, confidence, issues, metrics)
                
                # Write ON1 metadata (unless fast mode)
                if not fast:
                    on1_file = filepath.with_suffix('.on1')
                    file_existed = on1_file.exists()
                    
                    if write_on1_metadata(filepath, decision, metrics, confidence, issues, override):
                        on1_updated += 1
                        if file_existed:
                            action = "Overrode" if override else "Updated"
                            print(f"   ✅ {action} ON1 metadata")
                        else:
                            print(f"   ✅ Created ON1 metadata file")
                    else:
                        print(f"   ❌ Failed to write ON1 metadata")
                
                # Append to CSV
                append_to_csv(csv_path, filepath, decision, confidence, issues, metrics, processing_ms)
                
            else:
                print(f"❌ Failed to process {filepath.name}")
                results['Failed'].append(filepath)
                
        except Exception as e:
            print(f"❌ Error processing {filepath.name}: {e}")
            results['Failed'].append(filepath)
    
    # Summary
    print("=" * 60)
    print("🏁 CULLING COMPLETE")
    print("=" * 60)
    
    total = len(files)
    for decision, items in results.items():
        if items:
            percentage = len(items) / total * 100
            print(f"{decision:>8}: {len(items):>3} files ({percentage:4.1f}%)")
    
    # Move deletions if requested
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
    
    # Show results
    print(f"📊 Results saved to: {csv_path}")
    if not fast and on1_updated > 0:
        print(f"📝 Updated {on1_updated} ON1 metadata files")
        print(f"🔄 Restart ON1 Photo RAW to see new keywords")
    
    print(f"\n🎯 Recommendation: Review the {len(results.get('Delete', []))} deletion candidates")
    
    # Show what to look for in ON1
    if not fast and on1_updated > 0:
        print(f"\n🔍 In ON1 Photo RAW, search for:")
        print(f"   • PhotoCuller:Delete - See deletion candidates")
        print(f"   • CullerConfidence:0.8 - High confidence decisions")
        print(f"   • CullerSuggestedRating:1 - Photos rated 1 star")


if __name__ == '__main__':
    cull_on1()