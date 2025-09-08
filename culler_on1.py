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

    # ON1 must create the file first - we can only edit existing files
    if not on1_file.exists():
        return False

    try:
        with open(on1_file, 'r') as f:
            data = json.load(f)
    except:
        return False

    # Find the photo entry by matching the "name" field
    photos = data.get('photos', {})
    photo_id = None
    filename = str(filepath.name)

    # Look through all photo entries to find one with matching name
    for key, photo_data in photos.items():
        if photo_data.get('name') == filename:
            photo_id = key
            break

    if photo_id is None:
        return False

    photo_data = photos[photo_id]

    # Get or create metadata section
    metadata = photo_data.setdefault('metadata', {})

    # Build new keywords list
    new_keywords = []

    if override:
        # Override mode: Start fresh with NO existing keywords
        # Also clear ON1's ML classifications to prevent keyword regeneration
        if 'ml_classes' in photo_data:
            # Clear the ML classifications that generate keywords
            ml_classes = photo_data.get('ml_classes', {})
            classifications = ml_classes.get('classifications', [])

            # Remove ONPanopticSegmenterV0 and PartialLabelingCSL entries
            # These are what generate the annoying keywords
            filtered_classifications = []
            for classification in classifications:
                class_type = classification.get('type', '')
                if class_type not in ['ONPanopticSegmenterV0', 'PartialLabelingCSL']:
                    # Keep only non-keyword generating classifications
                    filtered_classifications.append(classification)

            ml_classes['classifications'] = filtered_classifications
    else:
        # Preserve mode: Keep ONLY user-added keywords (not ON1's auto-generated ones)
        existing_keywords = metadata.get('Keywords', [])

        # List of typical ON1 auto-generated keywords to filter out
        on1_auto_keywords = {
            'Aqua', 'Banner', 'Brick wall', 'Bridge', 'Building', 'Color', 'Dirt',
            'Fence', 'Floor', 'Human action', 'Human head', 'Person', 'Plant', 'Sea',
            'Skin', 'Sky', 'Snowboard', 'Sports equipment', 'Swimming pool', 'T-shirt',
            'Tree', 'Vivid', 'Wall', 'Natural environment', 'Sand', 'Snow', 'Mountain',
            'Azure', 'Photography', 'Snapshot', 'Material property', 'Font', 'Gesture',
            'Grass', 'Wood', 'Metal', 'Glass', 'Plastic', 'Paper', 'Textile', 'Concrete'
        }

        # Keep only keywords that are:
        # 1. Not culler keywords (PhotoCuller:, etc.)
        # 2. Not in the ON1 auto-generated list
        # 3. Likely user-added (contain spaces, lowercase, or specific patterns)
        for kw in existing_keywords:
            if (not kw.startswith('PhotoCuller:') and
                not kw.startswith('CullerConfidence:') and
                not kw.startswith('CullerIssues:') and
                not kw.startswith('CullerSuggestedRating:') and
                kw not in on1_auto_keywords):
                # This is likely a user keyword, keep it
                new_keywords.append(kw)

    # Add our AI-generated keywords
    if hasattr(metrics, 'keywords') and metrics.keywords:
        ai_keywords = metrics.keywords if isinstance(metrics.keywords, list) else []
        new_keywords.extend(ai_keywords[:8])

    # Add culler analysis keywords
    issues_str = ', '.join(issues) if issues else 'none'
    culler_keywords = [
        f'PhotoCuller:{decision}',
        f'CullerConfidence:{confidence:.2f}',
        f'CullerIssues:{issues_str}'
    ]

    # Add rating suggestion if no rating exists
    existing_rating = metadata.get('Rating', 0)
    if existing_rating == 0:
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

    # Combine all keywords
    new_keywords.extend(culler_keywords)
    metadata['Keywords'] = new_keywords

    # Handle description
    if override:
        # Always replace with AI description in override mode
        if hasattr(metrics, 'description') and metrics.description:
            metadata['Description'] = metrics.description
    else:
        # Only add if empty
        if not metadata.get('Description'):
            if hasattr(metrics, 'description') and metrics.description:
                metadata['Description'] = metrics.description

    # Store technical analysis
    analysis = {
        'decision': decision,
        'confidence': f'{confidence:.2f}',
        'issues': issues_str,
        'blur_score': f'{metrics.blur_score:.2f}',
        'exposure_score': f'{metrics.exposure_score:.2f}',
        'composition_score': f'{metrics.composition_score:.2f}',
        'overall_quality': f'{metrics.overall_quality:.2f}'
    }

    if hasattr(metrics, 'enhanced_focus') and metrics.enhanced_focus:
        ef = metrics.enhanced_focus
        analysis['enhanced_focus'] = {
            'subject_type': ef.get('subject_type', 'unknown'),
            'subject_sharpness': f"{ef.get('subject_sharpness', 0):.2f}",
            'background_blur': f"{ef.get('background_blur', 0):.2f}",
            'is_shallow_dof': ef.get('is_shallow_dof', False),
            'focus_regions': len(ef.get('focus_regions', [])),
            'recommendations': ef.get('recommendations', [])
        }

    metadata['PhotoCullerAnalysis'] = analysis

    # Update metadata date
    metadata['MetadataDate'] = datetime.now().strftime('%a %b %d %H:%M:%S %Y')
    if 'MetadataDateOffset' not in metadata:
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


def process_sequential(files, culler, detail, fast, override, csv_path):
    """Original single-threaded processing"""
    from tqdm import tqdm
    
    results = {'Keep': [], 'Delete': [], 'Review': [], 'Failed': []}
    on1_updated = 0
    
    print("🚀 Starting sequential processing with progress tracking...")
    
    with tqdm(total=len(files), desc="Processing images", unit="img") as pbar:
        for filepath in files:
            pbar.set_description(f"Processing {filepath.name}")
            
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
                    
                    # Print result only if detail flag is enabled
                    if detail:
                        print_result(filepath, decision, confidence, issues, metrics)
                    
                    # Write ON1 metadata (unless fast mode)
                    if not fast:
                        on1_file = filepath.with_suffix('.on1')
                        file_existed = on1_file.exists()
                        
                        if write_on1_metadata(filepath, decision, metrics, confidence, issues, override):
                            on1_updated += 1
                            if detail:
                                if file_existed:
                                    action = "Overrode" if override else "Updated"
                                    print(f"   ✅ {action} ON1 metadata")
                                else:
                                    print(f"   ✅ Created ON1 metadata file")
                        else:
                            if detail:
                                print(f"   ❌ Failed to write ON1 metadata")
                    
                    # Append to CSV
                    append_to_csv(csv_path, filepath, decision, confidence, issues, metrics, processing_ms)
                    
                else:
                    if detail:
                        print(f"❌ Failed to process {filepath.name}")
                    results['Failed'].append(filepath)
                    
            except Exception as e:
                if detail:
                    print(f"❌ Error processing {filepath.name}: {e}")
                results['Failed'].append(filepath)
            
            pbar.update(1)
    
    return results, on1_updated


def process_concurrent(files, culler, concurrent, chunk_size, detail, fast, override, csv_path):
    """Concurrent processing with multiple Ollama instances"""
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    results = {'Keep': [], 'Delete': [], 'Review': [], 'Failed': []}
    on1_updated = 0
    results_lock = threading.Lock()
    on1_lock = threading.Lock()
    
    print(f"🚀 Starting concurrent processing: {concurrent} Ollama instances, chunk size {chunk_size}")
    
    # Auto-start additional Ollama servers if needed
    if concurrent > 1:
        import subprocess
        import time
        import requests
        import os
        
        print(f"   🔧 Setting up {concurrent} Ollama servers...")
        
        # Check which Ollama servers are already running
        running_ports = []
        for i in range(concurrent):
            port = 11434 + i
            try:
                response = requests.get(f"http://localhost:{port}/api/tags", timeout=1)
                if response.status_code == 200:
                    running_ports.append(port)
                    print(f"   ✅ Ollama server already running on port {port}")
            except:
                pass
        
        # Start missing Ollama servers
        started_processes = []
        for i in range(concurrent):
            port = 11434 + i
            if port not in running_ports:
                try:
                    print(f"   🚀 Starting Ollama server on port {port}...")
                    # Create full environment with OLLAMA_HOST
                    env = os.environ.copy()
                    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
                    
                    process = subprocess.Popen(
                        ["ollama", "serve"],
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    started_processes.append((port, process))
                    time.sleep(3)  # Give server more time to start
                    
                    # Verify server started
                    try:
                        response = requests.get(f"http://localhost:{port}/api/tags", timeout=3)
                        if response.status_code == 200:
                            print(f"   ✅ Ollama server started successfully on port {port}")
                        else:
                            print(f"   ⚠️  Ollama server on port {port} may not be ready yet")
                    except:
                        print(f"   ⚠️  Ollama server on port {port} starting (may take a moment)")
                        
                except Exception as e:
                    print(f"   ❌ Failed to start Ollama server on port {port}: {e}")
    
    # Create multiple culler instances with different Ollama ports
    cullers = []
    if concurrent > 1:
        for i in range(concurrent):
            ollama_port = 11434 + i  # Use ports 11434, 11435, 11436, etc.
            ollama_host = f"http://localhost:{ollama_port}"
            
            # Create new BatchCuller instance with different port
            concurrent_culler = BatchCuller(
                cache_dir=culler.cache_dir,
                mode=culler.mode,
                max_workers=culler.max_workers,
                batch_size=culler.batch_size,
                force_cpu=False,  # Use original force_cpu setting if available
                use_ollama=culler.use_ollama,
                ollama_model=culler.ollama_model,
                ollama_host=ollama_host,
                learning_enabled=culler.learning_enabled
            )
            cullers.append(concurrent_culler)
            print(f"   🔗 Created Ollama client {i+1} for port {ollama_port}")
    else:
        cullers.append(culler)
    
    def process_chunk(file_chunk, culler_index=0):
        """Process a chunk of files with assigned culler"""
        batch_culler = cullers[culler_index % len(cullers)]
        batch_results = []
        batch_on1_updated = 0
        
        for filepath in file_chunk:
            try:
                result = batch_culler.process_image(filepath)
                
                if result:
                    decision = result.decision
                    confidence = result.confidence
                    issues = result.issues
                    metrics = result.metrics
                    processing_ms = result.processing_ms
                    
                    batch_results.append((decision, result))
                    
                    # Print result only if detail flag is enabled
                    if detail:
                        print_result(filepath, decision, confidence, issues, metrics)
                    
                    # Write ON1 metadata (unless fast mode)
                    if not fast:
                        on1_file = filepath.with_suffix('.on1')
                        file_existed = on1_file.exists()
                        
                        if write_on1_metadata(filepath, decision, metrics, confidence, issues, override):
                            batch_on1_updated += 1
                            if detail:
                                if file_existed:
                                    action = "Overrode" if override else "Updated"
                                    print(f"   ✅ {action} ON1 metadata")
                                else:
                                    print(f"   ✅ Created ON1 metadata file")
                        else:
                            if detail:
                                print(f"   ❌ Failed to write ON1 metadata")
                    
                    # Append to CSV
                    append_to_csv(csv_path, filepath, decision, confidence, issues, metrics, processing_ms)
                    
                else:
                    if detail:
                        print(f"❌ Failed to process {filepath.name}")
                    batch_results.append(('Failed', filepath))
                    
            except Exception as e:
                if detail:
                    print(f"❌ Error processing {filepath.name}: {e}")
                batch_results.append(('Failed', filepath))
        
        return batch_results, batch_on1_updated
    
    # Split files into chunks
    chunks = []
    for i in range(0, len(files), chunk_size):
        chunks.append(files[i:i + chunk_size])
    
    # Process chunks concurrently
    with tqdm(total=len(files), desc="Processing images", unit="img") as pbar:
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            # Submit all chunks with culler assignment
            future_to_chunk = {
                executor.submit(process_chunk, chunk, chunk_idx): chunk 
                for chunk_idx, chunk in enumerate(chunks)
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    batch_results, batch_on1_updated = future.result()
                    
                    # Thread-safe result aggregation
                    with results_lock:
                        for decision, item in batch_results:
                            results[decision].append(item)
                    
                    with on1_lock:
                        on1_updated += batch_on1_updated
                    
                    pbar.update(len(chunk))
                    
                except Exception as e:
                    print(f"❌ Chunk processing failed: {e}")
                    pbar.update(len(chunk))
    
    return results, on1_updated


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
@click.option('--detail', is_flag=True, help='Show individual file results during processing')
@click.option('--extensions', default='nef,cr2,arw,jpg,jpeg', help='File extensions to process')
@click.option('--override', is_flag=True, help='Override ALL existing keywords and descriptions (preserves ratings only)')
@click.option('--learning', is_flag=True, help='Enable adaptive learning mode')
@click.option('--chunk-size', default=1, help='Number of images per worker chunk (for load balancing)')
@click.option('--concurrent', default=1, help='Number of concurrent Ollama instances (requires multiple Ollama servers on different ports)')
def cull_on1(folder, fast, cache_dir, csv_file, move_deletes, use_ollama, ollama_model, verbose, detail, extensions, override, learning, chunk_size, concurrent):
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
            ollama_model=ollama_model,
            learning_enabled=learning
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
    
    # Process with concurrent Ollama instances
    if concurrent > 1 or chunk_size > 1:
        results, on1_updated = process_concurrent(files, culler, concurrent, chunk_size, detail, fast, override, csv_path)
    else:
        # Single threaded processing (original)
        results, on1_updated = process_sequential(files, culler, detail, fast, override, csv_path)
    
    # Save session data (adaptive learning, caches, etc.)
    try:
        culler.save_session()
        
        # Show adaptive learning insights if available
        session_summary = culler.get_session_summary()
        if session_summary and session_summary.get('total_processed', 0) > 0:
            print("\n📚 ADAPTIVE LEARNING SUMMARY")
            print("=" * 60)
            
            # Show detected style preferences
            detected_style = session_summary.get('detected_style', {})
            if detected_style.get('uses_shallow_dof'):
                print("📸 Detected: Preference for shallow depth of field photography")
            if detected_style.get('prefers_dark_mood'):
                print("🌙 Detected: Preference for dark/moody exposure style")
            if detected_style.get('common_subjects'):
                subjects = ', '.join(detected_style['common_subjects'])
                print(f"🎯 Common subjects: {subjects}")
            
            # Show average quality scores
            for metric in ['blur', 'exposure', 'composition']:
                avg_key = f'avg_{metric}'
                if avg_key in session_summary:
                    score = session_summary[avg_key]
                    print(f"📈 Average {metric} score: {score:.2f}")
            
            print("📊 Learning data saved - future sessions will be more accurate")
            
    except Exception as e:
        print(f"⚠️  Warning: Failed to save session data: {e}")
    
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
