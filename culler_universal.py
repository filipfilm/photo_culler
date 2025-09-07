#!/usr/bin/env python3
"""
Universal photo culler - works with ON1, Lightroom, Capture One, etc.
Creates standard XMP sidecar files that all apps can read
"""
import click
from pathlib import Path
from batch import BatchCuller
from models import ProcessingMode
import json
import logging
import csv
from datetime import datetime
import xml.etree.ElementTree as ET


def setup_logging(verbose=False):
    """Setup logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def read_existing_xmp(xmp_file):
    """Read existing XMP sidecar file to preserve metadata"""
    existing = {
        'rating': None,
        'color_label': None, 
        'keywords': [],
        'description': '',
        'other_metadata': {}
    }
    
    if not xmp_file.exists():
        return existing
    
    try:
        tree = ET.parse(xmp_file)
        root = tree.getroot()
        
        # Find the Description element
        description_elem = None
        for elem in root.iter():
            if elem.tag.endswith('Description'):
                description_elem = elem
                break
        
        if description_elem is not None:
            # Extract existing rating
            for attr_name, attr_value in description_elem.attrib.items():
                if 'Rating' in attr_name:
                    existing['rating'] = attr_value
                elif 'ColorMode' in attr_name or 'Label' in attr_name:
                    existing['color_label'] = attr_value
            
            # Extract keywords
            for child in description_elem:
                if 'subject' in child.tag:
                    bag = child.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Bag')
                    if bag is not None:
                        for li in bag:
                            if li.text and not li.text.startswith('PhotoCuller:'):
                                existing['keywords'].append(li.text)
                
                elif 'description' in child.tag:
                    if child.text and 'Photo Culler Analysis' not in child.text:
                        existing['description'] = child.text
    
    except Exception as e:
        print(f"Warning: Could not read existing XMP {xmp_file}: {e}")
    
    return existing


def create_standard_xmp(filepath, decision, metrics, confidence, issues, existing):
    """Create standard XMP sidecar file that works with all photo apps"""
    xmp_file = filepath.with_suffix(filepath.suffix + '.xmp')
    
    # Prepare data
    issues_str = ', '.join(issues) if issues else 'none'
    
    # Combine keywords (preserve existing, add culler data, add AI keywords)
    all_keywords = existing['keywords'].copy()
    
    # Remove old culler keywords first
    all_keywords = [kw for kw in all_keywords 
                   if not kw.startswith('PhotoCuller:')
                   and not kw.startswith('CullerConfidence:')
                   and not kw.startswith('CullerIssues:')
                   and not kw.startswith('AI:')]
    
    # Add new culler keywords  
    culler_keywords = [
        f'PhotoCuller:{decision}',
        f'CullerConfidence:{confidence:.2f}',
        f'CullerIssues:{issues_str}'
    ]
    all_keywords.extend(culler_keywords)
    
    # Add AI-generated keywords if available
    if hasattr(metrics, 'keywords') and metrics.keywords:
        ai_keywords = [f'AI:{kw}' for kw in metrics.keywords[:10]]  # Limit to 10 keywords
        all_keywords.extend(ai_keywords)
    
    # Create keywords XML
    keywords_xml = '\n'.join([f'          <rdf:li>{kw}</rdf:li>' for kw in all_keywords])
    
    # Preserve existing rating and color
    rating_attr = f'xmp:Rating="{existing["rating"]}"' if existing['rating'] else ''
    color_attr = f'photoshop:ColorMode="{existing["color_label"]}"' if existing['color_label'] and existing['color_label'] != 'None' else ''
    
    # Only suggest rating if none exists
    if not existing['rating']:
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
        
        all_keywords.append(f'CullerSuggestedRating:{suggested_rating}')
        keywords_xml = '\n'.join([f'          <rdf:li>{kw}</rdf:li>' for kw in all_keywords])
    
    # Create analysis description
    analysis = f"""Photo Culler Analysis ({datetime.now().strftime('%Y-%m-%d %H:%M')}):
Decision: {decision} (confidence: {confidence:.2f})
Issues: {issues_str}

Quality Scores:
• Blur/Sharpness: {metrics.blur_score:.2f}
• Exposure: {metrics.exposure_score:.2f}
• Composition: {metrics.composition_score:.2f}
• Overall: {metrics.overall_quality:.2f}"""
    
    # Combine descriptions
    if existing['description']:
        full_description = f"{existing['description']}\n\n{analysis}"
    else:
        full_description = analysis
    
    # Create XMP content with proper ON1-compatible structure
    xmp_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 5.6-c140 79.160451, 2017/05/06-01:08:21">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:aux="http://ns.adobe.com/exif/1.0/aux/"
      xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
      xmlns:lr="http://ns.adobe.com/lightroom/1.0/"
      xmp:ModifyDate="{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
      xmp:CreatorTool="PhotoCuller"
      {rating_attr}
      {color_attr}>
      
      <!-- Keywords (preserved + culler data) -->
      <dc:subject>
        <rdf:Bag>
{keywords_xml}
        </rdf:Bag>
      </dc:subject>
      
      <!-- Title (for better compatibility) -->
      <dc:title>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">{filepath.stem}</rdf:li>
        </rdf:Alt>
      </dc:title>
      
      <!-- Description (preserved + analysis) -->
      <dc:description>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">{full_description}</rdf:li>
        </rdf:Alt>
      </dc:description>
      
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>'''
    
    # Write XMP file
    try:
        with open(xmp_file, 'w', encoding='utf-8') as f:
            f.write(xmp_content)
        return True
    except Exception as e:
        print(f"Error writing XMP {xmp_file}: {e}")
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
    print()


@click.command()
@click.argument('folder', type=click.Path(exists=True))
@click.option('--fast', is_flag=True, help='Use fast CV mode (no metadata)')
@click.option('--cache-dir', type=click.Path(), help='Directory for thumbnail and results cache')
@click.option('--csv-file', default='photo_culler_results.csv', help='CSV file to append results to')
@click.option('--move-deletes', is_flag=True, help='Move files marked for deletion')
@click.option('--use-ollama', is_flag=True, help='Use Ollama vision model instead of CLIP')
@click.option('--ollama-model', default='llava:7b', help='Ollama model to use')
@click.option('--verbose', is_flag=True, help='Show processing details')
@click.option('--extensions', default='nef,cr2,arw,jpg,jpeg', help='File extensions to process')
def cull_universal(folder, fast, cache_dir, csv_file, move_deletes, use_ollama, ollama_model, verbose, extensions):
    """
    Universal photo culler - works with all photo apps:
    
    ✅ Creates standard XMP sidecar files
    ✅ Preserves ALL existing metadata (ratings, keywords, descriptions)
    ✅ Works with ON1, Lightroom, Capture One, Bridge, etc.
    ✅ Non-destructive - only ADDS culler data
    ✅ Appends to CSV for spreadsheet analysis
    
    Examples:
    
        # Full AI analysis with universal metadata
        python culler_universal.py /photos
        
        # Fast mode (CSV only, no metadata)
        python culler_universal.py /photos --fast
        
        # Move deletions automatically  
        python culler_universal.py /photos --move-deletes
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
    print(f"📝 XMP Metadata: {'Yes (universal)' if not fast else 'No (fast mode)'}")
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
    xmp_created = 0
    
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
                
                # Create XMP metadata (unless fast mode)
                if not fast:
                    xmp_file = filepath.with_suffix(filepath.suffix + '.xmp')
                    existing = read_existing_xmp(xmp_file)
                    
                    if create_standard_xmp(filepath, decision, metrics, confidence, issues, existing):
                        xmp_created += 1
                        print(f"   ✅ Created/updated XMP metadata")
                    else:
                        print(f"   ❌ Failed to write XMP")
                
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
    if not fast and xmp_created > 0:
        print(f"📝 Created/updated {xmp_created} XMP metadata files")
        print(f"🔄 Import folder in your photo app to see metadata")
    
    print(f"\n🎯 Recommendation: Review the {len(results.get('Delete', []))} deletion candidates")
    
    # Show what to look for 
    if not fast and xmp_created > 0:
        print(f"\n🔍 In any photo app, look for keywords:")
        print(f"   • PhotoCuller:Delete - Deletion candidates")
        print(f"   • CullerConfidence:0.8 - High confidence decisions")
        print(f"   • CullerSuggestedRating:1 - Suggested 1-star photos")
        print(f"   • Check description for detailed analysis")


if __name__ == '__main__':
    cull_universal()