#!/usr/bin/env python3
"""
Enhanced Photo Culler with all new features integrated
"""
import click
from pathlib import Path
from batch import BatchCuller
from models import ProcessingMode
from similarity_detector import SimilarityDetector
from smart_sequencer import SmartSequencer
from dynamic_model_selector import DynamicModelSelector
from metadata_enhancer import MetadataEnhancer
from report_generator import ReportGenerator
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
import time


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
@click.option('--fast', is_flag=True, help='Use fast CV mode')
@click.option('--cache-dir', type=click.Path(), help='Directory for cache')
@click.option('--detect-similar', is_flag=True, help='Detect and group similar/duplicate images')
@click.option('--similarity-threshold', default=0.90, help='Similarity threshold (0-1)')
@click.option('--smart-sequence', is_flag=True, help='Use smart sequencing for batch processing')
@click.option('--dynamic-model', is_flag=True, help='Use dynamic model selection for Ollama')
@click.option('--generate-report', is_flag=True, help='Generate detailed HTML report')
@click.option('--report-dir', type=click.Path(), help='Directory for reports')
@click.option('--enhanced-metadata', is_flag=True, help='Extract comprehensive metadata')
@click.option('--use-ollama', is_flag=True, help='Use Ollama vision model')
@click.option('--ollama-model', default='llava:13b', help='Ollama model to use')
@click.option('--verbose', is_flag=True, help='Show processing details')
@click.option('--extensions', default='nef,cr2,arw,jpg,jpeg', help='File extensions to process')
@click.option('--concurrent', default=1, help='Number of concurrent Ollama instances')
def enhanced_cull(folder, fast, cache_dir, detect_similar, similarity_threshold,
                 smart_sequence, dynamic_model, generate_report, report_dir,
                 enhanced_metadata, use_ollama, ollama_model, verbose, 
                 extensions, concurrent):
    """
    Enhanced photo culler with advanced features:
    
    - Similarity/duplicate detection
    - Smart batch sequencing  
    - Dynamic model selection
    - Comprehensive metadata extraction
    - Detailed HTML reports
    - Enhanced technical QC
    
    Examples:
    
        # Full analysis with all features
        python enhanced_culler.py /photos --detect-similar --smart-sequence --generate-report
        
        # Quick duplicate detection only
        python enhanced_culler.py /photos --fast --detect-similar
        
        # Professional workflow with metadata and reports
        python enhanced_culler.py /photos --enhanced-metadata --generate-report --dynamic-model
    """
    
    logger = setup_logging(verbose)
    session_start = datetime.now()
    
    # Setup paths
    folder = Path(folder)
    cache = Path(cache_dir) if cache_dir else None
    report_path = Path(report_dir) if report_dir else folder / 'reports'
    
    # Parse extensions
    ext_list = ['.' + ext.strip().lstrip('.') for ext in extensions.split(',')]
    
    logger.info(f"🚀 Enhanced Photo Culler starting")
    logger.info(f"📁 Folder: {folder}")
    logger.info(f"🔧 Features: {_get_enabled_features(detect_similar, smart_sequence, dynamic_model, generate_report, enhanced_metadata)}")
    print("=" * 80)
    
    # Find all files
    files = []
    for ext in ext_list:
        files.extend(folder.glob(f'*{ext}'))
        files.extend(folder.glob(f'*{ext.upper()}'))
    
    files = sorted(files)
    
    if not files:
        logger.error(f"❌ No files found with extensions: {ext_list}")
        return
    
    logger.info(f"🔍 Found {len(files)} files to process")
    
    # Initialize components
    session_info = {
        'start_time': session_start,
        'folder': str(folder),
        'total_files': len(files),
        'features_enabled': _get_enabled_features(detect_similar, smart_sequence, dynamic_model, generate_report, enhanced_metadata),
        'processing_mode': 'fast' if fast else 'accurate'
    }
    
    # Similarity detection
    if detect_similar:
        logger.info("🔍 Running similarity detection...")
        similarity_detector = SimilarityDetector(threshold=similarity_threshold)
        
        similar_groups = similarity_detector.find_similar_groups(files)
        similarity_stats = similarity_detector.get_similarity_stats(files)
        
        logger.info(f"📊 Similarity analysis complete:")
        logger.info(f"   Similar groups: {similarity_stats['similarity_groups']}")
        logger.info(f"   Potential deletions: {similarity_stats['potential_deletions']}")
        logger.info(f"   Storage savings: {similarity_stats['storage_savings_potential']} files")
        
        session_info['similarity_analysis'] = {
            'groups_found': len(similar_groups),
            'stats': similarity_stats,
            'threshold_used': similarity_threshold
        }
        
        print()
    
    # Smart sequencing
    processing_groups = None
    if smart_sequence:
        logger.info("🧠 Creating smart processing sequence...")
        sequencer = SmartSequencer()
        processing_groups = sequencer.sequence_for_processing(files)
        
        sequence_summary = sequencer.get_processing_order_summary(processing_groups)
        logger.info(f"📋 Processing sequence created:")
        logger.info(f"   Groups: {sequence_summary['total_groups']}")
        logger.info(f"   Burst sequences: {sequence_summary['groups_by_type'].get('burst', 0)}")
        logger.info(f"   Sessions: {sequence_summary['groups_by_type'].get('session', 0)}")
        
        session_info['smart_sequencing'] = {
            'groups_created': len(processing_groups),
            'sequence_summary': sequence_summary
        }
        
        print()
    
    # Dynamic model selection
    selected_model = ollama_model
    if dynamic_model and use_ollama and not fast:
        logger.info("🎯 Running dynamic model selection...")
        
        # Sample a few images for model selection
        sample_files = files[:min(10, len(files))]
        sample_images = []
        
        from extractor import RawThumbnailExtractor
        extractor = RawThumbnailExtractor(cache)
        
        for sample_file in sample_files:
            img = extractor.extract(sample_file)
            if img:
                sample_images.append(img)
        
        if sample_images:
            model_selector = DynamicModelSelector()
            recommendations = model_selector.get_model_recommendations(sample_images, "quality")
            
            if recommendations['unified_recommendation']:
                selected_model = recommendations['unified_recommendation']
                logger.info(f"🎯 Selected model: {selected_model}")
                
                # Optimize concurrent instances
                optimal_concurrent = model_selector.get_optimal_concurrent_instances(selected_model)
                if optimal_concurrent != concurrent:
                    concurrent = optimal_concurrent
                    logger.info(f"🔧 Adjusted concurrent instances: {concurrent}")
            
            session_info['dynamic_model_selection'] = {
                'recommendations': recommendations,
                'selected_model': selected_model,
                'concurrent_instances': concurrent
            }
        
        print()
    
    # Initialize enhanced culler
    mode = ProcessingMode.FAST if fast else ProcessingMode.ACCURATE
    
    try:
        culler = BatchCuller(
            cache_dir=cache,
            mode=mode,
            max_workers=4,
            batch_size=8,
            use_ollama=use_ollama,
            ollama_model=selected_model,
            learning_enabled=True
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize culler: {e}")
        return
    
    # Process images
    logger.info("🔄 Processing images...")
    
    if processing_groups and smart_sequence:
        # Process using smart groups
        results = _process_smart_groups(processing_groups, culler, logger)
    else:
        # Standard folder processing
        results = culler.process_folder_batch(folder, ext_list)
    
    session_end = datetime.now()
    session_info['end_time'] = session_end
    session_info['total_processing_time'] = str(session_end - session_start)
    
    # Enhanced metadata extraction
    if enhanced_metadata:
        logger.info("📋 Extracting enhanced metadata...")
        metadata_enhancer = MetadataEnhancer()
        
        enhanced_metadata_results = {}
        processed_count = 0
        
        for decision, items in results.items():
            if isinstance(items, list):
                for item in items:
                    if hasattr(item, 'filepath'):
                        try:
                            metadata = metadata_enhancer.extract_comprehensive_metadata(item.filepath)
                            enhanced_metadata_results[str(item.filepath)] = metadata
                            processed_count += 1
                            
                            if processed_count % 50 == 0:
                                logger.info(f"   Processed {processed_count} metadata extractions...")
                                
                        except Exception as e:
                            logger.warning(f"Metadata extraction failed for {item.filepath.name}: {e}")
        
        logger.info(f"📋 Enhanced metadata extracted for {processed_count} files")
        session_info['enhanced_metadata'] = {
            'files_processed': processed_count,
            'extraction_enabled': True
        }
        
        print()
    
    # Generate reports
    if generate_report:
        logger.info("📊 Generating detailed report...")
        
        report_generator = ReportGenerator()
        report_path.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML report
        html_report_path = report_path / f"photo_culler_report_{session_start.strftime('%Y%m%d_%H%M%S')}.html"
        report_generator.generate_html_report(results, session_info, html_report_path)
        
        # Generate JSON report
        json_report_path = report_path / f"photo_culler_data_{session_start.strftime('%Y%m%d_%H%M%S')}.json"
        report_generator.generate_json_report(results, session_info, json_report_path)
        
        # Generate CSV summary
        csv_report_path = report_path / f"photo_culler_summary_{session_start.strftime('%Y%m%d_%H%M%S')}.csv"
        report_generator.generate_csv_summary(results, csv_report_path)
        
        logger.info(f"📊 Reports generated:")
        logger.info(f"   HTML: {html_report_path}")
        logger.info(f"   JSON: {json_report_path}")
        logger.info(f"   CSV: {csv_report_path}")
        
        print()
    
    # Save session data
    try:
        culler.save_session()
    except Exception as e:
        logger.warning(f"⚠️  Failed to save session: {e}")
    
    # Final summary
    print("=" * 80)
    logger.info("🏁 ENHANCED CULLING COMPLETE")
    print("=" * 80)
    
    total = sum(len(v) for v in results.values() if isinstance(v, list))
    for decision, items in results.items():
        if isinstance(items, list) and items:
            percentage = len(items) / total * 100
            logger.info(f"{decision:>8}: {len(items):>3} files ({percentage:4.1f}%)")
    
    processing_duration = session_end - session_start
    logger.info(f"⏱️  Total time: {processing_duration}")
    
    if detect_similar and 'similarity_analysis' in session_info:
        stats = session_info['similarity_analysis']['stats']
        logger.info(f"🔍 Similarity: {stats['potential_deletions']} potential duplicates found")
    
    if generate_report:
        logger.info(f"📊 Detailed reports saved to: {report_path}")


def _get_enabled_features(detect_similar, smart_sequence, dynamic_model, 
                         generate_report, enhanced_metadata) -> List[str]:
    """Get list of enabled feature names"""
    features = []
    if detect_similar:
        features.append("Similarity Detection")
    if smart_sequence:
        features.append("Smart Sequencing") 
    if dynamic_model:
        features.append("Dynamic Model Selection")
    if generate_report:
        features.append("HTML Reports")
    if enhanced_metadata:
        features.append("Enhanced Metadata")
    
    return features


def _process_smart_groups(processing_groups, culler, logger) -> Dict:
    """Process images using smart groups"""
    results = {'Keep': [], 'Delete': [], 'Review': [], 'Failed': []}
    
    for i, group in enumerate(processing_groups):
        group_type = group.group_type
        file_count = len(group.files)
        
        logger.info(f"🔄 Processing group {i+1}/{len(processing_groups)}: {group_type} ({file_count} files)")
        
        # Process each file in the group
        for filepath in group.files:
            try:
                result = culler.process_image(filepath)
                if result:
                    results[result.decision].append(result)
                else:
                    results['Failed'].append(filepath)
            except Exception as e:
                logger.warning(f"Failed to process {filepath.name}: {e}")
                results['Failed'].append(filepath)
    
    return results


if __name__ == '__main__':
    enhanced_cull()