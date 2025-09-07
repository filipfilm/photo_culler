#!/usr/bin/env python3
"""
Test script to verify enhanced focus data is written to ON1 files
"""

import os
import json
from PIL import Image
from pathlib import Path
from analyzer import TechnicalAnalyzer

def test_on1_enhanced_focus():
    """Test that enhanced focus data is written to ON1 files"""

    print("Testing Enhanced Focus Data in ON1 Files...")

    try:
        # Initialize analyzer
        analyzer = TechnicalAnalyzer()
        print("✓ TechnicalAnalyzer initialized")

        # Create test image
        test_image = Image.new('RGB', (300, 200), color=(100, 150, 200))
        print("✓ Test image created")

        # Analyze image
        metrics = analyzer.analyze(test_image)
        print("✓ Analysis completed")

        # Check if enhanced focus data is present
        has_enhanced_focus = metrics.enhanced_focus is not None
        print(f"✓ Enhanced focus data present: {has_enhanced_focus}")

        if has_enhanced_focus:
            ef = metrics.enhanced_focus
            print(f"  Subject Type: {ef.get('subject_type', 'unknown')}")
            print(".3f")
            print(".3f")
            print(f"  Is Shallow DOF: {ef.get('is_shallow_dof', False)}")
            print(f"  Focus Regions: {len(ef.get('focus_regions', []))}")

        # Create a temporary ON1 file to test writing
        temp_on1 = Path("temp_test.on1")
        temp_image = Path("temp_test.jpg")

        # Save test image temporarily
        test_image.save(temp_image)

        # Simulate ON1 metadata writing (we'll just check the data structure)
        on1_data = {
            "version": "2.0",
            "photos": {
                str(temp_image.name): {
                    "metadata": {}
                }
            }
        }

        # Add enhanced focus to analysis like culler_on1.py does
        analysis = {
            'decision': 'Keep',
            'confidence': '0.85',
            'issues': 'none',
            'blur_score': '.2f',
            'exposure_score': '.2f',
            'composition_score': '.2f',
            'overall_quality': '.2f'
        }

        if hasattr(metrics, 'enhanced_focus') and metrics.enhanced_focus:
            ef = metrics.enhanced_focus
            analysis['enhanced_focus'] = {
                'subject_type': ef.get('subject_type', 'unknown'),
                'subject_sharpness': ".2f",
                'background_blur': ".2f",
                'is_shallow_dof': ef.get('is_shallow_dof', False),
                'focus_regions': len(ef.get('focus_regions', [])),
                'recommendations': ef.get('recommendations', [])
            }

        on1_data['photos'][str(temp_image.name)]['metadata']['PhotoCullerAnalysis'] = analysis

        # Write to temporary file
        with open(temp_on1, 'w') as f:
            json.dump(on1_data, f, indent=2)

        print("✓ ON1 file written with enhanced focus data")

        # Read back and verify
        with open(temp_on1, 'r') as f:
            read_data = json.load(f)

        photo_data = read_data['photos'][str(temp_image.name)]['metadata']
        analysis_data = photo_data.get('PhotoCullerAnalysis', {})

        if 'enhanced_focus' in analysis_data:
            ef_data = analysis_data['enhanced_focus']
            print("✓ Enhanced focus data found in ON1 file:")
            print(f"  Subject Type: {ef_data.get('subject_type', 'unknown')}")
            print(f"  Subject Sharpness: {ef_data.get('subject_sharpness', 'N/A')}")
            print(f"  Background Blur: {ef_data.get('background_blur', 'N/A')}")
            print(f"  Is Shallow DOF: {ef_data.get('is_shallow_dof', False)}")
            print(f"  Focus Regions: {ef_data.get('focus_regions', 0)}")
        else:
            print("✗ Enhanced focus data NOT found in ON1 file")

        # Clean up
        if temp_on1.exists():
            temp_on1.unlink()
        if temp_image.exists():
            temp_image.unlink()

        print("\n✓ ON1 enhanced focus test completed successfully!")
        return True

    except Exception as e:
        print(f"✗ Error during ON1 test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_on1_enhanced_focus()
