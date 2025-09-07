#!/usr/bin/env python3
"""
Test script for the enhanced focus analyzer
"""

import sys
import os
from PIL import Image
from enhanced_focus_analyzer import EnhancedFocusAnalyzer

def test_enhanced_focus_analyzer():
    """Test the enhanced focus analyzer initialization and basic functionality"""

    print("Testing Enhanced Focus Analyzer...")

    try:
        # Initialize the analyzer
        print("Initializing analyzer...")
        analyzer = EnhancedFocusAnalyzer()
        print("✓ Analyzer initialized successfully")

        # Test with a synthetic image
        print("Creating synthetic test image...")
        # Create a simple test image (100x100 pixels)
        test_image = Image.new('RGB', (100, 100), color=(128, 128, 128))
        print("✓ Test image created")

        print("Analyzing focus...")
        result = analyzer.analyze_focus(test_image)
        print("✓ Analysis completed")

        print("\n=== Enhanced Focus Analysis Results ===")
        print(f"Subject Type: {result['subject_type']}")
        print(".3f")
        print(".3f")
        print(".3f")
        print(f"Is Shallow DOF: {result['is_shallow_dof']}")

        print(f"\nIssues: {result['issues']}")
        print(f"Recommendations: {result['recommendations']}")

        if result['focus_regions']:
            print(f"\nFocus Regions ({len(result['focus_regions'])}):")
            for i, region in enumerate(result['focus_regions']):
                print(f"  Region {i+1}: x={region.x}, y={region.y}, "
                      f"size={region.width}x{region.height}, "
                      ".3f"
                      f"importance={region.importance}")
        else:
            print("\nNo focus regions detected")

        print("\n✓ Test completed successfully!")
        return True

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_focus_analyzer()
