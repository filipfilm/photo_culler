#!/usr/bin/env python3
"""
Test script to verify integration of enhanced focus analyzer with existing system
"""

from PIL import Image
from analyzer import TechnicalAnalyzer

def test_integration():
    """Test that the enhanced focus analyzer integrates properly"""

    print("Testing Enhanced Focus Analyzer Integration...")

    try:
        # Initialize the technical analyzer (which should use enhanced focus)
        print("Initializing TechnicalAnalyzer...")
        analyzer = TechnicalAnalyzer()
        print("✓ TechnicalAnalyzer initialized")

        # Check if enhanced focus is available
        has_enhanced = analyzer.enhanced_focus is not None
        print(f"✓ Enhanced focus available: {has_enhanced}")

        # Create a test image
        print("Creating test image...")
        test_image = Image.new('RGB', (200, 200), color=(100, 150, 200))
        print("✓ Test image created")

        # Analyze the image
        print("Analyzing image with TechnicalAnalyzer...")
        metrics = analyzer.analyze(test_image)
        print("✓ Analysis completed")

        print("\n=== TechnicalAnalyzer Results ===")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")
        print(f"Processing Mode: {metrics.processing_mode}")

        print("\n✓ Integration test completed successfully!")
        return True

    except Exception as e:
        print(f"✗ Error during integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_integration()
