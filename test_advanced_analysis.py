#!/usr/bin/env python3
"""
Test script to compare Ollama vs Advanced Analysis on a single image
"""
import click
from pathlib import Path
from PIL import Image
from advanced_focus_detector import AdvancedFocusDetector
from smart_exposure_analyzer import SmartExposureAnalyzer  
from intelligent_composition_analyzer import IntelligentCompositionAnalyzer
from ollama_vision import ImprovedOllamaVisionAnalyzer
import json


@click.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--ollama-model', default='llava:13b', help='Ollama model to compare against')
def test_analysis(image_path, ollama_model):
    """
    Compare Ollama vs Advanced CV analysis on a single image
    
    Usage: python test_advanced_analysis.py /path/to/image.jpg
    """
    
    image_path = Path(image_path)
    
    print(f"🔍 Analyzing: {image_path.name}")
    print("=" * 60)
    
    # Load image
    try:
        image = Image.open(image_path)
        print(f"📐 Image size: {image.width}x{image.height}")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return
    
    # Ollama Analysis
    print("\n🤖 OLLAMA ANALYSIS (llava:13b)")
    print("-" * 30)
    try:
        ollama = ImprovedOllamaVisionAnalyzer(model=ollama_model)
        ollama_result = ollama.analyze(image)
        
        print(f"Blur Score:       {ollama_result.blur_score:.3f}")
        print(f"Exposure Score:   {ollama_result.exposure_score:.3f}")
        print(f"Composition Score: {ollama_result.composition_score:.3f}")
        print(f"Overall Quality:  {ollama_result.overall_quality:.3f}")
        if ollama_result.keywords:
            print(f"Keywords: {', '.join(ollama_result.keywords[:5])}")
        if ollama_result.description:
            print(f"Description: {ollama_result.description}")
            
    except Exception as e:
        print(f"❌ Ollama analysis failed: {e}")
        return
    
    # Advanced CV Analysis  
    print("\n🔬 ADVANCED CV ANALYSIS")
    print("-" * 30)
    
    # Focus Analysis
    print("\n📷 FOCUS ANALYSIS:")
    try:
        focus_detector = AdvancedFocusDetector()
        focus_analysis = focus_detector.analyze_focus(image)
        
        print(f"Focus Score:      {focus_analysis['focus_score']:.3f}")
        print(f"Focus Quality:    {focus_analysis['focus_quality']}")
        print(f"Is Sharp:         {focus_analysis['is_sharp']}")
        
        if focus_analysis['recommendations']:
            print(f"Recommendations:  {'; '.join(focus_analysis['recommendations'])}")
            
        # Show individual metrics for debugging
        print("\n  Individual Focus Metrics:")
        for metric, score in focus_analysis['individual_metrics'].items():
            print(f"    {metric:12}: {score:.3f}")
            
    except Exception as e:
        print(f"❌ Focus analysis failed: {e}")
    
    # Exposure Analysis
    print("\n💡 EXPOSURE ANALYSIS:")
    try:
        exposure_analyzer = SmartExposureAnalyzer()
        exposure_analysis = exposure_analyzer.analyze_exposure(image)
        
        print(f"Exposure Score:   {exposure_analysis['exposure_score']:.3f}")
        print(f"Exposure Category: {exposure_analysis['exposure_category']}")
        print(f"Is Well Exposed:  {exposure_analysis['is_well_exposed']}")
        
        if exposure_analysis['issues']:
            print(f"Issues:           {'; '.join(exposure_analysis['issues'])}")
        if exposure_analysis['recommendations']:
            print(f"Recommendations:  {'; '.join(exposure_analysis['recommendations'])}")
            
    except Exception as e:
        print(f"❌ Exposure analysis failed: {e}")
    
    # Composition Analysis
    print("\n🖼️  COMPOSITION ANALYSIS:")
    try:
        composition_analyzer = IntelligentCompositionAnalyzer()
        composition_analysis = composition_analyzer.analyze_composition(image)
        
        print(f"Composition Score: {composition_analysis['composition_score']:.3f}")
        print(f"Category:         {composition_analysis['composition_category']}")
        print(f"Well Composed:    {composition_analysis['is_well_composed']}")
        
        if composition_analysis['issues']:
            print(f"Issues:           {'; '.join(composition_analysis['issues'])}")
        if composition_analysis['recommendations']:
            print(f"Recommendations:  {'; '.join(composition_analysis['recommendations'])}")
            
        # Show specific composition details
        framing = composition_analysis['detailed_analysis']['framing_issues']
        if framing['framing_issues']:
            print(f"Framing Issues:   {'; '.join(framing['framing_issues'])}")
            
    except Exception as e:
        print(f"❌ Composition analysis failed: {e}")
    
    # Comparison
    print("\n📊 COMPARISON")
    print("-" * 30)
    print(f"{'Metric':<12} {'Ollama':<8} {'Advanced CV':<12} {'Difference':<10}")
    print("-" * 50)
    
    try:
        focus_diff = focus_analysis['focus_score'] - ollama_result.blur_score
        exp_diff = exposure_analysis['exposure_score'] - ollama_result.exposure_score  
        comp_diff = composition_analysis['composition_score'] - ollama_result.composition_score
        
        print(f"{'Focus':<12} {ollama_result.blur_score:<8.3f} {focus_analysis['focus_score']:<12.3f} {focus_diff:+.3f}")
        print(f"{'Exposure':<12} {ollama_result.exposure_score:<8.3f} {exposure_analysis['exposure_score']:<12.3f} {exp_diff:+.3f}")
        print(f"{'Composition':<12} {ollama_result.composition_score:<8.3f} {composition_analysis['composition_score']:<12.3f} {comp_diff:+.3f}")
        
    except:
        print("Could not compare - analysis incomplete")
    
    print("\n" + "=" * 60)
    print("💡 This helps you see if Ollama is missing technical issues!")


if __name__ == '__main__':
    test_analysis()