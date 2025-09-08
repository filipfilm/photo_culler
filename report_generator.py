"""
Session Report Generator - Creates detailed HTML reports for photo culling sessions
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import asdict
import base64
from PIL import Image
import io
import numpy as np


class ReportGenerator:
    """Generate detailed culling session reports"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_html_report(self, results: Dict, session_info: Dict, 
                           output_path: Path, include_thumbnails: bool = True):
        """Create interactive HTML report"""
        
        try:
            # Prepare data for template
            report_data = self._prepare_report_data(results, session_info)
            
            # Generate HTML content
            html_content = self._generate_html_content(
                report_data, include_thumbnails
            )
            
            # Write to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {e}")
            raise
    
    def _prepare_report_data(self, results: Dict, session_info: Dict) -> Dict:
        """Prepare and analyze data for the report"""
        
        # Calculate summary statistics
        total_files = sum(len(items) for items in results.values() if isinstance(items, list))
        keep_count = len(results.get('Keep', []))
        delete_count = len(results.get('Delete', []))
        review_count = len(results.get('Review', []))
        failed_count = len(results.get('Failed', []))
        
        # Calculate percentages
        keep_pct = (keep_count / total_files * 100) if total_files > 0 else 0
        delete_pct = (delete_count / total_files * 100) if total_files > 0 else 0
        review_pct = (review_count / total_files * 100) if total_files > 0 else 0
        failed_pct = (failed_count / total_files * 100) if total_files > 0 else 0
        
        # Analyze quality scores
        quality_analysis = self._analyze_quality_scores(results)
        
        # Analyze processing times
        timing_analysis = self._analyze_processing_times(results)
        
        # Analyze technical issues
        issues_analysis = self._analyze_technical_issues(results)
        
        # Get top deletion reasons
        deletion_reasons = self._analyze_deletion_reasons(results.get('Delete', []))
        
        # Session metadata
        session_start = session_info.get('start_time', datetime.now())
        session_end = session_info.get('end_time', datetime.now())
        processing_time = session_end - session_start if isinstance(session_start, datetime) else timedelta(0)
        
        return {
            'summary': {
                'total_files': total_files,
                'keep_count': keep_count,
                'delete_count': delete_count,
                'review_count': review_count,
                'failed_count': failed_count,
                'keep_pct': keep_pct,
                'delete_pct': delete_pct,
                'review_pct': review_pct,
                'failed_pct': failed_pct,
                'processing_time': str(processing_time),
                'session_start': session_start.strftime('%Y-%m-%d %H:%M:%S') if isinstance(session_start, datetime) else 'Unknown',
                'session_end': session_end.strftime('%Y-%m-%d %H:%M:%S') if isinstance(session_end, datetime) else 'Unknown'
            },
            'quality_analysis': quality_analysis,
            'timing_analysis': timing_analysis,
            'issues_analysis': issues_analysis,
            'deletion_reasons': deletion_reasons,
            'results': results,
            'session_info': session_info
        }
    
    def _analyze_quality_scores(self, results: Dict) -> Dict:
        """Analyze quality score distribution"""
        
        all_scores = {'blur': [], 'exposure': [], 'composition': [], 'overall': []}
        
        for decision, items in results.items():
            if not isinstance(items, list):
                continue
                
            for item in items:
                if hasattr(item, 'metrics'):
                    metrics = item.metrics
                    all_scores['blur'].append(getattr(metrics, 'blur_score', 0))
                    all_scores['exposure'].append(getattr(metrics, 'exposure_score', 0))
                    all_scores['composition'].append(getattr(metrics, 'composition_score', 0))
                    all_scores['overall'].append(getattr(metrics, 'overall_quality', 0))
        
        # Calculate statistics for each metric
        analysis = {}
        for metric, scores in all_scores.items():
            if scores:
                analysis[metric] = {
                    'mean': np.mean(scores),
                    'median': np.median(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'scores': scores  # For histograms
                }
            else:
                analysis[metric] = {
                    'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'scores': []
                }
        
        return analysis
    
    def _analyze_processing_times(self, results: Dict) -> Dict:
        """Analyze processing time statistics"""
        
        processing_times = []
        
        for decision, items in results.items():
            if not isinstance(items, list):
                continue
                
            for item in items:
                if hasattr(item, 'processing_ms'):
                    processing_times.append(item.processing_ms)
        
        if processing_times:
            return {
                'mean_ms': np.mean(processing_times),
                'median_ms': np.median(processing_times),
                'min_ms': np.min(processing_times),
                'max_ms': np.max(processing_times),
                'total_ms': np.sum(processing_times),
                'times': processing_times
            }
        else:
            return {
                'mean_ms': 0, 'median_ms': 0, 'min_ms': 0, 'max_ms': 0, 'total_ms': 0, 'times': []
            }
    
    def _analyze_technical_issues(self, results: Dict) -> Dict:
        """Analyze common technical issues"""
        
        issue_counts = {}
        
        for decision, items in results.items():
            if not isinstance(items, list):
                continue
                
            for item in items:
                if hasattr(item, 'issues') and item.issues:
                    for issue in item.issues:
                        issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Sort by frequency
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'issue_counts': dict(sorted_issues),
            'total_issues': sum(issue_counts.values()),
            'most_common': sorted_issues[:5] if sorted_issues else []
        }
    
    def _analyze_deletion_reasons(self, delete_items: List) -> List[Dict]:
        """Analyze reasons for deletion decisions"""
        
        deletion_analysis = []
        
        for item in delete_items:
            if not hasattr(item, 'filepath'):
                continue
                
            analysis = {
                'filename': item.filepath.name,
                'confidence': getattr(item, 'confidence', 0),
                'issues': getattr(item, 'issues', []),
                'quality_scores': {}
            }
            
            # Get quality scores if available
            if hasattr(item, 'metrics'):
                metrics = item.metrics
                analysis['quality_scores'] = {
                    'blur': getattr(metrics, 'blur_score', 0),
                    'exposure': getattr(metrics, 'exposure_score', 0),
                    'composition': getattr(metrics, 'composition_score', 0),
                    'overall': getattr(metrics, 'overall_quality', 0)
                }
            
            deletion_analysis.append(analysis)
        
        # Sort by confidence (highest confidence deletions first)
        deletion_analysis.sort(key=lambda x: x['confidence'], reverse=True)
        
        return deletion_analysis
    
    def _generate_html_content(self, report_data: Dict, include_thumbnails: bool) -> str:
        """Generate the complete HTML content"""
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Photo Culling Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f7;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        
        .header .subtitle {
            margin-top: 10px;
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .section {
            padding: 30px 40px;
            border-bottom: 1px solid #eee;
        }
        
        .section:last-child {
            border-bottom: none;
        }
        
        .section h2 {
            margin-top: 0;
            color: #2c3e50;
            font-size: 1.8em;
            font-weight: 600;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            border-color: #667eea;
            transform: translateY(-2px);
        }
        
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .stat-label {
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .keep { color: #27ae60; }
        .delete { color: #e74c3c; }
        .review { color: #f39c12; }
        .failed { color: #95a5a6; }
        
        .chart-container {
            margin: 30px 0;
            height: 400px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .issues-list {
            background: #fff5f5;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .issue-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #ffe0e0;
        }
        
        .issue-item:last-child {
            border-bottom: none;
        }
        
        .issue-count {
            background: #e74c3c;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        
        .deletion-list {
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        
        .deletion-item {
            padding: 15px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .deletion-item:last-child {
            border-bottom: none;
        }
        
        .deletion-item:hover {
            background-color: #f8f9fa;
        }
        
        .filename {
            font-weight: bold;
            color: #2c3e50;
        }
        
        .confidence {
            background: #3498db;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
        }
        
        .quality-scores {
            display: flex;
            gap: 10px;
            margin-top: 5px;
        }
        
        .score {
            background: #ecf0f1;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8em;
        }
        
        .metadata {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
            overflow-x: auto;
        }
        
        @media (max-width: 768px) {
            body { padding: 10px; }
            .section { padding: 20px; }
            .stats-grid { grid-template-columns: 1fr; }
        }
    </style>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>📸 Photo Culling Report</h1>
            <div class="subtitle">Session completed on {session_date}</div>
        </div>
        
        <!-- Summary Section -->
        <div class="section">
            <h2>📊 Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{total_files}</div>
                    <div class="stat-label">Total Images</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value keep">{keep_count}</div>
                    <div class="stat-label">Keep ({keep_pct:.1f}%)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value delete">{delete_count}</div>
                    <div class="stat-label">Delete ({delete_pct:.1f}%)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value review">{review_count}</div>
                    <div class="stat-label">Review ({review_pct:.1f}%)</div>
                </div>
            </div>
            
            <p><strong>Processing Time:</strong> {processing_time}</p>
            <p><strong>Average per Image:</strong> {avg_time_per_image:.1f}ms</p>
        </div>
        
        <!-- Quality Analysis -->
        <div class="section">
            <h2>🎯 Quality Analysis</h2>
            <div id="qualityChart" class="chart-container"></div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{avg_blur_score:.2f}</div>
                    <div class="stat-label">Average Blur Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_exposure_score:.2f}</div>
                    <div class="stat-label">Average Exposure Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_composition_score:.2f}</div>
                    <div class="stat-label">Average Composition Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_overall_score:.2f}</div>
                    <div class="stat-label">Average Overall Score</div>
                </div>
            </div>
        </div>
        
        <!-- Issues Analysis -->
        <div class="section">
            <h2>⚠️ Common Issues</h2>
            <div class="issues-list">
                {issues_html}
            </div>
            <div id="issuesChart" class="chart-container"></div>
        </div>
        
        <!-- Deletion Analysis -->
        <div class="section">
            <h2>🗑️ Deletion Candidates</h2>
            <p>Images marked for deletion, sorted by confidence:</p>
            <div class="deletion-list">
                {deletion_items_html}
            </div>
        </div>
        
        <!-- Session Info -->
        <div class="section">
            <h2>🔧 Session Information</h2>
            <div class="metadata">{session_metadata}</div>
        </div>
    </div>
    
    <script>
        // Quality distribution chart
        var qualityData = [
            {{
                x: {blur_scores},
                type: 'histogram',
                name: 'Blur Score',
                opacity: 0.7,
                marker: {{ color: '#3498db' }}
            }},
            {{
                x: {exposure_scores},
                type: 'histogram',
                name: 'Exposure Score', 
                opacity: 0.7,
                marker: {{ color: '#2ecc71' }}
            }},
            {{
                x: {composition_scores},
                type: 'histogram',
                name: 'Composition Score',
                opacity: 0.7,
                marker: {{ color: '#e74c3c' }}
            }},
            {{
                x: {overall_scores},
                type: 'histogram',
                name: 'Overall Quality',
                opacity: 0.7,
                marker: {{ color: '#9b59b6' }}
            }}
        ];
        
        var qualityLayout = {{
            title: 'Quality Score Distribution',
            xaxis: {{ title: 'Score (0-1)' }},
            yaxis: {{ title: 'Number of Images' }},
            barmode: 'overlay',
            showlegend: true
        }};
        
        Plotly.newPlot('qualityChart', qualityData, qualityLayout, {{responsive: true}});
        
        // Issues pie chart
        var issuesData = [{{
            values: {issue_counts_values},
            labels: {issue_counts_labels},
            type: 'pie',
            hole: 0.3,
            marker: {{
                colors: ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6', '#1abc9c']
            }}
        }}];
        
        var issuesLayout = {{
            title: 'Distribution of Technical Issues'
        }};
        
        if ({issue_counts_values}.length > 0) {{
            Plotly.newPlot('issuesChart', issuesData, issuesLayout, {{responsive: true}});
        }} else {{
            document.getElementById('issuesChart').innerHTML = '<p style="text-align: center; color: #2ecc71; font-size: 1.2em;">🎉 No technical issues detected!</p>';
        }}
    </script>
</body>
</html>
"""
        
        # Prepare template variables
        summary = report_data['summary']
        quality = report_data['quality_analysis']
        issues = report_data['issues_analysis']
        timing = report_data['timing_analysis']
        deletions = report_data['deletion_reasons']
        
        # Generate issues HTML
        issues_html = ""
        for issue, count in issues['most_common']:
            issues_html += f'<div class="issue-item"><span>{issue}</span><span class="issue-count">{count}</span></div>'
        
        if not issues_html:
            issues_html = '<div class="issue-item"><span>No issues detected</span><span class="issue-count" style="background: #27ae60;">0</span></div>'
        
        # Generate deletion items HTML
        deletion_items_html = ""
        for item in deletions[:20]:  # Show top 20
            issues_str = ', '.join(item['issues']) if item['issues'] else 'No specific issues'
            scores_html = ""
            for metric, score in item['quality_scores'].items():
                scores_html += f'<span class="score">{metric}: {score:.2f}</span>'
            
            deletion_items_html += f"""
            <div class="deletion-item">
                <div>
                    <div class="filename">{item['filename']}</div>
                    <div style="font-size: 0.9em; color: #666; margin-top: 4px;">{issues_str}</div>
                    <div class="quality-scores">{scores_html}</div>
                </div>
                <div class="confidence">{item['confidence']:.2f}</div>
            </div>
            """
        
        # Calculate average processing time per image
        avg_time_per_image = timing['mean_ms'] if timing['mean_ms'] > 0 else 0
        
        # Prepare chart data
        blur_scores = quality['blur']['scores']
        exposure_scores = quality['exposure']['scores'] 
        composition_scores = quality['composition']['scores']
        overall_scores = quality['overall']['scores']
        
        issue_counts_labels = list(issues['issue_counts'].keys())
        issue_counts_values = list(issues['issue_counts'].values())
        
        # Session metadata
        session_metadata = json.dumps(report_data['session_info'], indent=2, default=str)
        
        # Fill template
        html_content = html_template.format(
            session_date=summary['session_end'],
            total_files=summary['total_files'],
            keep_count=summary['keep_count'],
            delete_count=summary['delete_count'],
            review_count=summary['review_count'],
            keep_pct=summary['keep_pct'],
            delete_pct=summary['delete_pct'],
            review_pct=summary['review_pct'],
            processing_time=summary['processing_time'],
            avg_time_per_image=avg_time_per_image,
            avg_blur_score=quality['blur']['mean'],
            avg_exposure_score=quality['exposure']['mean'],
            avg_composition_score=quality['composition']['mean'],
            avg_overall_score=quality['overall']['mean'],
            issues_html=issues_html,
            deletion_items_html=deletion_items_html,
            session_metadata=session_metadata,
            blur_scores=json.dumps(blur_scores),
            exposure_scores=json.dumps(exposure_scores),
            composition_scores=json.dumps(composition_scores),
            overall_scores=json.dumps(overall_scores),
            issue_counts_labels=json.dumps(issue_counts_labels),
            issue_counts_values=json.dumps(issue_counts_values)
        )
        
        return html_content
    
    def generate_json_report(self, results: Dict, session_info: Dict, output_path: Path):
        """Generate detailed JSON report"""
        
        try:
            report_data = self._prepare_report_data(results, session_info)
            
            # Convert numpy arrays to lists for JSON serialization
            self._convert_numpy_to_lists(report_data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"JSON report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate JSON report: {e}")
            raise
    
    def _convert_numpy_to_lists(self, data: Any):
        """Recursively convert numpy arrays to lists for JSON serialization"""
        
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = self._convert_numpy_to_lists(value)
        elif isinstance(data, list):
            return [self._convert_numpy_to_lists(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        
        return data
    
    def generate_csv_summary(self, results: Dict, output_path: Path):
        """Generate CSV summary of results"""
        
        try:
            import csv
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'filename', 'decision', 'confidence', 'blur_score',
                    'exposure_score', 'composition_score', 'overall_quality',
                    'issues', 'processing_ms'
                ])
                
                # Write data
                for decision, items in results.items():
                    if not isinstance(items, list):
                        continue
                        
                    for item in items:
                        if not hasattr(item, 'filepath'):
                            continue
                            
                        row = [
                            item.filepath.name,
                            decision,
                            getattr(item, 'confidence', 0),
                            getattr(item.metrics, 'blur_score', 0) if hasattr(item, 'metrics') else 0,
                            getattr(item.metrics, 'exposure_score', 0) if hasattr(item, 'metrics') else 0,
                            getattr(item.metrics, 'composition_score', 0) if hasattr(item, 'metrics') else 0,
                            getattr(item.metrics, 'overall_quality', 0) if hasattr(item, 'metrics') else 0,
                            '; '.join(getattr(item, 'issues', [])),
                            getattr(item, 'processing_ms', 0)
                        ]
                        writer.writerow(row)
            
            self.logger.info(f"CSV summary generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate CSV summary: {e}")
            raise