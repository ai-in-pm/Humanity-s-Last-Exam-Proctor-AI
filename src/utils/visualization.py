import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
import numpy as np

def plot_performance_metrics(metrics: Dict):
    """Create visualizations for performance metrics."""
    # Set up the style
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(15, 10))

    # 1. Overall Performance Metrics
    ax1 = plt.subplot(2, 2, 1)
    metrics_data = {
        'Accuracy': metrics['accuracy'],
        'Calibration Error': metrics['confidence_calibration_error'],
        'Avg Confidence': metrics['average_confidence'],
        'Hallucination Rate': metrics['hallucination_rate']
    }
    plt.bar(metrics_data.keys(), metrics_data.values())
    plt.title('Overall Performance Metrics')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

    # 2. Subject Area Performance
    ax2 = plt.subplot(2, 2, 2)
    subjects = metrics['subject_area_performance']
    plt.bar(subjects.keys(), subjects.values())
    plt.title('Performance by Subject Area')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

    # 3. Difficulty Level Performance
    ax3 = plt.subplot(2, 2, 3)
    difficulties = metrics['difficulty_performance']
    plt.bar(difficulties.keys(), difficulties.values())
    plt.title('Performance by Difficulty Level')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

    # 4. Question Type Performance
    ax4 = plt.subplot(2, 2, 4)
    q_types = metrics['question_type_performance']
    plt.bar(q_types.keys(), q_types.values())
    plt.title('Performance by Question Type')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig

def plot_confidence_calibration(bin_accuracies: List[float], 
                              bin_confidences: List[float],
                              bin_counts: List[int]):
    """Plot confidence calibration curve."""
    plt.figure(figsize=(10, 6))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    
    # Actual calibration curve
    plt.scatter(bin_confidences, bin_accuracies, 
               s=[count/10 for count in bin_counts],
               alpha=0.6, label='Model Calibration')
    
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Confidence Calibration Curve')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()

def create_performance_heatmap(subject_areas: List[str], 
                             difficulty_levels: List[str],
                             performance_matrix: np.ndarray):
    """Create a heatmap showing performance across subjects and difficulty levels."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(performance_matrix,
                xticklabels=difficulty_levels,
                yticklabels=subject_areas,
                annot=True,
                fmt='.2f',
                cmap='YlOrRd')
    
    plt.title('Performance Heatmap: Subjects vs Difficulty')
    plt.xlabel('Difficulty Level')
    plt.ylabel('Subject Area')
    
    return plt.gcf()

def plot_performance_metrics_new(metrics: dict) -> plt.Figure:
    """Create a visualization of performance metrics."""
    # Set up the style
    sns.set_theme(style="whitegrid")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy by subject area
    if 'accuracy_by_subject' in metrics:
        subjects = list(metrics['accuracy_by_subject'].keys())
        accuracies = list(metrics['accuracy_by_subject'].values())
        
        sns.barplot(x=accuracies, y=subjects, ax=ax1)
        ax1.set_title('Accuracy by Subject Area')
        ax1.set_xlabel('Accuracy')
    
    # Confidence calibration
    if 'confidence_scores' in metrics and 'correctness' in metrics:
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(confidence_bins)-1):
            mask = (np.array(metrics['confidence_scores']) >= confidence_bins[i]) & \
                   (np.array(metrics['confidence_scores']) < confidence_bins[i+1])
            if np.sum(mask) > 0:
                bin_accuracies.append(np.mean(np.array(metrics['correctness'])[mask]))
                bin_counts.append(np.sum(mask))
        
        ax2.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
        ax2.scatter(confidence_bins[:-1] + 0.05, bin_accuracies, 
                   s=[c*50 for c in bin_counts], alpha=0.6)
        ax2.set_title('Confidence Calibration')
        ax2.set_xlabel('Predicted Confidence')
        ax2.set_ylabel('Actual Accuracy')
        ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_confidence_calibration_new(confidence_scores: list, correctness: list) -> plt.Figure:
    """Create a visualization of confidence calibration."""
    # Set up the style
    sns.set_theme(style="whitegrid")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    confidence_bins = np.linspace(0, 1, 11)
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(confidence_bins)-1):
        mask = (np.array(confidence_scores) >= confidence_bins[i]) & \
               (np.array(confidence_scores) < confidence_bins[i+1])
        if np.sum(mask) > 0:
            bin_accuracies.append(np.mean(np.array(correctness)[mask]))
            bin_counts.append(np.sum(mask))
    
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    ax.scatter(confidence_bins[:-1] + 0.05, bin_accuracies, 
               s=[c*50 for c in bin_counts], alpha=0.6)
    ax.set_title('Confidence Calibration')
    ax.set_xlabel('Predicted Confidence')
    ax.set_ylabel('Actual Accuracy')
    ax.legend()
    
    plt.tight_layout()
    return fig
