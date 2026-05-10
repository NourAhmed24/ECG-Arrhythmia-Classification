#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
    roc_auc_score
)
import tensorflow as tf

# Global Constants
CLASS_NAMES   = ['N', 'S', 'V', 'F', 'Q']
COLORS_MAP    = {
    'N': '#2196F3', 'S': '#4CAF50', 'V': '#F44336',
    'F': '#FF9800', 'Q': '#9C27B0'
}
CLASS_DESC = {
    'N': 'Normal',
    'S': 'Supraventricular',
    'V': 'Ventricular',
    'F': 'Fusion',
    'Q': 'Unknown/PM',
}

def evaluate_model(model, X_test, y_test, results_dir='results'):
    """
    Main entry point for model evaluation. Generates plots, metrics, and reports.
    """
    os.makedirs(results_dir, exist_ok=True)
    print(" FULL EVALUATION ")
    
    # Model Prediction
    probs  = model.predict(X_test, verbose=1)           # Shape: (N, 5)
    y_pred = np.argmax(probs, axis=1)                    # Shape: (N,)

    # 1. Confusion Matrix
    print("\n Plotting Confusion Matrix...")
    _plot_confusion_matrix(y_test, y_pred, results_dir)

    # 2. Metrics Summary
    print("\n Computing Metrics Summary...")
    metrics = _compute_metrics(y_test, y_pred)
    _plot_metrics_summary(metrics, results_dir)
    _print_metrics_table(metrics)

    # 3. Classification Report
    _save_classification_report(y_test, y_pred, metrics, results_dir)

    # 4. Analyze Clinical Errors
    print("\n Analyzing Clinical Errors:")
    _analyze_clinical_errors(y_test, y_pred)

    return metrics

def _plot_confusion_matrix(y_true, y_pred, results_dir):

    cm      = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Confusion Matrix — ECG Arrhythmia Classification',
                 fontsize=14, fontweight='bold')

    labels = [f"{n}\n{CLASS_DESC[n]}" for n in CLASS_NAMES]

    # Plot 1: Counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=labels,
        ax=axes[0], linewidths=0.5, linecolor='#cccccc',
        annot_kws={'size': 10}
    )
    axes[0].set_title('Counts', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)

    # Plot 2: Normalized
    annot_norm = np.array([
        [f"{v:.1%}" for v in row] for row in cm_norm
    ])
    sns.heatmap(
        cm_norm, annot=annot_norm, fmt='', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=labels,
        ax=axes[1], linewidths=0.5, linecolor='#cccccc',
        vmin=0, vmax=1, annot_kws={'size': 9}
    )
    axes[1].set_title('Row-Normalized (= Recall per class)',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)

    # Highlight High-Risk Clinical Errors (V→N, F→N) in Red
    danger_pairs = [(2, 0), (3, 0)]  # (true_idx, pred_idx)
    for ti, pi in danger_pairs:
        for ax in axes:
            ax.add_patch(plt.Rectangle(
                (pi, ti), 1, 1, fill=False, edgecolor='red', lw=2.5,
                linestyle='--'
            ))

    plt.tight_layout()
    path = f'{results_dir}/confusion_matrix.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path}")

def _compute_metrics(y_true, y_pred):
    #Computes Precision, Recall, and F1 for each class and calculates averages.
    precision_per = precision_score(y_true, y_pred, average=None,
                                    labels=range(len(CLASS_NAMES)),
                                    zero_division=0)
    recall_per    = recall_score(y_true, y_pred, average=None,
                                 labels=range(len(CLASS_NAMES)),
                                 zero_division=0)
    f1_per        = f1_score(y_true, y_pred, average=None,
                               labels=range(len(CLASS_NAMES)),
                               zero_division=0)

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            'precision': float(precision_per[i]),
            'recall':    float(recall_per[i]),
            'f1':        float(f1_per[i]),
            'support':   int(np.sum(y_true == i)),
        }

    overall_acc   = float(np.mean(y_true == y_pred))
    macro_f1      = float(f1_score(y_true, y_pred, average='macro',  zero_division=0))
    weighted_f1   = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    macro_prec    = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    macro_recall  = float(recall_score(y_true, y_pred, average='macro', zero_division=0))

    return {
        'per_class':       per_class,
        'overall_acc':      overall_acc,
        'macro_f1':        macro_f1,
        'weighted_f1':     weighted_f1,
        'macro_precision': macro_prec,
        'macro_recall':    macro_recall,
    }

def _print_metrics_table(metrics):
    """Prints a clean CLI table of the results."""
    print(f"  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("─"*65)
    for name in CLASS_NAMES:
        m = metrics['per_class'][name]
        # Flag low recall for critical classes
        flag = " ⚠" if name in ('V', 'F') and m['recall'] < 0.70 else ""
        print(f"  {name:<8} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>10,}{flag}")
    print("─"*65)
    print(f"  {'Accuracy':<8}                         {metrics['overall_acc']:>10.4f}")
    print(f"  {'MacroF1':<8}                         {metrics['macro_f1']:>10.4f}")
    print(f"  {'WgtdF1':<8}                         {metrics['weighted_f1']:>10.4f}")
    print("─"*65)

def _plot_metrics_summary(metrics, results_dir):
    """Generates a bar chart comparison of metrics across classes."""
    names  = CLASS_NAMES
    prec   = [metrics['per_class'][n]['precision'] for n in names]
    rec    = [metrics['per_class'][n]['recall']    for n in names]
    f1     = [metrics['per_class'][n]['f1']        for n in names]

    x    = np.arange(len(names))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))

    bars_p = ax.bar(x - w,   prec, w, label='Precision', color='#42A5F5', alpha=0.85)
    bars_r = ax.bar(x,       rec,  w, label='Recall',    color='#66BB6A', alpha=0.85)
    bars_f = ax.bar(x + w,   f1,   w, label='F1-Score',  color='#FFA726', alpha=0.85)

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=7.5)

    # Threshold line at 0.70
    ax.axhline(0.70, color='red', linestyle='--', linewidth=1, alpha=0.6,
               label='0.70 Threshold')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n{CLASS_DESC[n]}" for n in names], fontsize=9)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.12)
    ax.set_title('Per-Class Metrics: Precision / Recall / F1',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Add Overall Summary Box
    txt = (f"Overall Acc:  {metrics['overall_acc']*100:.2f}%\n"
           f"Macro F1:     {metrics['macro_f1']*100:.2f}%\n"
           f"Weighted F1:  {metrics['weighted_f1']*100:.2f}%")
    ax.text(0.99, 0.97, txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=8.5, fontfamily='monospace',
            bbox=dict(facecolor='lightyellow', edgecolor='gray',
                      boxstyle='round', alpha=0.9))

    plt.tight_layout()
    path = f'{results_dir}/metrics_summary.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path}")

def _save_classification_report(y_true, y_pred, metrics, results_dir):
    """Saves the scikit-learn classification report to a text file."""
    report = classification_report(
        y_true, y_pred,
        target_names=[f"{n} ({CLASS_DESC[n]})" for n in CLASS_NAMES],
        digits=4
    )
    path = f'{results_dir}/classification_report.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write("ECG Arrhythmia Classification Report\n")
        f.write("="*40 + "\n")
        f.write(report)
        f.write("\n\nSummary Metrics:\n")
        f.write(f"  Overall Accuracy : {metrics['overall_acc']*100:.2f}%\n")
        f.write(f"  Macro F1         : {metrics['macro_f1']*100:.2f}%\n")
        f.write(f"  Weighted F1      : {metrics['weighted_f1']*100:.2f}%\n")
    print(f"  ✓ {path}")

def _analyze_clinical_errors(y_true, y_pred):
    """
    Identifies and prints errors based on medical severity.
    Critical: V→N or F→N (Missing life-threatening beats).
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))

    danger, other = [], []
    for ti in range(len(CLASS_NAMES)):
        for pi in range(len(CLASS_NAMES)):
            if ti == pi or cm[ti, pi] == 0:
                continue
            entry = {
                'true': CLASS_NAMES[ti], 'pred': CLASS_NAMES[pi],
                'count': int(cm[ti, pi]),
                'pct': cm[ti, pi] / (cm[ti].sum() + 1e-9) * 100
            }
            # Logic: If a dangerous arrhythmia (V or F) is predicted as Normal (N)
            if CLASS_NAMES[ti] in ('V', 'F') and CLASS_NAMES[pi] == 'N':
                danger.append(entry)
            else:
                other.append(entry)

    if danger:
        print(" CLINICALLY DANGEROUS ERRORS:")
        for e in sorted(danger, key=lambda x: x['count'], reverse=True):
            print(f"    {e['true']}→{e['pred']}: {e['count']:,} instances "
                  f"({e['pct']:.1f}% of true {e['true']})  <-- MISSED DANGEROUS BEAT!")
    else:
        print(" No dangerous errors (V/F classified as N) detected.")

    print(" General Errors:")
    for e in sorted(other, key=lambda x: x['count'], reverse=True)[:8]:
        print(f"    {e['true']}→{e['pred']}: {e['count']:,} ({e['pct']:.1f}%)")

# Execution Block
if __name__ == '__main__':
    from data_preprocessing import prepare_data
    print(" EVALUATE: ECG Arrhythmia ")
    _, X_test, _, y_test, _, le = prepare_data()
    print("Loading best model...")
    # Assumes best_model.keras exists in results/ from the training phase
    model = tf.keras.models.load_model('results/best_model.keras')
    metrics = evaluate_model(model, X_test, y_test)
    print("\n Evaluation complete!")


# In[ ]:




