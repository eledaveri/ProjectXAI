import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd

def plot_model_comparison(y_test, y_pred_full, y_pred_reduced, classes, 
                         save_path="results/model_comparison.png"):
    """
    Crea un grafico completo per confrontare modello completo vs ridotto.
    
    Args:
        y_test: True labels
        y_pred_full: Predictions from full model (110 features)
        y_pred_reduced: Predictions from reduced model (5 features)
        classes: List of class names
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(18, 12))
    
    # ============================================
    # 1. CONFUSION MATRICES (prima riga)
    # ============================================
    
    # Full model confusion matrix
    ax1 = plt.subplot(3, 2, 1)
    cm_full = confusion_matrix(y_test, y_pred_full)
    sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues', 
               xticklabels=classes, yticklabels=classes, ax=ax1,
               cbar_kws={'label': 'Count'})
    ax1.set_title('Full Model (110 features)\nConfusion Matrix', 
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=11)
    ax1.set_xlabel('Predicted Label', fontsize=11)
    
    # Reduced model confusion matrix
    ax2 = plt.subplot(3, 2, 2)
    cm_reduced = confusion_matrix(y_test, y_pred_reduced)
    sns.heatmap(cm_reduced, annot=True, fmt='d', cmap='Greens', 
               xticklabels=classes, yticklabels=classes, ax=ax2,
               cbar_kws={'label': 'Count'})
    ax2.set_title('Reduced Model (5 features)\nConfusion Matrix', 
                  fontsize=13, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=11)
    ax2.set_xlabel('Predicted Label', fontsize=11)
    
    # ============================================
    # 2. F1-SCORE PER CLASSE (seconda riga)
    # ============================================
    ax3 = plt.subplot(3, 2, 3)
    
    # Calcola F1 per entrambi i modelli
    _, _, f1_full, support = precision_recall_fscore_support(
        y_test, y_pred_full, labels=range(len(classes)), zero_division=0
    )
    _, _, f1_reduced, _ = precision_recall_fscore_support(
        y_test, y_pred_reduced, labels=range(len(classes)), zero_division=0
    )
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, f1_full, width, label='Full (110 feat)', 
                    alpha=0.8, color='steelblue')
    bars2 = ax3.bar(x + width/2, f1_reduced, width, label='Reduced (5 feat)', 
                    alpha=0.8, color='seagreen')
    
    # Aggiungi valori sopra le barre
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    ax3.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax3.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax3.set_title('F1-Score Comparison by Class', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes)
    ax3.legend()
    ax3.set_ylim(0, 1.1)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=1)
    
    # Aggiungi sample support
    ax3_twin = ax3.twiny()
    ax3_twin.set_xlim(ax3.get_xlim())
    ax3_twin.set_xticks(x)
    ax3_twin.set_xticklabels([f'n={s}' for s in support], fontsize=9, color='gray')
    
    # ============================================
    # 3. PRECISION E RECALL (seconda riga, destra)
    # ============================================
    ax4 = plt.subplot(3, 2, 4)
    
    prec_full, rec_full, _, _ = precision_recall_fscore_support(
        y_test, y_pred_full, labels=range(len(classes)), zero_division=0
    )
    prec_reduced, rec_reduced, _, _ = precision_recall_fscore_support(
        y_test, y_pred_reduced, labels=range(len(classes)), zero_division=0
    )
    
    x = np.arange(len(classes))
    width = 0.2
    
    ax4.bar(x - width*1.5, prec_full, width, label='Precision (Full)', 
            alpha=0.8, color='#1f77b4')
    ax4.bar(x - width*0.5, rec_full, width, label='Recall (Full)', 
            alpha=0.8, color='#ff7f0e')
    ax4.bar(x + width*0.5, prec_reduced, width, label='Precision (Reduced)', 
            alpha=0.8, color='#2ca02c')
    ax4.bar(x + width*1.5, rec_reduced, width, label='Recall (Reduced)', 
            alpha=0.8, color='#d62728')
    
    ax4.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax4.set_title('Precision & Recall by Class', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(classes)
    ax4.legend(fontsize=9, loc='lower left')
    ax4.set_ylim(0, 1.1)
    ax4.grid(axis='y', alpha=0.3)
    
    # ============================================
    # 4. METRICHE AGGREGATE (terza riga, sinistra)
    # ============================================
    ax5 = plt.subplot(3, 2, 5)
    
    # Calcola metriche aggregate
    metrics_data = []
    for model_name, y_pred in [('Full\n(110 feat)', y_pred_full), 
                                ('Reduced\n(5 feat)', y_pred_reduced)]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=range(len(classes)), average='weighted', zero_division=0
        )
        accuracy = (y_test == y_pred).mean()
        
        metrics_data.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    x_agg = np.arange(len(metrics_data))
    width_agg = 0.2
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics_to_plot):
        offset = width_agg * (i - len(metrics_to_plot)/2 + 0.5)
        bars = ax5.bar(x_agg + offset, df_metrics[metric], width_agg, 
                      label=metric, alpha=0.8, color=colors[i])
        
        # Valori sopra le barre
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax5.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax5.set_title('Weighted Average Metrics', fontsize=13, fontweight='bold')
    ax5.set_xticks(x_agg)
    ax5.set_xticklabels(df_metrics['Model'])
    ax5.legend(fontsize=9)
    ax5.set_ylim(0, 1.1)
    ax5.grid(axis='y', alpha=0.3)
    ax5.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=1)
    
    # ============================================
    # 5. DIFFERENZE (terza riga, destra)
    # ============================================
    ax6 = plt.subplot(3, 2, 6)
    
    # Calcola differenze (Reduced - Full)
    diff_f1 = f1_reduced - f1_full
    diff_prec = prec_reduced - prec_full
    diff_rec = rec_reduced - rec_full
    
    x = np.arange(len(classes))
    width = 0.25
    
    bars1 = ax6.bar(x - width, diff_prec, width, label='Δ Precision', 
                   alpha=0.8, color='#ff7f0e')
    bars2 = ax6.bar(x, diff_rec, width, label='Δ Recall', 
                   alpha=0.8, color='#2ca02c')
    bars3 = ax6.bar(x + width, diff_f1, width, label='Δ F1-Score', 
                   alpha=0.8, color='#d62728')
    
    # Aggiungi valori sopra/sotto le barre
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            label_y = height + 0.01 if height >= 0 else height - 0.01
            va = 'bottom' if height >= 0 else 'top'
            ax6.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{height:+.2f}',
                    ha='center', va=va, fontsize=8)
    
    ax6.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Difference (Reduced - Full)', fontsize=11, fontweight='bold')
    ax6.set_title('Performance Differences\n(Positive = Reduced Better)', 
                  fontsize=13, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(classes)
    ax6.legend(fontsize=9)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.grid(axis='y', alpha=0.3)
    
    # Limiti dinamici per mostrare bene le differenze
    max_diff = max(abs(diff_prec.max()), abs(diff_rec.max()), abs(diff_f1.max()))
    ax6.set_ylim(-max_diff*1.3, max_diff*1.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved model comparison: {save_path}")


def create_comparison_summary(y_test, y_pred_full, y_pred_reduced, classes,
                             save_path="results/comparison_summary.txt"):
    """Crea un report testuale del confronto."""
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("Full Model (110 features) vs Reduced Model (5 features)\n")
        f.write("="*80 + "\n\n")
        
        # Metriche aggregate
        for model_name, y_pred in [('FULL MODEL (110 features)', y_pred_full),
                                    ('REDUCED MODEL (5 features)', y_pred_reduced)]:
            f.write(f"\n{model_name}\n")
            f.write("-"*80 + "\n")
            
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred, labels=range(len(classes)), zero_division=0
            )
            accuracy = (y_test == y_pred).mean()
            
            # Aggregate
            prec_avg, rec_avg, f1_avg, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division=0
            )
            
            f.write(f"\nWeighted Average Metrics:\n")
            f.write(f"  Accuracy:  {accuracy:.4f}\n")
            f.write(f"  Precision: {prec_avg:.4f}\n")
            f.write(f"  Recall:    {rec_avg:.4f}\n")
            f.write(f"  F1-Score:  {f1_avg:.4f}\n")
            
            # Per classe
            f.write(f"\nPer-Class Metrics:\n")
            f.write(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
            f.write("-"*60 + "\n")
            
            for i, cls in enumerate(classes):
                f.write(f"{cls:<12} {precision[i]:<12.4f} {recall[i]:<12.4f} "
                       f"{f1[i]:<12.4f} {support[i]:<10}\n")
        
        # Confronto
        f.write(f"\n\n{'='*80}\n")
        f.write(f"PERFORMANCE DIFFERENCES (Reduced - Full)\n")
        f.write(f"{'='*80}\n\n")
        
        prec_full, rec_full, f1_full, _ = precision_recall_fscore_support(
            y_test, y_pred_full, labels=range(len(classes)), zero_division=0
        )
        prec_red, rec_red, f1_red, _ = precision_recall_fscore_support(
            y_test, y_pred_reduced, labels=range(len(classes)), zero_division=0
        )
        
        acc_full = (y_test == y_pred_full).mean()
        acc_red = (y_test == y_pred_reduced).mean()
        
        f.write(f"Overall Accuracy Difference: {acc_red - acc_full:+.4f}\n")
        f.write(f"  Full:    {acc_full:.4f}\n")
        f.write(f"  Reduced: {acc_red:.4f}\n\n")
        
        f.write(f"Per-Class F1-Score Differences:\n")
        f.write(f"{'Class':<12} {'Full F1':<12} {'Reduced F1':<12} {'Difference':<12}\n")
        f.write("-"*50 + "\n")
        
        for i, cls in enumerate(classes):
            diff = f1_red[i] - f1_full[i]
            f.write(f"{cls:<12} {f1_full[i]:<12.4f} {f1_red[i]:<12.4f} {diff:+12.4f}\n")
        
        # Interpretazione
        f.write(f"\n\n{'='*80}\n")
        f.write(f"INTERPRETATION\n")
        f.write(f"{'='*80}\n\n")
        
        avg_diff = np.mean(np.abs(f1_red - f1_full))
        
        if avg_diff < 0.05:
            f.write("   The reduced model performs similarly to the full model.\n")
            f.write("   This confirms that the 5 selected features capture most of the\n")
            f.write("   discriminative information.\n")
        elif acc_red > acc_full:
            f.write("   The reduced model slightly outperforms the full model.\n")
            f.write("   This suggests the full model may have some overfitting,\n")
            f.write("   and feature selection improved generalization.\n")
        else:
            f.write("   The reduced model shows some performance degradation.\n")
            f.write("   Consider including more features or investigating which\n")
            f.write("   classes are most affected.\n")
    
    print(f" Saved comparison summary: {save_path}")