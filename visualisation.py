"""
Module 6: Visualisation Suite
Publication-quality plots for dissertation figures.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


# Consistent style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

PALETTE = {
    'pythia': '#2E86C1',
    'bloomz': '#E74C3C',
}

VARIANT_LABELS = {
    'v1': 'V1 (SVO, no case)',
    'v2': 'V2 (VSO, no case)',
    'v3': 'V3 (SVO, +case)',
    'v4': 'V4 (VSO, +case)',
}


class Visualiser:
    """Generates all dissertation figures."""

    def __init__(
        self,
        results_df: pd.DataFrame,
        output_dir: str = 'figures/',
    ):
        self.df = results_df
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def plot_learning_curves(self):
        """
        Figure 1: Exact-match accuracy vs shot count
        for each model, with 95% CI bands.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                                 sharey=True)
        for ax, model in zip(axes, ['pythia', 'bloomz']):
            sub = self.df[self.df['model'] == model]
            for variant in ['v1', 'v2', 'v3', 'v4']:
                v_data = sub[sub['variant'] == variant]
                grouped = v_data.groupby('shot_count')
                means = grouped['exact_match'].mean()
                sems = grouped['exact_match'].sem()
                ax.plot(
                    means.index, means.values,
                    'o-', label=VARIANT_LABELS[variant],
                )
                ax.fill_between(
                    means.index,
                    means - 1.96 * sems,
                    means + 1.96 * sems,
                    alpha=0.15,
                )
            ax.set_xlabel('Number of shots')
            ax.set_xticks([0, 3, 8])
            ax.set_title(model.upper())
            ax.legend(fontsize=9)
        axes[0].set_ylabel('Exact Match Accuracy')
        plt.tight_layout()
        plt.savefig(self.out / 'learning_curves.png')
        plt.close()

    def plot_interaction(self):
        """
        Figure 2: Interaction plot for syntax x morphology.
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        summary = self.df.groupby(
            ['syntax', 'morphology']
        )['exact_match'].mean().reset_index()
        sns.pointplot(
            data=summary,
            x='syntax', y='exact_match',
            hue='morphology', ax=ax,
            markers=['o', 's'],
            palette=['#2E86C1', '#E74C3C'],
        )
        ax.set_ylabel('Exact Match Accuracy')
        ax.set_title('Syntax x Morphology Interaction')
        plt.tight_layout()
        plt.savefig(self.out / 'interaction_plot.png')
        plt.close()

    def plot_heatmap(self):
        """
        Figure 3: Heatmap of accuracy across all conditions.
        """
        pivot = self.df.pivot_table(
            index=['model', 'shot_count'],
            columns='variant',
            values='exact_match',
            aggfunc='mean',
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            pivot, annot=True, fmt='.2f',
            cmap='YlGnBu', ax=ax,
            vmin=0, vmax=1,
        )
        ax.set_title('Accuracy Across All Conditions')
        plt.tight_layout()
        plt.savefig(self.out / 'heatmap.png')
        plt.close()

    def plot_error_analysis(self):
        """
        Figure 4: Stacked bar chart of error types.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        err = self.df.groupby(
            ['model', 'variant']
        ).agg({
            'word_order_correct': 'mean',
            'case_marking_correct': 'mean',
        }).reset_index()
        # Plot grouped bars
        x = np.arange(len(err))
        width = 0.35
        ax.bar(x - width/2,
               err['word_order_correct'],
               width, label='Word Order',
               color='#2E86C1')
        ax.bar(x + width/2,
               err['case_marking_correct'],
               width, label='Case Marking',
               color='#E74C3C')
        ax.set_xticks(x)
        ax.set_xticklabels([
            f"{r['model']}\n{r['variant']}"
            for _, r in err.iterrows()
        ], fontsize=8)
        ax.set_ylabel('Feature Accuracy')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.out / 'error_analysis.png')
        plt.close()

    def generate_all(self):
        """Generate all dissertation figures."""
        self.plot_learning_curves()
        self.plot_interaction()
        self.plot_heatmap()
        self.plot_error_analysis()
        print(f'All figures saved to {self.out}/')
