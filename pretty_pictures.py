import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as PLT
import matplotlib as mpl
import seaborn as sns
import numpy as NP
import pandas
from pathlib import Path

PLT.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300

BLUE = '#2E86C1'
RED = '#E74C3C'
COLORS = {'pythia': BLUE, 'bloomz': RED}

_VL = [
    ['v1', 'V1 (SVO, no case)'],
    ['v2', 'V2 (VSO, no case)'],
    ['v3', 'V3 (SVO, +case)'],
    ['v4', 'V4 (VSO, +case)'],
]
VLABELS = {}
for _item in _VL:
    VLABELS[_item[0]] = _item[1]


class PictureMaker:

    def __init__(self, df, output_dir='figures/'):
        self.df = df
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self._fig_count = 0

    def plot_learning_curves(self):
        self._fig_count += 1
        fig, axes = PLT.subplots(1, 2, figsize=(12, 5), sharey=True)

        _model_list = ['pythia', 'bloomz']
        for idx in range(len(_model_list)):
            ax = axes[idx]
            model = _model_list[idx]
            sub = self.df[self.df['model'] == model]

            for v in ['v1', 'v2', 'v3', 'v4']:
                v_data = sub[sub['variant'] == v]
                grouped = v_data.groupby('shot_count')
                means = grouped['exact_match'].mean()
                sems = grouped['exact_match'].sem()

                ax.plot(means.index, means.values, 'o-', label=VLABELS[v])
                lower = means - 1.96 * sems
                upper = means + 1.96 * sems
                ax.fill_between(means.index, lower, upper, alpha=0.15)

            ax.set_xlabel('Number of shots')
            ax.set_xticks([0, 3, 8])
            ax.set_title(model.upper())
            ax.legend(fontsize=9)

        axes[0].set_ylabel('Exact Match Accuracy')
        PLT.tight_layout()
        _save_path = str(self.out) + '/learning_curves.png'
        PLT.savefig(_save_path)
        PLT.close()

    def plot_interaction(self):
        self._fig_count += 1
        fig, ax = PLT.subplots(figsize=(7, 5))
        summary = self.df.groupby(
            ['syntax', 'morphology']
        )['exact_match'].mean().reset_index()
        sns.pointplot(
            data=summary,
            x='syntax', y='exact_match',
            hue='morphology', ax=ax,
            markers=['o', 's'],
            palette=[BLUE, RED],
        )
        ax.set_ylabel('Exact Match Accuracy')
        ax.set_title('Syntax x Morphology Interaction')
        PLT.tight_layout()
        PLT.savefig(self.out / 'interaction_plot.png')
        PLT.close()

