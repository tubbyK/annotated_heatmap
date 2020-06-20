# as learned on https://towardsdatascience.com/annotated-heatmaps-in-5-simple-steps-cc2a0660a27d

import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

class annotated_heatmap():
    def __init__(self):
        self.df = None
        self.heatmap = None

    def run(self):
        self.df = self.load_data()
        self.df = self.handle_non_numeric()
        self.heatmap = self.gen_heatmap()
        plt.show()

    def load_data(self):
        data = datasets.load_boston()
        df = pd.DataFrame(data.data)
        df.columns = data.feature_names
        df['target'] = data.target
        return df

    def mk_corr_matrix(self):
        corr_matrix = self.df.corr()
        return corr_matrix

    def handle_non_numeric(self):
        # handle non-numeric columns
        if len(self.df.select_dtypes(exclude=[np.number]).columns) > 0:
            df_dummy = pd.get_dummies(self.df.select_dtypes(exclude='number'))
            df = pd.concat([self.df, df_dummy], axis=1)
        else:
            df = self.df
        return df

    def mk_upper_tri_mask(self, corr_matrix=None):
        if corr_matrix is None: corr_matrix = self.mk_corr_matrix()
        mask = np.zeros_like(corr_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        return mask

    def gen_heatmap(self, corr_matrix=None, mask=None):
        if corr_matrix is None: corr_matrix = self.mk_corr_matrix()
        if mask is None: mask = self.mk_upper_tri_mask()
        f, ax = plt.subplots(figsize=(11, 15))
        heatmap = sns.heatmap(corr_matrix,
                              mask=mask,
                              square=True,
                              linewidths=.5,
                              cmap= 'coolwarm',
                              cbar_kws={'shrink':.4,
                                        'ticks': [-1, -.5, 0, 0.5, 1]},
                              vmin = -1,
                              vmax = 1,
                              annot = True,
                              annot_kws = {'size': 8},
                              )
        ax.set_yticklabels(corr_matrix.columns, rotation=0)
        ax.set_xticklabels(corr_matrix.columns)
        sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
        return heatmap

if __name__ == '__main__':
    h = annotated_heatmap()
    h.run()