import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Graph:

    def __init__(self):
      pass

    def multi_histogram(self, dfs, qnt = 2):
      """
      @param dfs -- list of Pandas.DataFrame 1-D
      @param qnt -- quanty of histogram to plot

      """
      _, grid_plot = plt.subplots(1, qnt, figsize=(15, 7.5), sharex=True)
      array_len = [x for x in range(0, qnt)]

      for (df, index) in zip(dfs, array_len):
        sns.distplot(df, bins=25, ax=grid_plot[index], axlabel=f"sample_{index}")

    def multi_boxplot(self, dfs, qnt = 2):
      """
      @param dfs -- list of Pandas.DataFrame 1-D
      @param qnt -- quanty of boxplot to plot
      """
      
      _, grid_plot = plt.subplots(1, qnt, figsize=(18, 8), sharex=True)
      array_len = [x for x in range(0, qnt)]

      for (df, index) in zip(dfs, array_len):
        sns.boxplot(df, color="red", ax=grid_plot[index])
