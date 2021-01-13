import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
from ast import literal_eval

class DataPlotter(object):

    data = ''
    SETTINGS = ''

    def __init__(self, settings):
        self.SETTINGS = settings

    def plot_oned_data(self, data):
        sns.set(font='CMU Serif', font_scale=3.0, style="whitegrid")
        ax = sns.lineplot(x='Time', y=self.SETTINGS['feature_renames'][0], hue='ObjectID', data=data, legend=False,
                          palette=sns.color_palette(sns.hls_palette(8, l=.8, s=.7),
                                                    n_colors=len(self.data['ObjectID'].unique())))
        plt.show()
        return

    def plot_twod_data(self, data):
        sns.set(font='CMU Serif', font_scale=2.2, style="darkgrid")
        g = sns.FacetGrid(data, col='Time', col_wrap=self.SETTINGS['plot']['col_wrap'], palette='Set1')

        def f(x, y, z, **kwargs):
            ax = sns.scatterplot(x, y, **kwargs)

            for i in range(len(x)):
                ax.annotate(z.values[i], xy=(x.values[i], y.values[i]), fontsize=15,
                            xytext=(0, 0), textcoords="offset points",
                            bbox=dict(boxstyle='round', alpha=0.3),
                            va='center', ha='center', weight='bold', alpha=1)

        g.map(f, self.SETTINGS['feature_renames'][0], self.SETTINGS['feature_renames'][1], 'ObjectID')
        plt.subplots_adjust(top=0.8)
        plt.show()
        return

    def plot_onedim_clusters(self, clusters, outlier=None, draw_points=False):
        sns.set(font='CMU Serif', font_scale=3.0, style="whitegrid")

        self.data = clusters
        # ax = sns.lineplot(x='Time', y=self.SETTINGS['feature_renames'][0], hue='ObjectID', data=self.data, legend=False,
        #                   palette=sns.color_palette('muted', n_colors=len(self.data['ObjectID'].unique())))

        # MLDM Outlier Paper
        ax = sns.lineplot(x='Time', y=self.SETTINGS['feature_renames'][0], hue='ObjectID', data=self.data, legend=False,
                          palette=sns.color_palette(sns.hls_palette(8, l=.8, s=.7), n_colors=len(self.data['ObjectID'].unique())))

        if outlier is not None:
            for index, row in outlier.iterrows():
                plot_data = self.data[(self.data['ObjectID'] == row['ObjectID'])
                                      & (self.data['Time'] >= row['start_time'])
                                      & (self.data['Time'] <= row['end_time'])
                                      ]
                # print('PLOTTING DATA:')
                # print(plot_data)
                if (row['distance'] != -1):
                    axis = sns.lineplot(x='Time', y=self.SETTINGS['feature_renames'][0], hue='ObjectID',
                                        data=plot_data, legend=False, ax=ax, palette=['#000000'], )

            for index, row in outlier.iterrows():
                plot_data = self.data[(self.data['ObjectID'] == row['ObjectID'])
                                      & (self.data['Time'] >= row['start_time'])
                                      & (self.data['Time'] <= row['end_time'])
                                      ]
                # print('PLOTTING DATA:')
                # print(plot_data)
                if (row['distance'] == -1):
                    axis = sns.lineplot(x='Time', y=self.SETTINGS['feature_renames'][0], hue='ObjectID',
                                        data=plot_data, legend=False, ax=ax, palette=['#FFFFFF'])
                    axis.lines[len(axis.lines) - 1].set_linestyle("-.")

        if draw_points:
            cluster_color_palette = ['#1E88E5', '#FFC107', '#1CF2AA', '#81019B', '#92C135', '#9C9C6E', '#1033EA', '#55A1D7',
                                     '#38CA48', '#637645', '#E9E2A3', '#F0A054', '#1E88E5', '#FFC107', '#1CF2AA', '#81019B']
            gr_data = self.data.groupby(['Time'])
            for time, data in gr_data:
                counter = 0
                for cluster in data['cluster'].unique():
                    if cluster > -1:
                        colour = cluster_color_palette[counter]
                        counter += 1
                    else:
                        colour = '#D81B60'
                    for object_id in data[data['cluster'] == cluster]['ObjectID'].unique():
                        plot_data = data[(data['cluster'] == cluster) & (data['ObjectID'] == object_id)]
                        # plt.scatter(x=time, y=plot_data[self.SETTINGS['feature_renames'][0]].values.item(), marker="o",
                        #             c=colour)

                        # MLDM Outlier Paper
                        plt.scatter(x=time, y=plot_data[self.SETTINGS['feature_renames'][0]].values.item(), marker="p",
                                    c=colour)
        plt.show()

    def plot_twodim_clusters(self, data, outlier=True, tick_steps=0.5, remove_labels=False):
        data.loc[data['Time'] < 0, 'Time'] = -1
        plot_data = data[data['Time'].isin(data['Time'].unique()[:self.SETTINGS['plot']['no_timepoints']])]
        plot_data.rename(columns={'Time': self.SETTINGS['plot']['time_axis_name']}, inplace=True)

        sns.set(font='CMU Serif', font_scale=2.2, style = "darkgrid")
        g = sns.FacetGrid(plot_data, col=self.SETTINGS['plot']['time_axis_name'], hue="cluster",
                          col_wrap=self.SETTINGS['plot']['col_wrap'], palette='Set1')

        def f(x, y, z, w, v, **kwargs):
            ax = sns.scatterplot(x, y, **kwargs)
            if remove_labels:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
            else:
                ax.set(ylim=(-0.1, 1.1), yticks=np.arange(0.0, 1.2, tick_steps))
                ax.set(xlim=(-0.1, 1.1), xticks=np.arange(0.0, 1.2, tick_steps))
                ax.set(xlabel=self.SETTINGS['feature_renames'][0], ylabel=self.SETTINGS['feature_renames'][1])


            # Individual Colors
            # For every time point use 5 different colors (assume only max 4 clusters + outliers per time point)
            # Use a palette for color blind people https://davidmathlogic.com/colorblind/
            cluster_color_palette = ['#1E88E5', '#FFC107', '#1CF2AA', '#92C135', '#1033EA', '#55A1D7',
                                     '#38CA48', '#637645', '#81019B', '#E9E2A3', '#F0A054', '#1E88E5', '#FFC107']
            cluster_colors = list()
            cluster_color_labels = list()
            for time in plot_data[self.SETTINGS['plot']['time_axis_name']].unique():
                # reset used colors per time point
                color_index = 0
                for cluster_id in plot_data[plot_data[self.SETTINGS['plot']['time_axis_name']] == time]['cluster'].unique():
                    if cluster_id > -1:
                        cluster_colors.append(cluster_color_palette[color_index])
                        cluster_color_labels.append(cluster_id)
                        color_index = color_index + 1

            cluster_colors.append('#D81B60')
            cluster_color_labels.append(-1)

            cluster_colors.append('#000000')
            cluster_color_labels.append(-2)

            c = lambda x: cluster_colors[cluster_color_labels.index(x)]

            for i in range(len(x)):
                if outlier:
                    if v.values[i] > -1:
                        shape = "round"
                        color = c(w.values[i])
                        fcolor = "black"
                        ecolor = c(w.values[i])

                    # outliers which are detected by distance (type 1)
                    elif v.values[i] == -1:
                        shape = "circle"
                        color = c(-2)
                        fcolor = "white"
                        ecolor = "black"

                    # outliers which were detected by being followup cluster outliers (type 2)
                    elif v.values[i] == -2:
                        shape = "rarrow"
                        color = c(-2)
                        fcolor = "white"
                        ecolor = "black"

                    # outliers which are of type 1 and type 2 at the same time
                    elif v.values[i] == -3:
                        shape = "sawtooth"
                        color = c(-2)
                        fcolor = "white"
                        ecolor = "black"


                    if w.values[i] == -1:
                         ecolor = "red"
                         fcolor = "red"
                else:
                    shape = "round"
                    color = c(w.values[i])
                    fcolor = "black"
                    ecolor = c(w.values[i])

                ax.annotate(z.values[i], xy=(x.values[i], y.values[i]), fontsize=15,
                            xytext=(0, 0), textcoords="offset points",
                            color=fcolor,
                            bbox=dict(boxstyle=shape, alpha=0.3, color=color, edgecolor=ecolor),
                            va='center', ha='center', weight='bold', alpha=1)

        if outlier:
            g.map(f, self.SETTINGS['feature_renames'][0], self.SETTINGS['feature_renames'][1], "ObjectID", "cluster",
                  'outlier', alpha=0.6, s=5, legend='full')

        # cheated to handle missing 'outlier' column
        else:
            g.map(f, self.SETTINGS['feature_renames'][0], self.SETTINGS['feature_renames'][1], "ObjectID", "cluster",
                  'cluster', alpha=0.6, s=5, legend='full')

        if remove_labels:
            g.set_titles(row_template='', col_template='')
            g.set_xlabels('')
            g.set_ylabels('')

        plt.subplots_adjust(top=0.8)
        return plt



    def plot_twodim_fuzzy(self, data, outlier=True, tick_steps=0.5, remove_labels=False):
        data.loc[data['Time'] < 0, 'Time'] = -1
        plot_data = data[data['Time'].isin(data['Time'].unique()[:self.SETTINGS['plot']['no_timepoints']])]
        plot_data.rename(columns={'Time': self.SETTINGS['plot']['time_axis_name']}, inplace=True)

        sns.set(font='CMU Serif', font_scale=2.2, style="darkgrid")
        g = sns.FacetGrid(plot_data, col=self.SETTINGS['plot']['time_axis_name'], hue="cluster",
                          col_wrap=self.SETTINGS['plot']['col_wrap'], palette='Set1')

        def f(x, y, z, w, v, **kwargs):
            ax = sns.scatterplot(x, y, **kwargs)
            if remove_labels:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
            else:
                ax.set(ylim=(-0.1, 1.1), yticks=np.arange(0.0, 1.2, tick_steps))
                ax.set(xlim=(-0.1, 1.1), xticks=np.arange(0.0, 1.2, tick_steps))
                ax.set(xlabel=self.SETTINGS['feature_renames'][0], ylabel=self.SETTINGS['feature_renames'][1])

            # Individual Colors
            # For every time point use 5 different colors (assume only max 4 clusters + outliers per time point)
            # Use a palette for color blind people https://davidmathlogic.com/colorblind/
            cluster_color_palette = ['#1E88E5', '#FFC107', '#1CF2AA', '#92C135', '#1033EA', '#55A1D7',
                                     '#38CA48', '#637645', '#81019B', '#E9E2A3', '#F0A054', '#1E88E5', '#FFC107']
            cluster_colors = list()
            cluster_color_labels = list()
            for time in plot_data[self.SETTINGS['plot']['time_axis_name']].unique():
                # reset used colors per time point
                color_index = 0
                for cluster_id in plot_data[plot_data[self.SETTINGS['plot']['time_axis_name']] == time][
                    'cluster'].unique():
                    if cluster_id > -1:
                        cluster_colors.append(cluster_color_palette[color_index])
                        cluster_color_labels.append(cluster_id)
                        color_index = color_index + 1

            cluster_colors.append('#D81B60')
            cluster_color_labels.append(-1)

            cluster_colors.append('#000000')
            cluster_color_labels.append(-2)

            c = lambda x: cluster_colors[cluster_color_labels.index(x)]

            def get_max_fuzzy(memberships):
                memberships = eval(memberships)
                max = -999.0
                for k in memberships.keys():
                    if memberships[k] > max:
                        max = memberships[k]
                return float(max) * 1

            for i in range(len(x)):
                if w.values[i] > -1:
                    shape = "round"
                    color = c(w.values[i])
                    fcolor = "black"
                    ecolor = c(w.values[i])
                    alp = get_max_fuzzy(v.values[i])
                elif w.values[i] == -1:
                    shape = "round"
                    color = c(w.values[i])
                    ecolor = "red"
                    fcolor = "black"
                    alp = 0.3
                else:
                    shape = "round"
                    color = c(w.values[i])
                    fcolor = "black"
                    ecolor = c(w.values[i])
                    alp = 1


                ax.annotate(z.values[i], xy=(x.values[i], y.values[i]), fontsize=15,
                            xytext=(0, 0), textcoords="offset points",
                            color=fcolor,
                            bbox=dict(boxstyle=shape, alpha=alp, color=color, edgecolor=ecolor),
                            va='center', ha='center', weight='bold', alpha=alp)

        if outlier:
            g.map(f, self.SETTINGS['feature_renames'][0], self.SETTINGS['feature_renames'][1], "ObjectID", "cluster",
                  'memberships', alpha=0.5, s=5, legend='full')

        # cheated to handle missing 'outlier' column
        else:
            g.map(f, self.SETTINGS['feature_renames'][0], self.SETTINGS['feature_renames'][1], "ObjectID", "cluster",
                  'memberships', alpha=1, s=5, legend='full')

        if remove_labels:
            g.set_titles(row_template='', col_template='')
            g.set_xlabels('')
            g.set_ylabels('')

        plt.subplots_adjust(top=0.8)
        return plt

    def plot_outlier_mldm(self, data):
        data.loc[data['Time'] < 0, 'Time'] = -1
        plot_data = data[data['Time'].isin(data['Time'].unique()[:self.SETTINGS['plot']['no_timepoints']])]
        # plot_data = plot_data.sort_values(by=['Time', 'outlier'], ascending=False)
        plot_data.rename(columns={'Time': self.SETTINGS['plot']['time_axis_name']}, inplace=True)

        sns.set(font='CMU Serif', font_scale=2.2, style = "darkgrid")
        g = sns.FacetGrid(plot_data, col=self.SETTINGS['plot']['time_axis_name'], hue="cluster",
                          col_wrap=self.SETTINGS['plot']['col_wrap'], palette='Set1')

        def f(x, y, z, w, v, **kwargs):
            ax = sns.scatterplot(x, y, **kwargs)
            ax.set(ylim=(-0.2, 1.1))
            ax.set(xlim=(-0.2, 1.1))
            ax.set(xlabel=self.SETTINGS['feature_renames'][0], ylabel=self.SETTINGS['feature_renames'][1])
            # Individual Colors
            # For every time point use 5 different colors (assume only max 4 clusters + outliers per time point)
            # Use a palette for color blind people https://davidmathlogic.com/colorblind/
            cluster_color_palette = ['#1E88E5', '#FFC107', '#1CF2AA', '#81019B', '#89A4A8', '#562226', '#637645',
                                     '#F7EE75', '#637645', '#091E8C', '#E9E2A3', '#F0A054']
            cluster_colors = list()
            cluster_color_labels = list()
            for time in plot_data[self.SETTINGS['plot']['time_axis_name']].unique():
                # reset used colors per time point
                color_index = 0
                for cluster_id in plot_data[plot_data[self.SETTINGS['plot']['time_axis_name']] == time]['cluster'].unique():
                    if cluster_id > -1:
                        cluster_colors.append(cluster_color_palette[color_index])
                        cluster_color_labels.append(cluster_id)
                        color_index = color_index + 1

            cluster_colors.append('#D81B60')
            cluster_color_labels.append(-1)

            cluster_colors.append('#000000')
            cluster_color_labels.append(-2)

            c = lambda x: cluster_colors[cluster_color_labels.index(x)]

            for i in range(len(x)):
                if v.values[i] > -1:
                    shape = "round"
                    color = c(w.values[i])
                    fcolor = "black"
                    ecolor = c(w.values[i])
                    alpha = 0.3
                    fontweight = 'normal'
                    fontsize = 14

                # outliers which are detected by distance (type 1)
                else:
                    shape = "sawtooth,pad=0.2,tooth_size=0.25"
                    color = c(w.values[i])
                    fcolor = "black"
                    ecolor = "red"
                    alpha = 0.6
                    fontweight = 'extra bold'
                    fontsize = 15


                ax.annotate(z.values[i], xy=(x.values[i], y.values[i]), fontsize=fontsize, fontweight=fontweight,
                            xytext=(0, 0), textcoords="offset points",
                            color=fcolor,
                            bbox=dict(boxstyle=shape, alpha=alpha, facecolor=color, edgecolor=ecolor),
                            va='center', ha='center', weight='bold', alpha=1)

            # print(labels[z.values[i]]+' ' + str(w.values[i]) + str(kwargs.get("color", "k")))

        #first plot normal data without outliers
        g.map(f, self.SETTINGS['feature_renames'][0], self.SETTINGS['feature_renames'][1], "ObjectID", "cluster", 'outlier', alpha=0.6, s=5, legend='full')
        plt.subplots_adjust(top=0.8)
        #g.fig.suptitle(self.SETTINGS['plot']['title'])
        return plt

    def plot_onedim_outlier_example(self, clusters, outlier, draw_points=True):
        sns.set(font='CMU Serif', font_scale=3.0, style="white")

        self.data = clusters
        outlier_id = outlier['ObjectID'].unique()[0]
        ax = sns.lineplot(x='Time', y=self.SETTINGS['feature_renames'][0], hue='ObjectID', data=self.data[self.data['ObjectID'] != outlier_id], legend=False,
                          palette=sns.color_palette('ch:2.5,.2,d=.65, l=.8', n_colors=len(self.data['ObjectID'].unique())-1))
        ax.set(xlim=(0,20), ylim=(0,1), yticks=[], xticks=[])


        plot_data = self.data[(self.data['ObjectID'] == outlier_id)]
        axis = sns.lineplot(x='Time', y=self.SETTINGS['feature_renames'][0], hue='ObjectID',
                                data=plot_data, legend=False, ax=ax, palette=['#000000'], )
        axis.lines[len(axis.lines) - 1].set_linestyle("-.")
        axis.lines[len(axis.lines) - 1].set_linewidth(2)
        axis.set(xlim=(0.5, 20.5), ylim=(0.05, 0.9), yticks=[], xticks=[])
        marker_list = ['o', 'v', 's']

        if draw_points:
            cluster_color_palette = ['#1E88E5', '#FFC107', '#1CF2AA', '#81019B', '#92C135', '#9C9C6E', '#1033EA', '#55A1D7',
                                     '#38CA48', '#637645', '#E9E2A3', '#F0A054']
            gr_data = self.data.groupby(['Time'])
            for time, data in gr_data:
                counter = 0
                for cluster in data['cluster'].unique():
                    if cluster > -1:
                        colour = cluster_color_palette[counter]
                        counter += 1
                    else:
                        colour = '#D81B60'
                    for object_id in data[data['cluster'] == cluster]['ObjectID'].unique():
                        if self.data[(self.data['ObjectID'] == object_id) & (self.data['Time'] == time)][self.SETTINGS['feature_renames'][0]].item() > 0.66:
                            marker_index = 0
                        elif self.data[(self.data['ObjectID'] == object_id) & (self.data['Time'] == time)][self.SETTINGS['feature_renames'][0]].item() > 0.4:
                            marker_index = 1
                        else:
                            marker_index = 2
                        plot_data = data[(data['cluster'] == cluster) & (data['ObjectID'] == object_id)]
                        plt.scatter(x=time, y=plot_data[self.SETTINGS['feature_renames'][0]].values.item(), marker=marker_list[marker_index],
                                    c=colour)
        plt.xlabel("Time")
        plt.ylabel("Feature")
        plt.show()
