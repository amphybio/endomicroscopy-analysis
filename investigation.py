#   Copyright (C) 2020 AMPhyBio
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with program.  If not, see <http://www.gnu.org/licenses/>.

# =============================================================================
#          FILE: crypt-plot.py
#
#   DESCRIPTION: Program to generate plots based on file output of
#                Endomicroscopy morphometry analysis
#
#       OPTIONS:
#  REQUIREMENTS:  Python, Numpy, Seaborn
#          BUGS:  ---
#         NOTES:  ---
#         AUTOR:  Alan U. Sabino <alan.sabino@usp.br>
#       VERSION:  0.1
#       CREATED:  05/06/2020
#      REVISION:  ---
# =============================================================================

# USAGE
# python plot.py -f box_plot -p midia/main/1234/data/stitch100/axisr_data.csv
# python plot.py -f box_plot -p midia/main/1234/data/stitch100/spher_data.csv -o 0
# python plot.py -f hist_plot -p midia/main/BKP/data/AB/axisr_data.csv
# python plot.py -f distb_dist -p midia/main/BKP/data/AB/axisr_data.csv
# python plot.py -f all_csv -p midia/main/1234/data/stitch100/
# python plot.py -f join_csv -p midia/main/1234/data/ -o axis


import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# Define style plots
sns.set(context='notebook', style='ticks', palette='bright', font='Arial',
        font_scale=2, rc={'axes.grid': True, 'grid.linestyle': 'dashed',
                          'lines.linewidth': 2, 'xtick.direction': 'in',
                          'ytick.direction': 'in', 'figure.figsize': (7, 3.09017)})


def read_csv(data_source):
    data = []
    with open(data_source, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            data.append(row)
    return data


def all_csv(source):
    import subprocess
    csv_files = subprocess.run(f"ls -1v {source}*csv", shell=True,  stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True)
    for path in csv_files.stdout.splitlines():
        data = read_csv(path)
        box_plot(data)
        print(path)


def join_csv(source, data):
    # Not working properly
    import subprocess
    csv_files = subprocess.run(f"ls -1v {source}*/{data}*", shell=True,  stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True)
    import pandas as pd
    all_files = csv_files.stdout.splitlines()
    combined_csv = pd.concat([pd.read_csv(f) for f in all_files])
    combined_csv.to_csv("combined_csv.csv", index=False, encoding='utf-8')
    # combined_csv.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')


def distribution_distance(data):
    data_float = [np.asarray(list(filter(None, arr[1:])), dtype=np.float)
                  for arr in data[1:]]
    for index, first_distribution in enumerate(data_float):
        p = first_distribution / np.asarray(first_distribution).sum()
        p = np.asarray(p[:19])  # CHUMBADO
        # p = np.asarray(p[24:])  # CHUMBADO
        print(f"{data[index+1][0]}\n+++++++++++++++")
        for compare, second_distribution in enumerate(data_float[index+1:]):
            print(f"{data[index+2+compare][0]}")
            q = second_distribution / np.asarray(second_distribution).sum()
            print(f"Hellinger:\t {hellinger_distance(p,q)}"
                  f"\nJensen-Shannon:\t {jensen_shannon_distance(p,q)}"
                  f"\nBhattacharyya:\t {bhattacharyya_distance(p,q)}\n---------------")


def hellinger_distance(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


def bhattacharyya_distance(p, q):
    return -np.log(np.sum(np.sqrt(p*q)))


def jensen_shannon_distance(p, q):
    m = (p+q)/2
    left = kullback_leibler_divergence(p, m)
    right = kullback_leibler_divergence(q, m)
    return np.sqrt((left+right)/2)


def kullback_leibler_divergence(p, q):
    return np.sum(p*np.log2((p/q)))


def shannon_entropy(data):
    p = data / np.asarray(data).sum()
    return -np.sum(p*np.log2(p))


def freq_plot(data):
    sorted_set = natural_sort(list(set(data)))
    ax = sns.barplot(x=data, y=data, estimator=lambda x: len(
        x) / len(data) * 100, palette=["C0"], order=sorted_set)
    ax.set(xlabel="Classes",
           ylabel="Frequency (%)")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plot = ax.get_figure()
    plot.canvas.draw()
    for label in ax.get_yticklabels():
        if int(label.get_text().replace('−', '-')) % 10 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    plot.savefig(f"{data[0][0]}_plot.tif", dpi=600, bbox_inches="tight")
    plt.clf()


def hist_plot(data, ticks_number=[13, 7], decimals=[0, 2]):
    data_float = [np.asarray(list(filter(None, arr[1:])), dtype=np.float)
                  for arr in data[1:]]
    X_MAX = max(map(max, data_float))
    X_MIN = min(map(min, data_float))
    x_interval = np.round((X_MAX - X_MIN) / ticks_number[0], decimals[0])
    x_ticks = np.round(
        np.arange(X_MIN, X_MAX+x_interval, x_interval), decimals[0])
    fig, ax = plt.subplots(1)
    densities = [0]
    for index, img_data in enumerate(data_float):
        densities.append(ax.hist(img_data, density=True, bins=x_ticks,
                                 alpha=.85, label=data[index+1][0])[0])
    densities = densities[1:]
    Y_MAX = max(map(max, densities))
    Y_MIN = min(map(min, densities))
    y_interval = np.round((Y_MAX - Y_MIN) / ticks_number[1], decimals[1])
    y_ticks = np.round(
        np.arange(Y_MIN, Y_MAX+y_interval, y_interval), decimals[1])
    ax.set(title=data[0][1], ylabel="Density",
           xlabel=data[0][3], xticks=x_ticks, yticks=y_ticks)
    # Optional line | IF decimals 0 >> astype(np.int)
    ax.set_xticklabels(ax.get_xticks().astype(np.int), size=16)
    plt.legend(loc='upper left', prop={'size': 12})
    plot = ax.get_figure()
    plot.canvas.draw()
    for index, label in enumerate(ax.get_yticklabels()):
        if index % 2 == 1:
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.savefig(f"{data[0][0]}_plot.tif", dpi=600, bbox_inches="tight")
    plt.clf()


def box_plot(data, ticks_number=10, decimals=1):
    data_float = [np.asarray(list(filter(None, arr[1:])), dtype=np.float)
                  for arr in data[1:]]
    ax = sns.boxplot(data=data_float, width=0.45,
                     palette=["C0"])
    ax.set(title=data[0][1], xlabel=data[0][2],
           ylabel=data[0][3], xticklabels=list(np.asarray(data)[1:, 0]))
    interval = round((max(data_float[0]) - min(data_float[0])) /
                     ticks_number, decimals)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plot = ax.get_figure()
    plot.canvas.draw()
    for index, label in enumerate(ax.get_yticklabels()):
        if index % 2 == 1:
            label.set_visible(True)
        else:
            label.set_visible(False)
    plot.savefig(f"{data[0][0]}_plot.tif", dpi=600, bbox_inches="tight")
    plt.clf()


def natural_sort(l):
    import re
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--function", type=str, required=True,
                    help="Set a function to call (box_plot, freq_plot)")
    ap.add_argument("-p", "--path", type=str, required=False,
                    help="Input file or directory of images path")
    ap.add_argument("-o", "--optional", type=str, required=False,
                    help="Optional parameter for functions")
    args = vars(ap.parse_args())
    function = args["function"]
    source = args["path"]

    if (function == "box_plot"):
        data = read_csv(source)
        decimals = args["optional"]
        if decimals is None:
            box_plot(data)
        else:
            box_plot(data, decimals=int(decimals))
    elif (function == "hist_plot"):
        data = read_csv(source)
        hist_plot(data)
    elif (function == "all_csv"):
        all_csv(source)
    elif (function == "join_csv"):
        data = args["optional"]
        join_csv(source, data)
    elif (function == "distb_dist"):
        data = read_csv(source)
        distribution_distance(data)
    else:
        print("Undefined function")


if __name__ == "__main__":
    main()
