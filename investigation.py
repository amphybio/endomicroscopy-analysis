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
#          FILE: investigation.py
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
# python investigation.py -f box-plot  -p data/0001/perim_data.csv -d 7 2
# python investigation.py -f hist-plot -p data/0002/perim_data.csv -d 6 6 0 3
# python investigation.py -f summary   -p data/0003/
# python investigation.py -f dist-plot -p data/perim_combined_data.csv -d 6 6 0 3
# python investigation.py -f join-csv  -p data/ -m perim


import csv
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import logendo as le

def plot_style():
    # Define style plots
    sns.set(context='notebook', style='ticks', palette='colorblind', font='Arial',
            font_scale=2, rc={'axes.grid': True, 'grid.linestyle': 'dashed',
                              'lines.linewidth': 2, 'xtick.direction': 'in',
                              'ytick.direction': 'in', 'figure.figsize': (7, 3.09017)})

def search_files(source, pattern):
    logger.debug('Initializing search files function.')
    import subprocess
    output = subprocess.run(f'find {source} -type f -name "{pattern}"', shell=True,  stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    files = output.stdout.splitlines()
    files.sort()
    logger.info(f'Pattern: {pattern}. No. of files found: {len(files)}')
    logger.debug(f'Files found: {files}')
    logger.debug('Finished search files function.')
    return files

def data_stats(source, outliers=False, measures=['axisr', 'dist', 'elong',
                                                'feret', 'min_dist', 'perim', 'round', 'spher', 'wall']):
    start_time = timer()
    logger.debug('Initializing STAT csv data')

    # import subprocess
    import pathlib

    # measure_list = ['axisr', 'dist', 'elong', 'feret', 'min_dist', 'perim',
    #                 'round', 'spher', 'wall']

    # summary_data = [['Measure', 'Mean', 'STD','No. Images','No. Crypts']]
    data_export = [['Measure', 'Mean', 'STD']]

    for measure in measures:
        # output = subprocess.run(f'find {source} -type f -name "{measure}_data.csv"', shell=True,  stdout=subprocess.PIPE,
        #                         stderr=subprocess.STDOUT, universal_newlines=True)
        # files = output.stdout.splitlines()
        # files.sort()
        # logger.debug(f'Measure: {measure}. No. of csv files found: {len(files)}'
        #              f'. Files: {files}')

        files = search_files(source, f'{measure}_data.csv')

        prev_path = pathlib.Path(files[0])
        # data = [read_csv(prev_path)[0]]
        data = [read_csv(prev_path)[0]]
        data_id = read_csv(prev_path)[1]
        prev_path = pathlib.Path(files[0]).parents[2]
        data_id.insert(0, prev_path)
        sum_crypts = 0
        for f_path in files[1:]:
            path = pathlib.Path(f_path).parents[2]
            # if prev_path == path:
            if str(prev_path)[:-1] == str(path)[:-1]:
                dta = read_csv(f_path)[1]
                data_id.extend(dta)
            else:
                sum_crypts += len(data_id)-1
                logger.debug(f'ID: {np.asarray(data_id[1:]).astype("float")}')
                logger.debug(f'Path: {prev_path}. No Crypts: {len(data_id)-1}. Mean: {np.mean(np.asarray(data_id[1:]).astype("float")):.2f}')
                data.append(data_id)
                data_id = read_csv(f_path)[1]
                data_id.insert(0, path)
            prev_path = path

        sum_crypts += len(data_id)-1
        logger.debug(f'Path: {prev_path}. No Crypts: {len(data_id)-1}. Mean: {np.mean(np.asarray(data_id[1:]).astype("float")):.2f}')
        data.append(data_id)
        data_float = [np.asarray(
            list(filter(None, arr[1:])), dtype=np.float) for arr in data[1:]]
        flat_list = [item for sublist in data_float for item in sublist]
        if not outliers:
            flat_list = rm_outliersG(data[0][1], flat_list)

        logger.debug(f'GLOBAL {measure}. No. Crypts: {len(flat_list)}/{sum_crypts}.'
                     f' Mean: {np.mean(flat_list):.2f}.'
                     f' STD: {np.std(flat_list):.2f}')
        # logger.debug(f'NaN: {flat_list}')
        data_export.append(
            [data[0][1], np.mean(flat_list), np.std(flat_list)])

        global_list = [data[0]]
        global_list.append(flat_list)
        to_csv(global_list, f'{measure}_global_data')
        to_csv(data_export, 'summary-comb')
        to_csv(data, f'{measure}_combined_data')
    end_time = timer()
    logger.debug(
        f'STAT function time elapsed: {end_time-start_time:.2f}s')


def dist_plotG(data, measure, ticks_number=[6, 6], decimals=[0, 3], outliers=False):
    start_time = timer()
    logger.info('Initializing distance histogram')
    global_data = read_csv(f'{measure}_global_data.csv')
    # logger.debug(f'Global: {global_data}')
    global_float = [np.asarray(list(filter(None, arr)), dtype=np.float)
                    for arr in global_data[1:]]
    # logger.debug(f'Global Float: {global_float}')

    data_float = [np.asarray(list(filter(None, arr[1:])), dtype=np.float)
                  for arr in data[1:]]
    if not outliers:
        data_float = rm_outliers(data_float)  # Verificar
    # logger.debug(f'S: {len(data_float)}. Data: {data_float}')

    labels = [l[0][-3:] for l in data[1:]]
    # labels = [l[0][-4:] for l in data[1:]]
    labels.insert(0, 'Global')
    x_ticks = ticks_interval(global_float, ticks_number[0], decimals[0])
    interval = x_ticks[1]-x_ticks[0]
    logger.debug(
        f'X-ticks range: {x_ticks[-1]-x_ticks[0]:.5f} |  No. of bins: {len(x_ticks)-1} | '
        f'Bin width: {interval:.5f}')
    hellinger_list = []
    disorder_list = []
    for idx, data_id in enumerate(data_float):
        # logger.debug(f'>>> {len(data_id)}')
        logger.debug(f'Data {labels[idx+1]} - Range: {min(data_id):.2f}..{max(data_id):.2f}'
                     f' No. crypsts: {len(data_id)} | Values: data_id')
        compare_data = []
        compare_data.append(global_float[0])
        compare_data.append(data_id)
        plot_style()
        _, ax = plt.subplots(1)
        # logger.debug(f'S: {len(compare_data)}. Data: {compare_data}')
        # x_ticks = ticks_interval(
        #     compare_data, ticks_number[0], decimals[0])
        # interval = x_ticks[1]-x_ticks[0]
        # logger.debug(
        #     f'X-ticks range: {x_ticks[-1]-x_ticks[0]:.5f} |  No. of bins: {len(x_ticks)-1} | '
        #     f'Bin width: {interval:.5f}')
        frequencies_list = [0]
        densities_list = [0]
        for index, img_data in enumerate(compare_data[: 2]):
            lab_ind = (index % 2)*idx + index
            densities_list.append(ax.hist(img_data, density=True, bins=x_ticks,
                                          alpha=.85, label=labels[lab_ind])[0])
            densities = densities_list[index+1]
            frequencies_list.append(densities * interval)
            entropy, max_entropy, degree_disorder = shannon_entropy(
                densities, x_ticks)
            logger.info(f'Data {index} - Densities: '
                        + str([f'{value:.5f}' for value in densities])
                        + ' | Frequencies: '
                        + str([f'{value:.5f}' for value in frequencies_list[index+1]])
                        + f' | Entropy {entropy:.3f}, Max {max_entropy:.3f} | '
                        + f'Degree of disorder: {degree_disorder:.3f}')
        disorder_list.append((degree_disorder, f'{labels[idx+1]}'))
        frequencies_list = frequencies_list[1:]
        hellinger_dist = hellinger_distance(
            frequencies_list[0], frequencies_list[1])
        logger.info(f'Hellinger distance: {hellinger_dist:.3f}')
        hellinger_list.append((hellinger_dist, f'{labels[idx+1]}'))
        densities_list = densities_list[1:]

        ax.set(title=data[0][1], ylabel="Density", xlabel=data[0][3], xticks=x_ticks,
               yticks=ticks_interval(densities_list, ticks_number[1], decimals[1]))
        # Optional line | IF decimals 0 >> astype(np.int)
        # ax.set_xticklabels(ax.get_xticks().astype(int), size=17)
        plt.legend(loc='upper right', prop={'size': 12})
        # plt.legend(loc='upper left', prop={'size': 12})
        plot = ax.get_figure()
        plot.canvas.draw()
        for index, label in enumerate(ax.get_yticklabels()):
            if index % 2 == 1:
                label.set_visible(True)
            else:
                label.set_visible(False)
        name = data[idx+1][0]
        plt.savefig(f"G-{name[-3:]}_{data[0][0]}_plot.tif",
                    dpi=600, bbox_inches="tight")
        plt.clf()
    # hellinger_list.sort()
    # disorder_list.sort()
    np.set_printoptions(precision=2)
    # logger.debug('Ordered Hellinger: ' +
    logger.debug('Hellinger: ' +
                 str([f'({value[0]:.3f}, {value[1]})' for value in hellinger_list]))
    # logger.debug('Ordered Degree of disorder: ' +
    logger.debug('Degree of disorder: ' +
                 str([f'({value[0]:.3f}, {value[1]})' for value in disorder_list]))
    logger.info('Finished distance histogram')
    end_time = timer()
    logger.debug(
        f'Distance histogram function time elapsed: {end_time-start_time:.2f}s')


def subgroup(data, group, measure, ticks_number=[6, 6], decimals=[0, 3], outliers=False):
    start_time = timer()
    logger.info('Initializing subgroup')
    # logger.debug(f'Group: {group} |  Mesuare: {measure}')
    group_data = []
    complementary_data = []
    data_id = [l[0][-3:] for l in data[1:]]
    # data_id = [l[0][-4:] for l in data[1:]]
    # data_float = [np.asarray(list(filter(None, arr)), dtype=np.float)
    #                 for arr in data[1:]]
    for index, value in enumerate(data_id):
        if value in group:
            group_data.extend(data[index+1][1:])
        else:
            complementary_data.extend(data[index+1][1:])
    # logger.debug(f'Group: {group_data}')
    # logger.debug(f'Complementary: {complementary_data}')
    # for v in group_data:
        # print(float(v))
    # quit()
    group_float = [ float(arr) for arr in group_data]
    complementary_float = [ float(arr) for arr in complementary_data]
    # logger.debug(f'G: {group_float}')
    if not outliers:
        group_float = rm_outliersG(measure, group_float)
        complementary_float = rm_outliersG(measure, complementary_float)
    logger.debug(
        f'No. crypts - Group: {len(group_float)} Complementary: {len(complementary_float)}'
        f' | Mean ± std - Group: {np.mean(group_float):.2f}±{np.std(group_float):.2f}'
        f' Complementary: {np.mean(complementary_float):.2f}±{np.std(complementary_float):.2f}')

    ################################# Gráfico

    global_data = read_csv(f'{measure}_global_data.csv')
    # logger.debug(f'Global: {global_data}')
    global_float = [np.asarray(list(filter(None, arr)), dtype=np.float)
                    for arr in global_data[1:]]

    x_ticks = ticks_interval(global_float, ticks_number[0], decimals[0])
    interval = x_ticks[1]-x_ticks[0]
    logger.debug(
        f'X-ticks range: {x_ticks[-1]-x_ticks[0]:.5f} |  No. of bins: {len(x_ticks)-1} | '
        f'Bin width: {interval:.5f}')
    hellinger_list = []
    disorder_list = []
    data_float = []
    data_float.append(group_float)
    data_float.append(complementary_float)
    labels = ['Global', 'Complete', 'Incomplete']
    for idx, data_id in enumerate(data_float):
        logger.debug(f'Data {labels[idx]} - Range: {min(data_id):.2f}..{max(data_id):.2f}'
                     f' No. crypsts: {len(data_id)} | Values: data_id')
        compare_data = []
        compare_data.append(global_float[0])
        compare_data.append(data_id)
        plot_style()
        _, ax = plt.subplots(1)
        # logger.debug(f'S: {len(compare_data)}. Data: {compare_data}')
        # x_ticks = ticks_interval(
        #     compare_data, ticks_number[0], decimals[0])
        # interval = x_ticks[1]-x_ticks[0]
        # logger.debug(
        #     f'X-ticks range: {x_ticks[-1]-x_ticks[0]:.5f} |  No. of bins: {len(x_ticks)-1} | '
        #     f'Bin width: {interval:.5f}')
        frequencies_list = [0]
        densities_list = [0]
        for index, img_data in enumerate(compare_data[: 2]):
            lab_ind = (index % 2)*idx + index
            densities_list.append(ax.hist(img_data, density=True, bins=x_ticks,
                                          alpha=.85, label=labels[lab_ind])[0])
            densities = densities_list[index+1]
            frequencies_list.append(densities * interval)
            entropy, max_entropy, degree_disorder = shannon_entropy(
                densities, x_ticks)
            logger.info(f'Data {index} - Densities: '
                        + str([f'{value:.5f}' for value in densities])
                        + ' | Frequencies: '
                        + str([f'{value:.5f}' for value in frequencies_list[index+1]])
                        + f' | Entropy {entropy:.3f}, Max {max_entropy:.3f} | '
                        + f'Degree of disorder: {degree_disorder:.3f}')
        disorder_list.append((degree_disorder, f'{labels[idx+1]}'))
        frequencies_list = frequencies_list[1:]
        hellinger_dist = hellinger_distance(
            frequencies_list[0], frequencies_list[1])
        logger.info(f'Hellinger distance: {hellinger_dist:.3f}')
        hellinger_list.append((hellinger_dist, f'{labels[idx+1]}'))
        densities_list = densities_list[1:]

        ax.set(title=data[0][1], ylabel="Density", xlabel=data[0][3], xticks=x_ticks,
               yticks=ticks_interval(densities_list, ticks_number[1], decimals[1]))
        # Optional line | IF decimals 0 >> astype(np.int)
        # ax.set_xticklabels(ax.get_xticks().astype(int), size=17)
        # plt.legend(loc='upper right', prop={'size': 12})
        plt.legend(loc='upper left', prop={'size': 12})
        plot = ax.get_figure()
        plot.canvas.draw()
        for index, label in enumerate(ax.get_yticklabels()):
            if index % 2 == 1:
                label.set_visible(True)
            else:
                label.set_visible(False)
        name = labels[idx+1]
        plt.savefig(f"CG-{name}_{data[0][0]}_plot.tif",
                    dpi=600, bbox_inches="tight")
        plt.clf()
    hellinger_list.sort()
    disorder_list.sort()

    logger.info('Finished subgroups')
    end_time = timer()
    logger.debug(
        f'Subgroup function time elapsed: {end_time-start_time:.2f}s')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def join_csv(source, measure):
    start_time = timer()
    logger.debug('Initializing join csv data')
    import subprocess
    # csv_files = subprocess.run(f"ls -1v {source}*/{measure}_data.csv", shell=True,  stdout=subprocess.PIPE,
    #                            stderr=subprocess.STDOUT, universal_newlines=True)
    csv_files = subprocess.run(f'find {source} -type f -name "{measure}_data.csv"', shell=True,  stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True)
    all_files = csv_files.stdout.splitlines()
    all_files.sort()
    logger.info(f'No. of csv files found: {len(all_files)}')
    logger.debug(f'Measure: {measure} | Files: {all_files}')
    data = [read_csv(all_files[0])[0]]
    for csv_file in all_files:
        values = read_csv(csv_file)[1]
        values.insert(0, csv_file)
        data.append(values)

    to_csv(data, f'{measure}_combined_data')
    logger.debug('Finished join csv')
    end_time = timer()
    logger.debug(
        f'Join csv function time elapsed: {end_time-start_time:.2f}s')


def summary_comb(source, outliers=False):
    start_time = timer()
    logger.info('Initializing summary combination')
    import subprocess
    output = subprocess.run(f'find {source} -maxdepth 1 -type f -name "*combined*csv"', shell=True,  stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    csv_files = output.stdout.splitlines()
    csv_files.sort()

    logger.info(f'Files found - no.: {len(csv_files)} | Files: {csv_files}')
    data_export = [["Parameter", "Mean", "STD"]]
    for path in csv_files:
        data = read_csv(path)
        data_float = [np.asarray(
            list(filter(None, arr[1:])), dtype=np.float) for arr in data[1:]]

        flat_list = [item for sublist in data_float for item in sublist]

        if not outliers:
            flat_list = rm_outliersG(data[0][1], flat_list)

        data_export.append(
            [data[0][1], np.mean(flat_list), np.std(flat_list)])

    to_csv(data_export, "summary-comb")
    logger.info('Finished summary combination')
    end_time = timer()
    logger.debug(
        f'Summary combination function time elapsed: {end_time-start_time:.2f}s')


def rm_outliersG(par_name, data):
    logger.debug(f'{par_name} length: {len(data)}')
    ordered = np.sort(data)
    # logger.debug(f'>>> {len(ordered)}')
    ordered = ordered[~np.isnan(ordered)]
    Q1 = np.quantile(ordered, 0.25)
    Q3 = np.quantile(ordered, 0.75)
    IQR = Q3 - Q1
    output = ordered[(ordered >= Q1 - 1.5*IQR) &
                     (ordered <= Q3 + 1.5*IQR)]
    logger.debug(f'Final length: {len(output)}')
    return output


def hist_plotG(data, ticks_number=[6, 6], decimals=[0, 3], outliers=False):
    start_time = timer()
    logger.info('Initializing histogram')
    data_float = [np.asarray(list(filter(None, arr[1:])), dtype=np.float)
                  for arr in data[1:]]
    flat_list = [item for sublist in data_float for item in sublist]

    if not outliers:
        flat_list = rm_outliersG(data[0][1], flat_list)

    plot_style()
    _, ax = plt.subplots(1)
    x_ticks = ticks_intervalG(flat_list, ticks_number[0], decimals[0])
    densities = ax.hist(flat_list, density=True, bins=x_ticks, alpha=.85)[0]
    ax.set(title=data[0][1], ylabel="Density", xlabel=data[0][3], xticks=x_ticks,
           yticks=ticks_intervalG(densities, ticks_number[1], decimals[1]))
    # Optional line | IF decimals 0 >> astype(np.int)
    # ax.set_xticklabels(ax.get_xticks().astype(np.int), size=16)
    plot = ax.get_figure()
    plot.canvas.draw()
    for index, label in enumerate(ax.get_yticklabels()):
        if index % 2 == 1:
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.savefig(f"{data[0][0]}_hist_plot.tif",
                dpi=600, bbox_inches="tight")
    plt.clf()
    logger.info('Finished histogram')
    end_time = timer()
    logger.debug(
        f'Histogram function time elapsed: {end_time-start_time:.2f}s')


def ticks_intervalG(data, quantity, decimals):
    start_time = timer()
    logger.debug('Initializing ticks interval')
    max_value = max(data)
    min_value = min(data)

    interval = np.round((max_value - min_value) / quantity, decimals)
    logger.debug(f'Min value: {min_value:.5f} | Max value: {max_value:.5f} | '
                 f'No. ticks: {quantity} | Decimal places: {decimals} | '
                 f'Interval: {interval:.5f}')

    ticks = np.round(np.arange(min_value, max_value +
                               interval, interval), decimals)
    logger.debug('Pre-ticks: ' + str([f'{value:.5f}' for value in ticks]))

    if max(ticks) < max_value:
        ticks = np.append(ticks, ticks[-1]+interval)
    if min(ticks) > min_value:
        min_tick = ticks[0]-interval
        if min_tick >= 0:
            ticks = np.insert(ticks, 0,  min_tick)
        else:
            ticks = np.round(np.arange(0, max_value +
                                       interval, interval), decimals)
    logger.debug('Ticks: ' + str([f'{value:.5f}' for value in ticks]))
    logger.debug('Finished ticks interval')
    end_time = timer()
    logger.debug(
        f'Ticks interval function time elapsed: {end_time-start_time:.2f}s')
    return ticks


###############################################################################

def rm_outliers(data):
    logger.debug('Initializing remove outliers')
    clean = []
    # if data
    for index, line in enumerate(data):
        line = line[~np.isnan(line)]
        logger.debug(f'Data {index} length: {len(line)}')
        ordered = np.sort(line)
        Q1 = np.quantile(ordered, 0.25)
        Q3 = np.quantile(ordered, 0.75)
        IQR = Q3 - Q1
        clean.append(ordered[(ordered >= Q1 - 1.5*IQR) &
                             (ordered <= Q3 + 1.5*IQR)])
        logger.debug(f'Data {index} final length: {len(clean[index])}')
    logger.debug('Finished remove outliers')
    return clean


def dist_plot(data, ticks_number=[5, 7], decimals=[2, 3], outliers=True):
    start_time = timer()
    logger.info('Initializing distance histogram')
    data_float = [np.asarray(list(filter(None, arr[1:])), dtype=np.float)
                  for arr in data[1:]]
    if not outliers:
        data_float = rm_outliers(data_float)
    plot_style()
    _, ax = plt.subplots(1)
    x_ticks = ticks_interval(
        data_float, ticks_number[0], decimals[0])

    interval = x_ticks[1]-x_ticks[0]
    logger.debug(
        f'X-ticks range: {x_ticks[-1]-x_ticks[0]:.5f} |  No. of bins: {len(x_ticks)-1} | '
        f'Bin width: {interval:.5f}')
    frequencies_list = [0]
    densities_list = [0]
    for index, img_data in enumerate(data_float[: 3]):
    # for index, img_data in enumerate(data_float[: 2]):
        densities_list.append(ax.hist(img_data, density=True, bins=x_ticks,
                                      alpha=(.85-(index*.1)), label=data[index+1][0])[0])
        densities = densities_list[index+1]
        frequencies_list.append(densities * interval)
        entropy, max_entropy, degree_disorder = shannon_entropy(
            densities, x_ticks)
        logger.info(f'Data {index} - Densities: '
                    + str([f'{value:.5f}' for value in densities])
                    + ' | Frequencies: '
                    + str([f'{value:.5f}' for value in frequencies_list[index+1]])
                    + f' | Entropy {entropy:.3f}, Max {max_entropy:.3f} | '
                    + f'Degree of disorder: {degree_disorder:.3f}')
    frequencies_list = frequencies_list[1:]
    # logger.info('Hellinger distance: '
    #             f'{hellinger_distance(frequencies_list[0], frequencies_list[1]):.3f}')
    logger.info('Hellinger distance GxR: '
                f'{hellinger_distance(frequencies_list[0], frequencies_list[1]):.3f}')
    logger.info('Hellinger distance GxT: '
                f'{hellinger_distance(frequencies_list[0], frequencies_list[2]):.3f}')
    densities_list = densities_list[1:]

    ax.set(title=data[0][1], ylabel="Density", xlabel=data[0][3], xticks=x_ticks,
           yticks=ticks_interval(densities_list, ticks_number[1], decimals[1]))
    # Optional line | IF decimals 0 >> astype(np.int)
    # ax.set_xticklabels(ax.get_xticks().astype(int), size=17)
    # plt.legend(loc='upper right', prop={'size': 12})
    plt.legend(loc='upper left', prop={'size': 12})
    plot = ax.get_figure()
    plot.canvas.draw()
    for index, label in enumerate(ax.get_yticklabels()):
        if index % 2 == 1:
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.savefig(f"{data[0][0]}_plot.tif",
                dpi=600, bbox_inches="tight")
    plt.clf()
    logger.info('Finished distance histogram')
    end_time = timer()
    logger.debug(
        f'Distance histogram function time elapsed: {end_time-start_time:.2f}s')


def shannon_entropy(densities, ticks):
    bin_width = ticks[1]-ticks[0]
    relative_frequency = densities * bin_width

    entropy = 0
    for freq in relative_frequency:
        entropy += freq * np.log2(freq/bin_width) if freq != 0 else 0
    entropy *= -1

    interval_width = ticks[-1]-ticks[0]
    max_entropy = np.log2(interval_width)

    degree_disorder = (2 ** entropy) / interval_width

    return entropy, max_entropy, degree_disorder


def jeffreys_distance(p, q):
    return np.sum((np.sqrt(p) - np.sqrt(q))**2)


def kullback_leibler_divergence(p, q):
    return np.sum(p*np.log2(p/q))


def jeffrey_divergence(p, q):
    left = kullback_leibler_divergence(p, q)
    right = kullback_leibler_divergence(q, p)
    return left+right


def jeffrey_e_divergence(p, q):
    return np.sum((p-q)*np.log2(p/q))


def bhattacharyya_coefficient(p, q):
    return np.sum(np.sqrt(p*q))


def bhattacharyya_distance(p, q):
    return -np.log(bhattacharyya_coefficient(p, q))


def hellinger_distance(p, q):
    return np.sqrt(1-bhattacharyya_coefficient(p, q))


def hellinger_e_distance(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


def jensen_shannon_distance(p, q, a=0.5):
    m = a * p + (1 - a) * q
    left = kullback_leibler_divergence(p, m)
    right = kullback_leibler_divergence(q, m)
    jensen_shannon_divergence = a*left + (1 - a)*right
    return np.sqrt(jensen_shannon_divergence)


def hist_plot(data, ticks_number=[6, 6], decimals=[0, 3], outliers=True):
    start_time = timer()
    logger.info('Initializing histogram')
    data_float = [np.asarray(list(filter(None, arr)), dtype=np.float)
                  for arr in data[1:]]
    if not outliers:
        data_float = rm_outliers(data_float)
    plot_style()
    _, ax = plt.subplots(1)
    x_ticks = ticks_interval(data_float, ticks_number[0], decimals[0])
    densities = ax.hist(data_float, density=True, bins=x_ticks, alpha=.85)[0]
    ax.set(title=data[0][1], ylabel="Density", xlabel=data[0][3], xticks=x_ticks,
           yticks=ticks_interval([densities], ticks_number[1], decimals[1]))
    # Optional line | IF decimals 0 >> astype(np.int)
    # ax.set_xticklabels(ax.get_xticks().astype(np.int), size=16)
    plot = ax.get_figure()
    plot.canvas.draw()
    for index, label in enumerate(ax.get_yticklabels()):
        if index % 2 == 1:
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.savefig(f"{data[0][0]}_hist_plot.tif",
                dpi=600, bbox_inches="tight")
    plt.clf()
    logger.info('Finished histogram')
    end_time = timer()
    logger.debug(
        f'Histogram function time elapsed: {end_time-start_time:.2f}s')


def box_plot(data, ticks_number=7, decimals=2):
    start_time = timer()
    logger.info('Initializing box-plot')
    if len(data) > 2:
        x_labels = list(np.asarray(data)[1:, 0])
        data_float = [np.asarray(list(filter(None, arr[1:])), dtype=np.float)
                      for arr in data[1:]]
    else:
        x_labels = ''
        data_float = [np.asarray(list(filter(None, arr)), dtype=np.float)
                      for arr in data[1:]]
    plot_style()
    ax = sns.boxplot(data=data_float, width=0.45)
    ax.set(title=data[0][1], xlabel=data[0][2], ylabel=data[0][3],
           xticklabels=x_labels, yticks=ticks_interval(data_float, ticks_number, decimals))
    plot = ax.get_figure()
    plot.canvas.draw()
    for index, label in enumerate(ax.get_yticklabels()):
        if index % 2 == 1:
            label.set_visible(True)
        else:
            label.set_visible(False)
    plot.savefig(f"{data[0][0]}_box_plot.tif",
                 dpi=600, bbox_inches="tight")
    plt.clf()
    logger.info('Finished box-plot')
    end_time = timer()
    logger.debug(f'Box-plot function time elapsed: {end_time-start_time:.2f}s')


def ticks_interval(data, quantity, decimals, pct=True):
    start_time = timer()
    logger.debug('Initializing ticks interval')
    max_value = max(map(max, data))
    min_value = min(map(min, data))

    interval = np.round((max_value - min_value) / quantity, decimals)
    logger.debug(f'Min value: {min_value:.5f} | Max value: {max_value:.5f} | '
                 f'No. ticks: {quantity} | Decimal places: {decimals} | '
                 f'Interval: {interval:.5f}')

    ticks = np.round(np.arange(min_value, max_value +
                               interval, interval), decimals)
    logger.debug('Pre-ticks: ' + str([f'{value:.5f}' for value in ticks]))

    if pct and max(ticks) > 100:
        min_tick = min(min(ticks),100-interval*(quantity+1))
        ticks = np.round(np.arange( min_tick, 100.01, interval), decimals)
        logger.debug('Pre-ticks(%): ' + str([f'{value:.5f}' for value in ticks]))
    if max(ticks) < np.round(max_value, decimals):
        ticks = np.append(ticks, ticks[-1]+interval)
    if min(ticks) > min_value:
        min_tick = ticks[0]-interval
        if min_tick >= 0:
            ticks = np.insert(ticks, 0,  min_tick)
        else:
            ticks = np.round(np.arange(0, max_value +
                                       interval, interval), decimals)
    logger.debug('Ticks: ' + str([f'{value:.5f}' for value in ticks]))
    logger.debug('Finished ticks interval')
    end_time = timer()
    logger.debug(
        f'Ticks interval function time elapsed: {end_time-start_time:.2f}s')
    return ticks


def summary_stats(source, outliers=True):
    start_time = timer()
    logger.debug('Initializing summary data')
    import subprocess
    csv_files = subprocess.run(f"ls -1v {source}*csv", shell=True,  stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True)
    data_export = [["Parameter", "Mean", "STD"]]
    for path in csv_files.stdout.splitlines():
        data = read_csv(path)
        data_float = [np.asarray(
            list(filter(None, arr)), dtype=np.float) for arr in data[1:]]
        if not outliers:
            data_float = rm_outliers(data_float)
        data_export.append(
            [data[0][1], np.mean(data_float), np.std(data_float)])
    to_csv(data_export, "summary")
    logger.debug('Finished summary')
    end_time = timer()
    logger.debug(
        f'Summary function time elapsed: {end_time-start_time:.2f}s')


def read_csv(data_source):
    data = []
    with open(data_source, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            data.append(row)
    return data


def to_csv(data, name='data'):
    with open(f"{name}.csv", mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            writer.writerow(row)


def is_valid(source):
    import pathlib
    path = pathlib.Path(source)
    if path.is_dir() or path.is_file():
        return True
    return False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--function", type=str, required=True,
                    help="Set a function to call (box_plot, freq_plot)")
    ap.add_argument("-v", "--verbose", help="Increase output verbosity",
                    action="store_true")
    ap.add_argument("-p", "--path", type=str, required=False,
                    help="Input file or directory of images path")
    ap.add_argument("-d", "--decimals", nargs='+', type=int, required=False,
                    help="Define number of decimals for plots ticks")
    ap.add_argument("-g", "--group", nargs='+', type=str, required=False,
                    help="Define group for plots")
    ap.add_argument("-m", "--measure", type=str, required=False,
                    help="Set measure to join CSV files")

    args = vars(ap.parse_args())
    function = args["function"]
    source = args["path"]
    decimals = args["decimals"]
    verbose = args["verbose"]
    group = args["group"]

    global logger
    if verbose:
        logger = le.logging.getLogger('debug')

    if is_valid(source):
        logger.info(
            '\n\nFRAMEWORK FOR ENDOMICROSCOPY ANALYSIS - PLOT MODULE\n')
        logger.info(f'Source: {source}')
        if (function == "box-plot"):
            data = read_csv(source)
            if decimals is None:
                box_plot(data)
            else:
                box_plot(data, ticks_number=decimals[0], decimals=decimals[1])
        elif (function == "hist-plot"):
            data = read_csv(source)
            if decimals is None:
                hist_plot(data)
                # hist_plotG(data)
            else:
                hist_plotG(
                    # hist_plot(
                    data, ticks_number=decimals[:2], decimals=decimals[2:])
        elif (function == "dist-plot"):
            data = read_csv(source)
            # measure = input('Type measure: ')
            if decimals is None:
                dist_plot(data)
                # dist_plotG(data, measure)
            else:
                dist_plot(
                    data, ticks_number=decimals[:2], decimals=decimals[2:])
                # dist_plotG(
                #     data, measure, ticks_number=decimals[:2], decimals=decimals[2:])
        elif (function == "join-csv"):
            measure = args["measure"]
            join_csv(source, measure)
        elif (function == "summary"):
            summary_stats(source)
        elif (function == "data-stats"):
            data_stats(source)
        elif (function == 'subgroup'):
            data = read_csv(source)
            measure = input('Type measure: ')
            # measure = 'axisr'
            if decimals is None:
                subgroup(data, group, measure)
            else:
                subgroup(data, group, measure,
                         ticks_number=decimals[:2], decimals=decimals[2:])
        else:
            print("Undefined function")
            logger.error("Undefined function")
    else:
        logger.error(
            f'The path "{source}" is not a valid source! Exiting...')


if __name__ == "__main__":
    le.setup_logging()
    logger = le.logging.getLogger('default')
    main()
