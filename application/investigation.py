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
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import logendo as le
from timeit import default_timer as timer
import sys


def plot_style():
    # Define style plots
    sns.set(context='notebook', style='ticks', palette='colorblind', font='Arial',
            font_scale=2, rc={'axes.grid': True, 'grid.linestyle': 'dashed',
                              'lines.linewidth': 2, 'xtick.direction': 'in',
                              'ytick.direction': 'in', 'figure.figsize': (7, 3.09017)})


def ext_mode(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m], counts[m]


def find_files(source, pattern=['.csv']):
    cmd = f'find {source} -type f -name "*{pattern[0]}"'
    for patt in pattern[1:]:
        cmd += f' -o -name "*{patt}"'
    logger.debug(f'Find command: {cmd}')

    import subprocess
    output = subprocess.run(cmd, shell=True,  stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    paths = output.stdout.splitlines()
    paths.sort()
    logger.debug(f'No. of files: {len(paths)}. Files found: {paths}')
    return paths


def rm_noise_frame(histogram, frame_threshold=75, light_max_freq=0.95, dark_max_freq=0.988):
    frame_freq_hist = np.empty((0, 256), np.float)
    rm_list = []
    for index, hist in enumerate(histogram):
        hist_freq = hist / np.sum(hist)
        if np.sum(hist_freq[frame_threshold:]) > light_max_freq:
            rm_list.append(index)
            logger.debug(
                f'Ligth frame noise - idx {index}, freq: {np.sum(hist_freq[frame_threshold:]):.3f}')
        elif np.sum(hist_freq[:frame_threshold]) > dark_max_freq:
            rm_list.append(index)
            logger.debug(
                f'Dark frame noise - idx {index}, freq: {np.sum(hist_freq[frame_threshold]):.3f}')
        else:
            frame_freq_hist = np.vstack([frame_freq_hist, hist_freq])
    clean_histogram = np.delete(histogram, rm_list, axis=0)
    logger.info(f'No. frames removed: {len(rm_list)}. List: {rm_list}')
    return clean_histogram, frame_freq_hist


def count_to_intensities(histogram):
    intensities = []
    for intensity in range(256):
        intensities.extend(np.repeat(intensity, int(histogram[intensity])))
    return intensities
    
    
def histogram_distance(ref_histogram, hist_freq_frames, video_interval, path):
    mean_hist = ref_histogram.mean(axis=0)
    histogram_plot(mean_hist, path, 'Healthy')
    ref_hist_freq = mean_hist / np.sum(mean_hist)
    hellinger_list = []
    for idx in range(len(video_interval)-1):
        distances = []
        logger.debug(
            f'HD-Idx {idx}: interval[{video_interval[idx]},{video_interval[idx+1]}]')
        for hist in hist_freq_frames[video_interval[idx]:video_interval[idx+1]]:
            dist = hellinger_distance(hist, ref_hist_freq)
            distances.append(dist)
        hellinger_list.append(distances)
    return np.array([np.array(dist_list) for dist_list in hellinger_list])


def distance_distribution_plot(first_distribution, second_distribution, path, name='HT-hellinger-distibution'):
    aux_list = np.hstack([first_distribution, second_distribution])
    x_ticks = ticks_interval(aux_list, 5, 2)
    logger.debug(
        f'N-Range: {np.min(first_distribution):.3f}..{np.max(first_distribution):.3f}')
    logger.debug(
        f'T-Range: {np.min(second_distribution):.3f}..{np.max(second_distribution):.3f}')
    plot_style()
    _, ax = plt.subplots(1)
    densities_list = []
    densities_list.append(ax.hist(first_distribution, density=True,
                                  bins=x_ticks, alpha=.85, label='Healthy')[0])
    densities_list.append(ax.hist(second_distribution, density=True,
                                  bins=x_ticks, alpha=.85, label='Tumor')[0])
    ax.set(title='Hellinger distance distribution', ylabel="Density", xlabel='Hellinger distance', xticks=x_ticks,
           yticks=ticks_interval(densities_list, 6, 1))
    plt.legend(loc='upper right', prop={'size': 12})
    plot = ax.get_figure()
    plot.canvas.draw()
    for index, label in enumerate(ax.get_yticklabels()):
        if index % 2 == 1:
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.savefig(f'{path}/{name}.png', dpi=600, bbox_inches="tight")
    entropy, max_entropy, degree_disorder = shannon_entropy(
        densities_list[0], x_ticks)

    logger.debug(
        f'Shannon Normal - E:{entropy:.2f}/{max_entropy:.2f} DD:{degree_disorder:.2f}')

    entropy, max_entropy, degree_disorder = shannon_entropy(
        densities_list[1], x_ticks)
    logger.debug(
        f'Shannon Tumor - E:{entropy:.2f}/{max_entropy:.2f} DD:{degree_disorder:.2f}')
    return 0


def histogram_plot(histogram, path, name):
    # barplot = []
    # for intensity in range(256):
    #     barplot.extend(
    #         np.repeat(intensity, int(histogram[intensity])))
    barplot = count_to_intensities(histogram)
    plot_style()
    _, ax = plt.subplots(1)
    density = ax.hist(barplot, density=True, bins=256, alpha=.85)[0]
    ax.set(title=f'Intensity histogram', ylabel="Relative frequency", xlabel='Pixel intensity', xticks=np.arange(0, 255, 50),
           yticks=ticks_interval(density, 6, 3))
    plot = ax.get_figure()
    plot.canvas.draw()
    for index, label in enumerate(ax.get_yticklabels()):
        if index % 2 == 1:
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.savefig(f'{path}/{name}.png',
                dpi=600, bbox_inches="tight")
    plt.clf()


def build_ref_mean(histogram, video_intervals):
    mean_histograms = np.empty((0, 256), np.float)
    for idx in range(len(video_intervals)-1):
        logger.debug(
            f'Index {idx}: interval[{video_intervals[idx]},{video_intervals[idx+1]}]')
        mean_histograms = np.vstack([mean_histograms, np.mean(
            histogram[video_intervals[idx]:video_intervals[idx+1]], axis=0)])
    logger.debug(f'Mean output: {mean_histograms.shape}')
    return mean_histograms


def histogram_multplot(histogram, path_list, histogram_index):
    mean_histograms = build_ref_mean(histogram, histogram_index)
    for idx, hist in enumerate(mean_histograms):
        histogram_plot(hist, path_list[idx].parents[0], path_list[idx].stem)

def intensity_qq_plot(healthy_quantils, tumor_quantils, path, name):
    _, ax = plt.subplots(figsize=(7,7))
    plt.plot([8, 256], [8, 256], '--', color = 'r')
    plt.scatter(healthy_quantils, tumor_quantils)
    ticks = [10,25,50,100,200]
    ax.set(title='Intensity Q-Q plot', ylabel="Tumor", xlabel='Healthy',xscale="log",yscale="log", xticks=ticks, yticks=ticks)
    plt.xlim(7, 275)
    plt.ylim(7, 275)
    plt.savefig(f'{path}/{name}-intensity-qq-plot.png', dpi=600, bbox_inches='tight')

def intensity_qq_multplot(healthy_hist, tumor_hist, path_list, tumor_index, num_quant=25):
    logger.debug(f'H: {healthy_hist.shape}')
    # normal global
    healthy_ref = build_ref_mean(healthy_hist, [0, len(healthy_hist)-1])
    logger.debug(f'Href: {healthy_ref.shape}')
    # calcula quantiles do normal
    # healthy = []
    # for intensity in range(256):
    #     healthy.extend(np.repeat(intensity, int(healthy_ref[0][intensity])))
    healthy = count_to_intensities(healthy_ref[0])
    healthy_quantils = np.quantile(healthy, np.linspace(0,1,num_quant+1)[1:])
    tumor_ref_hists = build_ref_mean(tumor_hist, tumor_index)
    # Para cada tumoral:
    logger.debug(f'Tref: {tumor_ref_hists.shape}; PL: {path_list}')
    for idx, hist in enumerate(tumor_ref_hists):
        # tumor = []
        # for intensity in range(256):
        #     tumor.extend(np.repeat(intensity, int(hist[intensity])))
        tumor = count_to_intensities(hist)
    # calcula quantile do tumoral
        tumor_quantils = np.quantile(tumor, np.linspace(0,1,num_quant+1)[1:])
    # faz grafico normal-tumoral
        intensity_qq_plot(healthy_quantils, tumor_quantils, path_list[idx].parents[0], path_list[idx].stem)


def fractal_distribution_plot(first_distribution, second_distribution, path, name='HT-fractal-distibution'):
    logger.debug('Initialize fractal')
    logger.debug(
        f'(0) H-Range: {np.min(first_distribution):.3f}..{np.max(first_distribution):.3f}')
    logger.debug(
        f'(0) T-Range: {np.min(second_distribution):.3f}..{np.max(second_distribution):.3f}')
    first_distribution = first_distribution[first_distribution >= 1.0]
    second_distribution = second_distribution[second_distribution >= 1.0]
    logger.debug(
        f'(1) H-Range: {np.min(first_distribution):.3f}..{np.max(first_distribution):.3f}')
    logger.debug(
        f'(1) T-Range: {np.min(second_distribution):.3f}..{np.max(second_distribution):.3f}')

    aux_list = np.hstack([first_distribution, second_distribution])
    x_ticks = ticks_interval(aux_list, 5, 2)
    logger.debug(
        f'Range - aux_list: {np.min(aux_list):.3f}..{np.max(aux_list):.3f}, x_ticks: {np.min(x_ticks):.2f}..{np.max(x_ticks):.2f} ')
    plot_style()
    _, ax = plt.subplots(1)
    densities_list = []
    densities_list.append(ax.hist(first_distribution, density=True,
                                  bins=x_ticks, alpha=.85, label='Healthy')[0])
    logger.debug(
        f'(0) Densities: {len(densities_list)} : {len(densities_list[0])}')
    densities_list.append(ax.hist(second_distribution, density=True,
                                  bins=x_ticks, alpha=.85, label='Tumor')[0])
    logger.debug(
        f'(1) Densities: {len(densities_list)} : {len(densities_list[1])}')
    ax.set(title='Fractal dimension distribution', ylabel="Density", xlabel='Fractal dimension', xticks=x_ticks,
           yticks=ticks_interval(densities_list, 6, 2))
    ax.set_xticklabels(np.round(x_ticks, 2), size=16)
    plt.legend(loc='upper right', prop={'size': 12})
    plot = ax.get_figure()
    plot.canvas.draw()
    for index, label in enumerate(ax.get_yticklabels()):
        if index % 2 == 1:
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.savefig(f'{path}/{name}.png', dpi=600, bbox_inches="tight")
    entropy, max_entropy, degree_disorder = shannon_entropy(
        densities_list[0], x_ticks)

    logger.debug(
        f'Shannon Healthy - E:{entropy:.2f}/{max_entropy:.2f} DD:{degree_disorder:.2f}')

    entropy, max_entropy, degree_disorder = shannon_entropy(
        densities_list[1], x_ticks)
    logger.debug(
        f'Shannon Tumor - E:{entropy:.2f}/{max_entropy:.2f} DD:{degree_disorder:.2f}')
    return 0


def full_hist_analysis(source, clean=True):
    start_time = timer()
    files_path = find_files(source, ['-frame-histogram.csv'])
    files_frac = find_files(source, ['-fractal-dimension.csv'])
    flag = False
    prev_idx = None
    healthy_ref_hist = np.empty((0, 256), np.int32)
    healthy_ref_freq = np.empty((0, 256), np.float32)
    healthy_fractal = []
    healthy_idx = [0]
    tumor_ref_hist = np.empty((0, 256), np.int32)
    tumor_ref_freq = np.empty((0, 256), np.float32)
    tumor_fractal = []
    tumor_idx = [0]
    path_list = []
    import pathlib
    for index, (fpath, ffrac) in enumerate(zip(files_path, files_frac)):
        logger.debug(f'Histogram file path: {fpath}. Fractal file: {ffrac}')
        path_list.append(pathlib.Path(fpath))
        hist = pd.read_csv(fpath, header=None).values
        frac = pd.read_csv(ffrac, header=None).values
        frac = frac[~np.isnan(frac)]
        if clean:
            hist, hist_freq = rm_noise_frame(hist)
        else:
            hist_freq = np.empty((0, 256), np.float32)
            for hist_frame in hist:
                hist_frame_freq = hist_frame / np.sum(hist_frame)
                hist_freq = np.vstack([hist_freq, hist_frame_freq])

        if '/0/' in fpath:
            if flag:
                plot_path = path_list[prev_idx].parents[2]
                hellinger_HH = histogram_distance(
                    healthy_ref_hist, healthy_ref_freq, healthy_idx, plot_path)
                hellinger_HT = histogram_distance(
                    healthy_ref_hist, tumor_ref_freq, tumor_idx, plot_path)

                distance_distribution_plot(
                    np.hstack(hellinger_HH), np.hstack(hellinger_HT), plot_path)

                fractal_distribution_plot(np.hstack(healthy_fractal),
                                          np.hstack(tumor_fractal), plot_path)

                histogram_multplot(healthy_ref_hist,
                                   path_list[prev_idx:(prev_idx+len(healthy_idx)-1)], healthy_idx)
                histogram_multplot(tumor_ref_hist,
                                   path_list[(prev_idx+len(healthy_idx)-1):index], tumor_idx)

                logger.debug(f'PLX: {path_list}')
                intensity_qq_multplot(healthy_ref_hist, tumor_ref_hist, path_list[(prev_idx+len(healthy_idx)-1):index], tumor_idx) 

                mean_tumor_hist = tumor_ref_hist.mean(axis=0)
                histogram_plot(mean_tumor_hist, plot_path, 'Tumor')

                num_quant = 25
                tumor_intensities = count_to_intensities(mean_tumor_hist)
                tumor_quantils = np.quantile(tumor_intensities, np.linspace(0,1,num_quant+1)[1:])
                
                mean_healthy_hist = healthy_ref_hist.mean(axis=0)
                healthy_intensities = count_to_intensities(mean_healthy_hist)
                healthy_quantils = np.quantile(healthy_intensities, np.linspace(0,1,num_quant+1)[1:])

                intensity_qq_plot(healthy_quantils, tumor_quantils, plot_path, name='HT')
                
                # intensity_qq_multplot(healthy_ref_hist, tumor_ref_hist, path_list[(prev_idx+len(healthy_idx)-1):index], tumor_idx) 

                for idx, hellinger_distr in enumerate(hellinger_HT):
                    distance_distribution_plot(
                        np.hstack(hellinger_HH), np.hstack(
                            hellinger_distr), path_list[prev_idx+idx+len(healthy_idx)-1].parents[0],
                        f'{path_list[prev_idx+idx+len(healthy_idx)-1].stem[:-16]}-hellinger-distribution')

                for idx, fd_list in enumerate(tumor_fractal):
                    fractal_distribution_plot(np.hstack(healthy_fractal),
                                              np.asarray(
                                                  fd_list), path_list[prev_idx+idx+len(healthy_idx)-1].parents[0],
                                              f'{path_list[prev_idx+idx+len(healthy_idx)-1].stem[:-16]}-fractal-dimension')

                flag = False
                healthy_ref_hist = np.empty((0, 256), np.int32)
                healthy_ref_freq = np.empty((0, 256), np.float32)
                healthy_fractal = []
                healthy_idx = [0]
                tumor_ref_hist = np.empty((0, 256), np.int32)
                tumor_ref_freq = np.empty((0, 256), np.float32)
                tumor_fractal = []
                tumor_idx = [0]
            healthy_ref_hist = np.vstack([healthy_ref_hist, hist])
            healthy_ref_freq = np.vstack([healthy_ref_freq, hist_freq])
            healthy_idx.append(healthy_idx[-1]+len(hist))
            healthy_fractal.append(frac)
            prev_idx = index
        elif '/1/' in fpath:
            flag = True
            tumor_ref_hist = np.vstack([tumor_ref_hist, hist])
            tumor_ref_freq = np.vstack([tumor_ref_freq, hist_freq])
            tumor_idx.append(tumor_idx[-1]+len(hist))
            tumor_fractal.append(frac)
        else:
            logger.error('Incorrect structure! Exiting...')
            sys.exit()

    plot_path = path_list[prev_idx].parents[2]
    hellinger_HH = histogram_distance(
        healthy_ref_hist, healthy_ref_freq, healthy_idx, plot_path)
    hellinger_HT = histogram_distance(
        healthy_ref_hist, tumor_ref_freq, tumor_idx, plot_path)

    distance_distribution_plot(
        np.hstack(hellinger_HH), np.hstack(hellinger_HT), plot_path)

    fractal_distribution_plot(np.hstack(healthy_fractal),
                              np.hstack(tumor_fractal), plot_path)

    histogram_multplot(healthy_ref_hist,
                       path_list[prev_idx:(prev_idx+len(healthy_idx)-1)], healthy_idx)
    histogram_multplot(tumor_ref_hist,
                       path_list[(prev_idx+len(healthy_idx)-1):index+1], tumor_idx)

    logger.debug(f'PLY: {path_list}')
    intensity_qq_multplot(healthy_ref_hist, tumor_ref_hist, path_list[(prev_idx+len(healthy_idx)-1):index+1], tumor_idx)
    
    mean_tumor_hist = tumor_ref_hist.mean(axis=0)
    histogram_plot(mean_tumor_hist, plot_path, 'Tumor')

    num_quant = 25
    tumor_intensities = count_to_intensities(mean_tumor_hist)
    tumor_quantils = np.quantile(tumor_intensities, np.linspace(0,1,num_quant+1)[1:])
                
    mean_healthy_hist = healthy_ref_hist.mean(axis=0)
    healthy_intensities = count_to_intensities(mean_healthy_hist)
    healthy_quantils = np.quantile(healthy_intensities, np.linspace(0,1,num_quant+1)[1:])

    intensity_qq_plot(healthy_quantils, tumor_quantils, plot_path, name='HT')

    for idx, hellinger_distr in enumerate(hellinger_HT):
        distance_distribution_plot(np.hstack(hellinger_HH), np.hstack(hellinger_distr),
                                   path_list[prev_idx+idx+len(healthy_idx)-1].parents[0], f'{path_list[prev_idx+idx+len(healthy_idx)-1].stem[:-16]}-hellinger-distribution')

    for idx, fd_list in enumerate(tumor_fractal):
        fractal_distribution_plot(np.hstack(healthy_fractal), np.asarray(fd_list),
                                  path_list[prev_idx+idx+len(healthy_idx)-1].parents[0], f'{path_list[prev_idx+idx+len(healthy_idx)-1].stem[:-16]}-fractal-dimension')

    end_time = timer()
    logger.debug(f'Time elpased: {end_time-start_time:.2f}')
    return 0


def full_hist_analysisOLD(source):
    import subprocess
    output = subprocess.run(f'find {source} -type f -name "*-frame-histogram.csv"', shell=True,  stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    files = output.stdout.splitlines()
    files.sort()
    global_norm = np.zeros((1, 256), dtype=np.int)
    global_tumor = np.zeros((1, 256), dtype=np.int)
    for fpath in files:
        if '/0/' in fpath:
            global_norm = np.vstack(
                [global_norm, pd.read_csv(fpath, header=None).values])
        else:
            global_tumor = np.vstack(
                [global_tumor, pd.read_csv(fpath, header=None).values])
    global_norm = global_norm[1:]
    global_tumor = global_tumor[1:]
    logger.debug(
        f'Shape - norm: {global_norm.shape}, tumor: {global_tumor.shape}')
    global_norm_mean = global_norm.mean(axis=0)
    glob_norm_freq = global_norm_mean / np.sum(global_norm_mean)

    hellinger_global = []
    for hist in global_norm:
        hist_freq = hist / np.sum(hist)
        hellinger_global.append(hellinger_distance(glob_norm_freq, hist_freq))
    norm_threshold = np.mean(hellinger_global) + np.std(hellinger_global)
    logger.debug(f'Normal Hellinger - Threshold: {norm_threshold:.2f} '
                 f'| Mean: {np.mean(hellinger_global):.2f} | '
                 f'STD:{np.std(hellinger_global):.2f}')

    global_tumor_mean = global_tumor.mean(axis=0)
    glob_tumor_freq = global_tumor_mean / np.sum(global_tumor_mean)

    barplot_hist_N = []
    barplot_hist_T = []
    for pix in range(256):
        barplot_hist_N.extend(
            np.repeat(pix, int(global_norm_mean[pix])).tolist())
        barplot_hist_T.extend(
            np.repeat(pix, int(global_tumor_mean[pix])).tolist())

    logger.debug(f'Normal - Mean: {np.mean(barplot_hist_N):.2f}, STD: {np.std(barplot_hist_N):.2f}, '
                 f' Median: {np.median(barplot_hist_N):.2f}, Mode: {ext_mode(barplot_hist_N)[0]:.2f}')
    logger.debug(f'Tumor - Mean: {np.mean(barplot_hist_T):.2f}, STD: {np.std(barplot_hist_T):.2f}, '
                 f' Median: {np.median(barplot_hist_T):.2f}, Mode: {ext_mode(barplot_hist_T)[0]:.2f}')

    plot_style()
    _, ax = plt.subplots(1)
    # plt.plot(glob_norm_freq)
    density = ax.hist(barplot_hist_N, density=True,
                      bins=255, alpha=.85, label='Tumor')[0]
    ax.set(title='Normal frames', ylabel="Relative frequency", xlabel='Pixel intensity', xticks=np.arange(0, 255, 50),
           yticks=ticks_interval(density, 6, 3))
    # plt.hist(barplot_hist_N, bins=255, density=True)
    # plt.xlabel('Pixel intensity')
    # plt.ylabel('Frequency')
    # plt.title(r'Normal frames pixel distribuion')
    plot = ax.get_figure()
    plot.canvas.draw()
    for index, label in enumerate(ax.get_yticklabels()):
        if index % 2 == 1:
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.savefig('glob_norm_hist.png', dpi=600, bbox_inches="tight")
    plt.clf()
    plot_style()
    _, ax = plt.subplots(1)
    density = ax.hist(barplot_hist_T, density=True,
                      bins=255, alpha=.85, label='Tumor')[0]
    ax.set(title='Tumor frames', ylabel="Relative frequency", xlabel='Pixel intensity', xticks=np.arange(0, 255, 50),
           yticks=ticks_interval(density, 6, 3))
    # plt.hist(barplot_hist_N, bins=255, density=True)
    # plt.xlabel('Pixel intensity')
    # plt.ylabel('Frequency')
    # plt.title(r'Normal frames pixel distribuion')
    plot = ax.get_figure()
    plot.canvas.draw()
    for index, label in enumerate(ax.get_yticklabels()):
        if index % 2 == 1:
            label.set_visible(True)
        else:
            label.set_visible(False)
    # plt.plot(glob_tumor_freq)
    # plt.hist(barplot_hist_T, bins=255, density=True)
    # plt.xlabel('Pixel intensity')
    # plt.ylabel('Frequency')
    # plt.title(r'Tumor frames pixel distribuion')
    plt.savefig('glob_tumor_hist.png', dpi=600, bbox_inches="tight")
    plt.clf()
    logger.debug(
        f'Hellinger: {hellinger_distance(glob_norm_freq, glob_tumor_freq):.3f}')
    # exit()

    import pathlib as pl
    list_hell_NN = []
    for fpath in files[73:74]:
        frames_hist = pd.read_csv(fpath, header=None)
        count = 0
        for index, hist in frames_hist.iterrows():
            hist_freq = hist / np.sum(hist)
            dist_hell_N = hellinger_distance(glob_norm_freq, hist_freq)
            dist_hell_T = hellinger_distance(glob_tumor_freq, hist_freq)
            if dist_hell_N > norm_threshold:
                count += 1
                # frames_list.append(index)
            list_hell_NN.append(dist_hell_N)
            # list_hell_NN.append(dist_hell_T)
        logger.debug(
            f'Range: {np.min(list_hell_NN):.3f}..{np.max(list_hell_NN):.3f}')
        path = pl.Path(fpath)
        #
        frames_mean_hist_Y = np.mean(frames_hist, axis=0)
        barplot_hist_Y = []
        for pix in range(256):
            barplot_hist_Y.extend(
                np.repeat(pix, int(frames_mean_hist_Y[pix])).tolist())
        plot_style()
        _, ax = plt.subplots(1)
        density = ax.hist(barplot_hist_Y, density=True,
                          bins=255, alpha=.85, label='Tumor')[0]
        ax.set(title='Normal video', ylabel="Relative frequency", xlabel='Pixel intensity', xticks=np.arange(0, 255, 50),
               yticks=ticks_interval(density, 6, 3))
        plot = ax.get_figure()
        plot.canvas.draw()
        for index, label in enumerate(ax.get_yticklabels()):
            if index % 2 == 1:
                label.set_visible(True)
            else:
                label.set_visible(False)
        plt.savefig(f'{path.stem}.png', dpi=600, bbox_inches="tight")
        plt.clf()
        if True:
            continue

    tumor_videos = []
    frames_hist_TT = []
    list_hell_TT = []
    cont = 0
    for fpath in files[74:76]:
        cont += 1
        frames_hist = pd.read_csv(fpath, header=None)
        path = pl.Path(fpath)
        #
        frames_mean_hist = np.mean(frames_hist)
        count = 0
        for index, hist in frames_hist.iterrows():
            hist_freq = hist / np.sum(hist)
            dist_hell_N = hellinger_distance(glob_norm_freq, hist_freq)
            dist_hell_T = hellinger_distance(glob_tumor_freq, hist_freq)
            if dist_hell_N > norm_threshold:
                count += 1
                # frames_list.append(index)
            list_hell_TT.append(dist_hell_N)
            # list_hell_TT.append(dist_hell_T)

        frames_hist_TT.append(frames_mean_hist)
        barplot_hist_X = []
        for pix in range(256):
            barplot_hist_X.extend(
                np.repeat(pix, int(frames_mean_hist[pix])).tolist())
        tumor_videos.extend(barplot_hist_X)
        plot_style()
        _, ax = plt.subplots(1)
        density = ax.hist(barplot_hist_X, density=True,
                          bins=255, alpha=.85, label='Tumor')[0]
        ax.set(title='Tumor video', ylabel="Density", xlabel='Pixel intensity', xticks=np.arange(0, 255, 50),
               yticks=ticks_interval(density, 6, 3))
        plot = ax.get_figure()
        plot.canvas.draw()
        for index, label in enumerate(ax.get_yticklabels()):
            if index % 2 == 1:
                label.set_visible(True)
            else:
                label.set_visible(False)
        plt.savefig(f'{path.stem}-tumor.png', dpi=600, bbox_inches="tight")
        plt.clf()
        if True:
            if cont > 1:
                plot_style()
                _, ax = plt.subplots(1)
                density = ax.hist(barplot_hist_X, density=True,
                                  bins=255, alpha=.85, label='Tumor')[0]
                ax.set(title='Tumor videos', ylabel="Relative frequency", xlabel='Pixel intensity', xticks=np.arange(0, 255, 50),
                       yticks=ticks_interval(density, 6, 3))
                plot = ax.get_figure()
                plot.canvas.draw()
                for index, label in enumerate(ax.get_yticklabels()):
                    if index % 2 == 1:
                        label.set_visible(True)
                    else:
                        label.set_visible(False)
                plt.savefig('tumor-video.png', dpi=600, bbox_inches="tight")
                plt.clf()
                meanTT = np.mean(frames_hist_TT, axis=0)
                freq_TT = meanTT / np.sum(meanTT)
                freq_Y = frames_mean_hist_Y / np.sum(frames_mean_hist_Y)

                logger.debug(
                    f'Hellinger N-T: {hellinger_distance(freq_Y, freq_TT):.2f}')
                # # list_hell_T = []
                # for index, hist in frames_hist.iterrows():
                #     hist_freq = hist / np.sum(hist)
                #     dist_hell_N = hellinger_distance(glob_norm_freq, hist_freq)
                #     dist_hell_T = hellinger_distance(glob_tumor_freq, hist_freq)
                #     if dist_hell_N > norm_threshold:
                #         count += 1
                #         # frames_list.append(index)
                #     # list_hell_N.append(dist_hell_N)
                #     list_hell_T.append(dist_hell_TT)

                densities_list = []
                aux_list = list_hell_NN.copy()
                aux_list.extend(list_hell_TT)
                plot_style()
                _, ax = plt.subplots(1)
                # data_ticks = np.vstack([list_hell_NN, list_hell_TT])
                data_ticks = aux_list
                x_ticks = ticks_interval(data_ticks, 5, 2)
                logger.debug(
                    f'N-Range: {np.min(list_hell_NN):.3f}..{np.max(list_hell_NN):.3f}')
                logger.debug(
                    f'T-Range: {np.min(list_hell_TT):.3f}..{np.max(list_hell_TT):.3f}')
                densities_list.append(ax.hist(list_hell_NN, density=True,
                                              bins=x_ticks, alpha=.85, label='Normal')[0])
                densities_list.append(ax.hist(list_hell_TT, density=True,
                                              bins=x_ticks, alpha=.85, label='Tumor')[0])
                ax.set(title='Hellinger distance distribution', ylabel="Density", xlabel='Hellinger distance', xticks=x_ticks,
                       yticks=ticks_interval(densities_list, 6, 1))
                plt.legend(loc='upper right', prop={'size': 12})
                plot = ax.get_figure()
                plot.canvas.draw()
                for index, label in enumerate(ax.get_yticklabels()):
                    if index % 2 == 1:
                        label.set_visible(True)
                    else:
                        label.set_visible(False)
                plt.savefig(
                    'N-T-helling_dist.png', dpi=600, bbox_inches="tight")
                entropy, max_entropy, degree_disorder = shannon_entropy(
                    densities_list[0], x_ticks)

                logger.debug(
                    f'Shannon Normal - E:{entropy:.2f}/{max_entropy:.2f} DD:{degree_disorder:.2f}')

                entropy, max_entropy, degree_disorder = shannon_entropy(
                    densities_list[1], x_ticks)
                logger.debug(
                    f'Shannon Tumor - E:{entropy:.2f}/{max_entropy:.2f} DD:{degree_disorder:.2f}')
            continue
        #
        list_hell_N = []
        list_hell_T = []
        count = 0
        frames_list = []
        logger.debug(f'File: {fpath}')
        for index, hist in frames_hist.iterrows():
            hist_freq = hist / np.sum(hist)
            dist_hell_N = hellinger_distance(glob_norm_freq, hist_freq)
            dist_hell_T = hellinger_distance(glob_tumor_freq, hist_freq)
            if dist_hell_N > norm_threshold:
                count += 1
                frames_list.append(index)
            list_hell_N.append(dist_hell_N)
            list_hell_T.append(dist_hell_T)

        densities_list = []
        plot_style()
        _, ax = plt.subplots(1)
        data_ticks = np.vstack([list_hell_N, list_hell_T])
        x_ticks = ticks_interval(data_ticks, 5, 2)
        densities_list.append(ax.hist(list_hell_N, density=True,
                                      bins=x_ticks, alpha=.85, label='Normal')[0])
        densities_list.append(ax.hist(list_hell_T, density=True,
                                      bins=x_ticks, alpha=.85, label='Tumor')[0])
        ax.set(title='Hellinger distance distribution', ylabel="Density", xlabel='Hellinger distance', xticks=x_ticks,
               yticks=ticks_interval(densities_list, 6, 1))
        plt.legend(loc='upper right', prop={'size': 12})
        plot = ax.get_figure()
        plot.canvas.draw()
        for index, label in enumerate(ax.get_yticklabels()):
            if index % 2 == 1:
                label.set_visible(True)
            else:
                label.set_visible(False)
        plt.savefig(
            f'{path.parents[0]}/{path.stem}-helling_dist.png', dpi=600, bbox_inches="tight")
        entropy, max_entropy, degree_disorder = shannon_entropy(
            densities_list[0], x_ticks)

        logger.debug(
            f'Shannon Normal - E:{entropy:.2f}/{max_entropy:.2f} DD:{degree_disorder:.2f}')

        entropy, max_entropy, degree_disorder = shannon_entropy(
            densities_list[1], x_ticks)
        logger.debug(
            f'Shannon Tumor - E:{entropy:.2f}/{max_entropy:.2f} DD:{degree_disorder:.2f}')

        logger.debug(
            f'Frames above the Normal threshold: {count:4d} / {frames_hist.shape[0]:4d} '
            f'| %: {(count/frames_hist.shape[0])*100:.2f} | Frames: frames_list')
        logger.debug(
            f'NORMAL - Range: {np.min(list_hell_N):.3f} .. {np.max(list_hell_N):.3f} | ' f'Mean: {np.mean(list_hell_N):.3f} | ')
        logger.debug(
            f'TUMOR - Range: {np.min(list_hell_T):.3f} .. {np.max(list_hell_T):.3f} | ' f'Mean: {np.mean(list_hell_T):.3f} | ')


def histogram_analysis(data):
    plot_style()
    _, ax = plt.subplots(1)
    black_area = min(data.iloc[:, 0])
    mean_data = data.mean(axis=0)
    mean_data[0] = mean_data[0] - black_area
    # np.savetxt('mean.txt', mean_data, fmt='%d')
    # x = np.arange(256)
    # dict_data = dict(zip(x, mean_data))
    # keys = dict_data.keys()
    # vals = dict_data.values()
    # freq = list(vals) / np.sum(list(vals))
    glob_freq = mean_data / np.sum(mean_data)
    list_hell = []
    # for index, hist in enumerate(data):
    data = pd.read_csv(
        'midia/main/HIST/001/1/frame/005-2016-frame-histogram.csv', header=None)
    # data = pd.read_csv('midia/main/HIST/001/1/frame/012-2016-frame-histogram.csv', header=None)
    # data = pd.read_csv('midia/main/HIST/001/1/frame/017-2016-frame-histogram.csv', header=None)
    # data = pd.read_csv('midia/main/HIST/006/1/frame/001-2017-frame-histogram.csv', header=None)
    # data = pd.read_csv('midia/main/HIST/006/1/frame/005-2017-frame-histogram.csv', header=None)
    # data = pd.read_csv('midia/main/HIST/006/1/frame/007-2017-frame-histogram.csv', header=None)
    # data = pd.read_csv('midia/main/HIST/006/1/frame/013-2017-frame-histogram.csv', header=None)
    count = 0
    for index, hist in data.iterrows():
        # logger.debug(f'Hist {index}: {hist}')
        hist[0] = hist[0] - black_area
        hist_freq = hist / np.sum(hist)
        dist_hell = hellinger_distance(glob_freq, hist_freq)
        if dist_hell > 0.3:
            count += 1
        entropy, max_entropy, degree_disorder = shannon_entropy(hist_freq, [
            0, 1, 255])
        list_hell.append(dist_hell)
        logger.debug(
            f'Frame: {index:4d} | H: {dist_hell:.3f} | S: {entropy:.3f}/{max_entropy:.3f}')
    logger.debug(
        f'Frames above the threshold: {count:4d} / {data.shape[0]:4d} | %: {(count/data.shape[0])*100:.2f}')
    logger.debug(f'Range: {np.min(list_hell):.3f} .. {np.max(list_hell):.3f} | '
                 f'Mean: {np.mean(list_hell):.3f} | ')
    logger.debug(
        f'Soma: {np.sum(mean_data)} | Soma freq: {np.sum(glob_freq)} | Freq: glob_freq')
    # plt.bar(list(keys), list(vals))
    # sns.barplot(x=list(keys), y=list(vals))
    # plt.plot(list(vals))
    plt.plot(glob_freq)
    plt.savefig('glob_hist.png', dpi=600, bbox_inches="tight")
    plt.clf()
    plt.hist(list_hell)
    plt.savefig('helling_dist.png', dpi=600, bbox_inches="tight")
    # plt.show()


def full_stat(source, outliers=False):
    start_time = timer()
    logger.debug('Initializing STAT csv data')

    import subprocess
    import pathlib

    measure_list = ['axisr', 'dist', 'elong', 'feret',
                    'min_dist', 'perim', 'round', 'spher', 'wall']

    data_export = [["Parameter", "Mean", "STD"]]
    for measure in measure_list:
        output = subprocess.run(f'find {source} -type f -name "{measure}_data.csv"', shell=True,  stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True)
        files = output.stdout.splitlines()
        files.sort()
        logger.debug(f'Measure: {measure}. No. of csv files found: {len(files)}'
                     f'. Files: {files}')

        prev_path = pathlib.Path(files[0])
        data = [read_csv(prev_path)[0]]
        data_id = read_csv(prev_path)[1]
        prev_path = pathlib.Path(files[0]).parents[2]
        data_id.insert(0, prev_path)
        sum_crypts = 0
        for f_path in files[1:]:
            path = pathlib.Path(f_path).parents[2]
            if prev_path == path:
                # if str(prev_path)[:-1] == str(path)[:-1]:
                dta = read_csv(f_path)[1]
                data_id.extend(dta)
            else:
                sum_crypts += len(data_id)-1
                # logger.debug(f'ID: {np.asarray(data_id[1:]).astype("float")}')
                logger.debug(
                    f'Path: {prev_path}. No Crypts: {len(data_id)-1}. Mean: {np.mean(np.asarray(data_id[1:]).astype("float")):.2f}')
                data.append(data_id)
                data_id = read_csv(f_path)[1]
                data_id.insert(0, path)
            prev_path = path

        sum_crypts += len(data_id)-1
        logger.debug(
            f'Path: {prev_path}. No Crypts: {len(data_id)-1}. Mean: {np.mean(np.asarray(data_id[1:]).astype("float")):.2f}')
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
    group_float = [float(arr) for arr in group_data]
    complementary_float = [float(arr) for arr in complementary_data]
    # logger.debug(f'G: {group_float}')
    if not outliers:
        group_float = rm_outliersG(measure, group_float)
        complementary_float = rm_outliersG(measure, complementary_float)
    logger.debug(
        f'No. crypts - Group: {len(group_float)} Complementary: {len(complementary_float)}'
        f' | Mean ± std - Group: {np.mean(group_float):.2f}±{np.std(group_float):.2f}'
        f' Complementary: {np.mean(complementary_float):.2f}±{np.std(complementary_float):.2f}')

    # Gráfico

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
    # for index, img_data in enumerate(data_float[: 3]):
    for index, img_data in enumerate(data_float[: 2]):
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
    logger.info('Hellinger distance: '
                f'{hellinger_distance(frequencies_list[0], frequencies_list[1]):.3f}')
    # logger.info('Hellinger distance GxR: '
    #             f'{hellinger_distance(frequencies_list[0], frequencies_list[1]):.3f}')
    # logger.info('Hellinger distance GxT: '
    #             f'{hellinger_distance(frequencies_list[0], frequencies_list[2]):.3f}')
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


def ticks_interval(data, quantity, decimals, pct=False):
    start_time = timer()
    # logger.debug('Initializing ticks interval')
    # max_value = max(map(max, data))
    max_value = np.max(data)
    # min_value = min(map(min, data))
    min_value = np.min(data)

    interval = np.round((max_value - min_value) / quantity, decimals)
    logger.debug(f'Min value: {min_value:.5f} | Max value: {max_value:.5f} | '
                 f'No. ticks: {quantity} | Decimal places: {decimals} | '
                 f'Interval: {interval:.5f}')

    ticks = np.round(np.arange(min_value, max_value +
                               interval, interval), decimals)
    # logger.debug('Pre-ticks: ' + str([f'{value:.5f}' for value in ticks]))

    if pct and max(ticks) > 100:
        min_tick = min(min(ticks), 100-interval*(quantity+1))
        ticks = np.round(np.arange(min_tick, 100.01, interval), decimals)
        logger.debug('Pre-ticks(%): ' +
                     str([f'{value:.5f}' for value in ticks]))
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
    # logger.debug('Finished ticks interval')
    end_time = timer()
    # logger.debug(
    #     f'Ticks interval function time elapsed: {end_time-start_time:.2f}s')
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
        elif (function == "stat"):
            full_stat(source)
        elif (function == 'subgroup'):
            data = read_csv(source)
            measure = input('Type measure: ')
            # measure = 'axisr'
            if decimals is None:
                subgroup(data, group, measure)
            else:
                subgroup(data, group, measure,
                         ticks_number=decimals[:2], decimals=decimals[2:])
        elif (function == "frame-hist"):
            data = pd.read_csv(source, header=None)
            if decimals is None:
                histogram_analysis(data)
            else:
                box_plot(data, ticks_number=decimals[0], decimals=decimals[1])
        elif (function == 'full-frame'):
            full_hist_analysis(source)
        else:
            logger.error("Undefined function")
    else:
        logger.error(
            f'The path "{source}" is not a valid source! Exiting...')
        sys.exit()


if __name__ == "__main__":
    le.setup_logging()
    logger = le.logging.getLogger('default')
    main()
