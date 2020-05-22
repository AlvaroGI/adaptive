# -*- coding: utf-8 -*-

import random
from collections import defaultdict
import adaptive

import numpy as np
from time import sleep
from random import random
from adaptive.learner import AverageLearner, AverageLearner2D
import math
from functools import partial
from scipy.interpolate import interp1d
from scipy.integrate import quad

from matplotlib.ticker import ScalarFormatter, NullFormatter, LogFormatter

# For animations
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import copy
from IPython.display import HTML
import warnings

#____________________________________________________________________
#_____________________________TESTS__________________________________
#____________________________________________________________________
def test_single_NSR(learner, max_samples, final_plot=True, keep_init=False, title=None):
    '''Runs the learner until it contains max_samples samples.
       Then, calculates the NSR versus x.
       ---Input---
            learner: learner to test (learner)
            max_samples: maximum number of samples (int)
            final_plot: if True, plot NSR after running (bool)
            keep_init: if True, keep the initial state of the learner (bool)
            title: title of the plot (optional, string)'''

    if keep_init:
        learner1 = copy.deepcopy(learner)

    while learner.total_samples()<max_samples:
            xs, _ = learner.ask(1)
            for xx in xs:
                yy = learner.function(xx)
                learner.tell(xx,yy)

    NSR = calculate_NSR(learner)

    if final_plot:
        plt.plot(list(NSR.keys()),list(NSR.values()))
        plt.xlim(learner.bounds)
        plt.ylim([0,1])
        plt.xlabel('x')
        plt.ylabel('NSR(x)')
        if title:
            plt.title(title)

    if keep_init:
        learner = copy.deepcopy(learner1)

    return NSR

def test_single_ISR(learner, max_samples, final_plot=True, keep_init=False, title=None):
    '''Runs the learner until it contains max_samples samples.
       Then, calculates the ISR versus x.
       ---Input---
            learner: learner to test (learner)
            max_samples: maximum number of samples (int)
            final_plot: if True, plot NSR after running (bool)
            keep_init: if True, keep the initial state of the learner (bool)
            title: title of the plot (optional, string)'''

    if keep_init:
        learner1 = copy.deepcopy(learner)

    while learner.total_samples()<max_samples:
            xs, _ = learner.ask(1)
            for xx in xs:
                yy = learner.function(xx)
                learner.tell(xx,yy)

    ISR = calculate_NSR(learner)

    if final_plot:
        plt.plot(list(ISR.keys()),list(ISR.values()))
        plt.xlim(learner.bounds)
        plt.ylim([0,1])
        plt.xlabel('x')
        plt.ylabel('ISR(x)')
        if title:
            plt.title(title)

    if keep_init:
        learner = copy.deepcopy(learner1)

    return ISR

def test_single_error(learner, max_samples, errors=None, extrema=None, keep_init=False, return_errors=True, calculate_uniform=False,
                      fittings=True, generate_plot=True, ylim=None, save_plot=False, fig_name=None, progress_bars='notebook'):
    '''Runs the learner until it contains max_samples samples.
       Then, calculates the error and extrema NS and IS versus N.
       ---Input---
            learner: learner to test (learner)
            errors: dictionary containing the number of samples as key and error
                    between the real function and the interpolated one as value
                    ([0]: learner, [1]: uniform learner, [2]: number of data
                    points of the learner); optional (dict)
            extrema: dictionary containing the maximum NSR [0], minimum NSR [1],
                     maximum ISR [2], and minimum ISR [3]; optional (dict)
            max_samples: maximum number of samples (int)
            keep_init: if True, keep the initial state of the learner (bool)
            return_errors: set to True to return the errors (bool)
            calculate_uniform: set to True to calculate uniform learner errors
                               and plot them (bool)
            fittings: set to True to fit n=A*N^(1/3) to n(N) (bool)
            generate_plot: set to True to generate plots, either to show or
                           to save them (bool)
            ylim: y-limits of the function plot; if set to None, the limits are
                  automatic (tuple of floats)
            save_plot: set to True to save animation as .gif (bool)
            fig_name: name of the figure, only used if save_plot==True (str)
            progress_bars: set to 'simple' for Python progress bars, set to
                           'notebook' if running on a notebook, set to None for
                           no progress bars'''

    if progress_bars=='simple':
        from tqdm import tqdm
    elif progress_bars=='notebook':
        from tqdm.notebook import tqdm

    if save_plot and (not fig_name):
        raise ValueError('fig_name not specified.')
        assert isinstance(fig_name,str), 'fig_name must be str.'

    if keep_init:
        learner1 = copy.deepcopy(learner)

    if not errors:
        errors = [{},{},{}]

    if not extrema:
        extrema = [{},{},{},{}]

    # Run learner and calculate error
    N0 = learner.total_samples()
    if max_samples-N0>0:
        if progress_bars=='simple' or progress_bars=='notebook':
            for _ in tqdm(np.arange(max_samples-N0)):
                xs, _ = learner.ask(1)
                for xx in xs:
                    yy = learner.function(xx)
                    learner.tell(xx,yy)
        else:
            for _ in np.arange(max_samples-N0):
                xs, _ = learner.ask(1)
                for xx in xs:
                    yy = learner.function(xx)
                    learner.tell(xx,yy)
        errors[0][max_samples] = calculate_L1error(learner)
        errors[2][max_samples] = len(learner.data)
        maxNS, minNS = calculate_extremal_NS(learner)
        maxIS, minIS = calculate_extremal_IS(learner)
        extrema[0][max_samples] = maxNS
        extrema[1][max_samples] = minNS
        extrema[2][max_samples] = maxIS
        extrema[3][max_samples] = minIS

    # Run uniform learners
    if calculate_uniform:
        if progress_bars=='simple' or progress_bars=='notebook':
            n = learner.total_samples()
            avg_samples_per_point = n/len(learner.data)
            x_uniform = np.linspace(learner.bounds[0],learner.bounds[1],np.ceil(n/avg_samples_per_point))
            learner_uniform = adaptive.AverageLearner1D(learner.function, bounds=learner.bounds, strategy=1)
            for _ in tqdm(np.arange(avg_samples_per_point)):
                for xx in x_uniform:
                    yy = learner_uniform.function(xx)
                    learner_uniform.tell(xx,yy)
        else:
            n = learner.total_samples()
            avg_samples_per_point = n/len(learner.data)
            x_uniform = np.linspace(learner.bounds[0],learner.bounds[1],np.ceil(n/avg_samples_per_point))
            learner_uniform = adaptive.AverageLearner1D(learner.function, bounds=learner.bounds, strategy=1)
            for _ in np.arange(avg_samples_per_point):
                for xx in x_uniform:
                    yy = learner_uniform.function(xx)
                    learner_uniform.tell(xx,yy)
        errors[1][max_samples] = calculate_L1error(learner_uniform)

    if generate_plot:
        # Figure
        fig, axes = plt.subplots(1,3,figsize=(30/2.54,6/2.54))

        # Plot noisy function
        if True:
            x = np.linspace(learner.bounds[0],learner.bounds[1],100)
            y = []
            for xi in x:
                y.append(learner.function(xi))
            axes[0].plot(x,y,alpha=0.3,color='tab:gray',label='Noisy function')

        # Plot learner's data
        if True:
            x, y = zip(*sorted(learner.data.items()))
            axes[0].plot(x, y, linewidth=2)
            _, err = zip(*sorted(learner._error_in_mean.items()))
            axes[0].errorbar(x, y, yerr=err, linewidth=0, marker='o', color='k', markersize=2,
                             elinewidth=1, capsize=3, capthick=1, label='Learner data', alpha=0.5)
            #axes[0].text(-0.8,0.8,'N=%d'%learner.total_samples())

        # Plot errors
        if True:
            axes[1].scatter(list(errors[0].keys()),list(errors[0].values()),color='k',alpha=0.8,marker='v',label='AverageLearner1D')
            if calculate_uniform:
                axes[1].scatter(list(errors[1].keys()),list(errors[1].values()),color='tab:purple',alpha=0.8,marker='^',label='Uniform Learner')
                axes[1].legend()
            # Fitting err=a*N^(-d)
            if fittings:
                from scipy.optimize import curve_fit
                def fit_fun_err(logN,a,d):
                    return a-logN*d

                # To do a reliable fitting, we log-transform the data
                x = np.log10(list(errors[0].keys()))
                y = np.log10(list(errors[0].values()))
                popt, _ = curve_fit(fit_fun_err, x, y)

                Nvec = np.linspace(min(errors[0].keys()),max(errors[0].keys()),50)
                errvec = 10**(fit_fun_err(np.log10(Nvec), *popt))
                print('Error fitting: %.2f*N^(-%.2f)'%(popt[0],popt[1]))
                axes[1].plot(Nvec,errvec, color='k', alpha=0.6)

        # Plot number of data points
        if True:
            ax2 = axes[1].twinx()
            ax2.scatter(list(errors[2].keys()),list(errors[2].values()), marker='D', color='tab:orange')
            ax2.set_ylabel('n', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            ax2.spines['right'].set_color('tab:orange')
            ax2.tick_params(axis='y', which='both', colors='tab:orange')

            # Fitting n=a*N^(1/b)
            if fittings:
                from scipy.optimize import curve_fit
                def fit_fun_n(logN,a,b):
                    return a+logN*(1/b)
                # To do a reliable fitting, we log-transform the data
                x = np.log10(list(errors[2].keys()))
                y = np.log10(list(errors[2].values()))
                popt, _ = curve_fit(fit_fun_n, x, y)

                Nvec = np.linspace(min(errors[2].keys()),max(errors[2].keys()),50)
                nvec = 10**(fit_fun_n(np.log10(Nvec), *popt))
                print('n(N) fitting: %.2f*N^(1/%.2f)'%(popt[0],popt[1]))
                ax2.plot(Nvec,nvec, color='tab:orange', alpha=0.6)

        # Plot extrema NSR and ISR
        if True:
            axes[2].plot(list(extrema[0].keys()),list(extrema[0].values()),color='tab:blue',alpha=0.8,marker='v')
            axes[2].plot(list(extrema[1].keys()),list(extrema[1].values()),color='tab:blue',alpha=0.8,marker='^')
            axes[2].set_ylabel('Extremal NS',color='tab:blue')
            axes[2].tick_params(axis='y', which='both', colors='tab:blue')
            ax3 = axes[2].twinx()
            ax3.plot(list(extrema[2].keys()),list(extrema[2].values()),color='tab:green',alpha=0.8,marker='v')
            ax3.plot(list(extrema[3].keys()),list(extrema[3].values()),color='tab:green',alpha=0.8,marker='^')
            ax3.set_ylabel('Extremal IS', color='tab:green')
            ax3.tick_params(axis='y', which='both', colors='tab:green')
            ax3.spines['right'].set_color('tab:green')
            ax3.spines['left'].set_color('tab:blue')


        # Specs
        if True:
            axes[0].set_xlim(learner.bounds)
            axes[0].set_xlabel("x")
            # axes[0].legend()
            if ylim:
                axes[0].set_ylim(ylim)

            if calculate_uniform:
                errmin = min([min(errors[0].values()),min(errors[1].values())])
                errmax = max([max(errors[0].values()),max(errors[1].values())])
            else:
                errmin = min(errors[0].values())
                errmax = max(errors[0].values())
            axes[1].set_ylim([0.5*errmin,3*errmax])
            #axes[1].set_xlim([1,max(errors[0].keys())*1.1])
            axes[1].ticklabel_format(axis='x',style='sci')
            axes[1].set_xlabel('N')
            axes[1].set_ylabel('L1-error')
            axes[1].set_xscale('log')
            axes[1].set_yscale('log')
            ax2.set_yscale('log')
            ax2.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=10))
            ax2.yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2,0.6)))
            #ax2.ticklabel_format(style='plain', axis='y')
            axes[1].yaxis.set_label_position("left")
            ax2.yaxis.set_label_position("right")
            axes[1].yaxis.tick_left()
            ax2.yaxis.tick_right()

            axes[2].set_ylim([0.5*min(extrema[1].values()),3*max(extrema[0].values())])
            ax3.set_ylim([0.5*min(extrema[3].values()),3*max(extrema[2].values())])
            axes[2].ticklabel_format(axis='x',style='sci')
            ax3.ticklabel_format(axis='x',style='sci')
            axes[2].set_xlabel('N')
            axes[2].set_xscale('log')
            ax3.set_xscale('log')
            axes[2].set_yscale('log')
            ax3.set_yscale('log')
            axes[2].yaxis.set_label_position("left")
            ax3.yaxis.set_label_position("right")
            axes[2].yaxis.tick_left()
            ax3.yaxis.tick_right()

            plt.subplots_adjust(wspace=0.6)
            # plt.subplots_adjust(wspace=0.3,hspace=0.4)

        if save_plot:
            plt.savefig(fig_name+'.pdf',dpi=300,bbox_inches='tight')
        else:
            plt.show()

    if keep_init:
        learner = copy.deepcopy(learner1)

    if return_errors:
        return errors, extrema
    else:
        return


def test_NSR_ISR(max_samples, learners=None, errors=None, errors_uniform=None, sigmas = 0, plot_errors_uniform=True, return_learners=False, generate_plots=True, save_plots=False, fig_name=None, extra_learner_specs=None, progress_bars='notebook', **learner_kwargs):
    '''Generates 3x5 plots with the estimated function, the ISR and the NSR.
       ---Input---
            max_samples: maximum number of samples (int)
            learners: list containing either 5 or 10 learners; optional (list)
            errors: dictionary containing the number of samples as key and error
                    between the real function and the interpolated one as value;
                    optional (dict)
            errors_uniform: dictionary containing the number of samples as key
                            and error between the real function and the
                            interpolated one (from a uniform sampler) as value;
                            optional (dict)
            sigmas: base value for the std of the noise; see each function at the
                    end of this file for further details (float)
            plot_errors_uniform: set to True to compute and plot errors_uniform
                                 (bool)
            return_learners: set to True to return all the learners;
                             set to False to return nothing (bool)
            generate_plots: set to True to generate plots, either to show or
                            to save them (bool)
            save_plots: set to True to save animation as .gif (bool)
            fig_name: name of the figure, only used if save_plots==True (str)
            extra_learner_specs: parameters of a second batch of learners to be
                                 plotted in the same figure; it generates a
                                 second loading bar (dict). Example:
                                 {'strategy':1, 'delta':2, 'min_samples':5}
            progress_bars: set to 'simple' for Python progress bars, set to
                           'notebook' if running on a notebook, set to None for
                           no progress bars'''
    if progress_bars=='simple':
        from tqdm import tqdm
    elif progress_bars=='notebook':
        from tqdm.notebook import tqdm

    if save_plots and (not fig_name):
        raise ValueError('fig_name not specified.')
        assert isinstance(fig_name,str), 'fig_name must be str.'

    # Initialize learners
    if not learners:
        if not learner_kwargs:
            raise TypeError('Parameters of the learners not specified.')
        learners = []
        learners.append(adaptive.AverageLearner1D(partial(const, a=0, sigma=sigmas),
                                                  bounds=(-1,1), **learner_kwargs))
        learners.append(adaptive.AverageLearner1D(partial(const, a=0, sigma=sigmas,
                                                          sigma_end=sigmas*5, bounds=(-1,1)),
                                                  bounds=(-1,1), **learner_kwargs))
        learners.append(adaptive.AverageLearner1D(partial(peak, peak_width=0.01,
                                                          offset=0, sigma=sigmas),
                                                  bounds=(-1,1), **learner_kwargs))
        learners.append(adaptive.AverageLearner1D(partial(tanh, stretching=20,
                                                          offset=0, sigma=sigmas),
                                                  bounds=(-1,1), **learner_kwargs))
        learners.append(adaptive.AverageLearner1D(partial(lorentz, width=0.5,
                                                          offset=0, sigma=sigmas*3),
                                                  bounds=(-1,1), **learner_kwargs))
        if extra_learner_specs:
            learners.append(adaptive.AverageLearner1D(partial(const, a=0, sigma=sigmas),
                                                      bounds=(-1,1), **extra_learner_specs))
            learners.append(adaptive.AverageLearner1D(partial(const, a=0, sigma=sigmas,
                                                              sigma_end=sigmas*5, bounds=(-1,1)),
                                                      bounds=(-1,1), **extra_learner_specs))
            learners.append(adaptive.AverageLearner1D(partial(peak, peak_width=0.01,
                                                              offset=0, sigma=sigmas),
                                                      bounds=(-1,1), **extra_learner_specs))
            learners.append(adaptive.AverageLearner1D(partial(tanh, stretching=20,
                                                              offset=0, sigma=sigmas),
                                                      bounds=(-1,1), **extra_learner_specs))
            learners.append(adaptive.AverageLearner1D(partial(lorentz, width=0.5,
                                                              offset=0, sigma=sigmas*3),
                                                      bounds=(-1,1), **extra_learner_specs))

    # Run learners and calculate NSR, ISR, and error
    if True:
        NSR = []
        ISR = []
        if not errors:
            errors = [{},{},{},{},{},{},{},{},{},{}]

        if progress_bars=='simple' or progress_bars=='notebook':
            for i in tqdm(np.arange(5)):
                while learners[i].total_samples()<max_samples:
                        xs, _ = learners[i].ask(1)
                        for xx in xs:
                            yy = learners[i].function(xx)
                            learners[i].tell(xx,yy)
                NSR.append(calculate_NSR(learners[i]))
                ISR.append(calculate_ISR(learners[i]))
                errors[i][max_samples] = calculate_L1error(learners[i])
        else:
            for i in np.arange(5):
                while learners[i].total_samples()<max_samples:
                        xs, _ = learners[i].ask(1)
                        for xx in xs:
                            yy = learners[i].function(xx)
                            learners[i].tell(xx,yy)
                NSR.append(calculate_NSR(learners[i]))
                ISR.append(calculate_ISR(learners[i]))
                errors[i][max_samples] = calculate_L1error(learners[i])

        if len(learners)==10:
            if progress_bars=='simple' or progress_bars=='notebook':
                for i in tqdm(np.arange(5,10)):
                    while learners[i].total_samples()<max_samples:
                            xs, _ = learners[i].ask(1)
                            for xx in xs:
                                yy = learners[i].function(xx)
                                learners[i].tell(xx,yy)
                    NSR.append(calculate_NSR(learners[i]))
                    ISR.append(calculate_ISR(learners[i]))
                    errors[i][max_samples] = calculate_L1error(learners[i])
            else:
                for i in np.arange(5,10):
                    while learners[i].total_samples()<max_samples:
                            xs, _ = learners[i].ask(1)
                            for xx in xs:
                                yy = learners[i].function(xx)
                                learners[i].tell(xx,yy)
                    NSR.append(calculate_NSR(learners[i]))
                    ISR.append(calculate_ISR(learners[i]))
                    errors[i][max_samples] = calculate_L1error(learners[i])

    # Run uniform learners
    if True:
        if not errors_uniform:
            errors_uniform = [{},{},{},{},{}]
        if plot_errors_uniform:
            learners_uniform = []
            if progress_bars=='simple' or progress_bars=='notebook':
                for i in tqdm(np.arange(5)):
                    n = learners[i].total_samples()
                    avg_samples_per_point = n/len(learners[i].data)
                    x_uniform = np.linspace(learners[i].bounds[0],learners[i].bounds[1],np.ceil(n/avg_samples_per_point))
                    learners_uniform.append(adaptive.AverageLearner1D(learners[i].function, bounds=learners[i].bounds, strategy=1))
                                        # The strategy is not relevant for the uniform learner, so we just set it to 1
                    for _ in np.arange(avg_samples_per_point):
                        for xx in x_uniform:
                            yy = learners_uniform[i].function(xx)
                            learners_uniform[i].tell(xx,yy)
                    errors_uniform[i][max_samples] = calculate_L1error(learners_uniform[i])
            else:
                for i in np.arange(5):
                    n = learners[i].total_samples()
                    avg_samples_per_point = n/len(learners[i].data)
                    x_uniform = np.linspace(learners[i].bounds[0],learners[i].bounds[1],np.ceil(n/avg_samples_per_point))
                    learners_uniform.append(adaptive.AverageLearner1D(learners[i].function, bounds=learners[i].bounds, strategy=1))
                                        # The strategy is not relevant for the uniform learner, so we just set it to 1
                    for _ in np.arange(avg_samples_per_point):
                        for xx in x_uniform:
                            yy = learners_uniform[i].function(xx)
                            learners_uniform[i].tell(xx,yy)
                    errors_uniform[i][max_samples] = calculate_L1error(learners_uniform[i])

    if generate_plots:
        # Figure
        fig, axes = plt.subplots(4,5,figsize=(25/2.54,20/2.54))

        # Plot noisy functions
        if True:
            x = np.linspace(-1,1,100)
            y = []
            for xi in x:
                y.append(const(xi, a=0, sigma=sigmas))
            axes[0][0].plot(x,y,alpha=0.2,color='tab:gray')

            x = np.linspace(-1,1,100)
            y = []
            for xi in x:
                y.append(const(xi, a=0, sigma=sigmas, sigma_end=sigmas*5, bounds=(-1,1)))
            axes[0][1].plot(x,y,alpha=0.2,color='tab:gray')

            x = np.linspace(-1,1,100)
            y = []
            for xi in x:
                y.append(peak(xi, peak_width=0.01, offset=0, sigma=sigmas))
            axes[0][2].plot(x,y,alpha=0.2,color='tab:gray')

            x = np.linspace(-1,1,100)
            y = []
            for xi in x:
                y.append(tanh(xi, stretching=20, offset=0, sigma=sigmas))
            axes[0][3].plot(x,y,alpha=0.2,color='tab:gray')

            x = np.linspace(-1,1,100)
            y = []
            for xi in x:
                y.append(lorentz(xi, width=0.5, offset=0, sigma=sigmas*3))
            axes[0][4].plot(x,y,alpha=0.2,color='tab:gray')

        # Plot learners' data
        if True:
            for i in np.arange(5):
                #for x in learners[i].data.keys():
                #    for y in learners[i]._data_samples[x]:
                #        axes[0][i].scatter(x, y, s=2)
                x, y = zip(*sorted(learners[i].data.items()))
                axes[0][i].plot(x,y,color='tab:blue',alpha=0.8,marker='.',markersize=5)
                axes[1][i].plot(list(NSR[i].keys()),list(NSR[i].values()),color='tab:blue')
                axes[2][i].plot(list(ISR[i].keys()),list(ISR[i].values()),color='tab:blue')
                axes[3][i].scatter(list(errors[i].keys()),list(errors[i].values()),color='tab:blue',alpha=0.8,marker='v')
            if len(learners)==10:
                for i in np.arange(5):
                    x, y = zip(*sorted(learners[i+5].data.items()))
                    axes[0][i].plot(x,y,color='tab:purple',alpha=0.5,marker='.',markersize=5)
                    axes[1][i].plot(list(NSR[i+5].keys()),list(NSR[i+5].values()),color='tab:purple',alpha=0.5)
                    axes[2][i].plot(list(ISR[i+5].keys()),list(ISR[i+5].values()),color='tab:purple',alpha=0.5)
                    axes[3][i].scatter(list(errors[i+5].keys()),list(errors[i+5].values()),color='tab:purple',alpha=0.8,marker='>')
            if plot_errors_uniform:
                for i in np.arange(5):
                    x, y = zip(*sorted(learners_uniform[i].data.items()))
                    axes[0][i].plot(x,y,color='tab:orange',alpha=0.4,marker='.',markersize=5)
                    axes[3][i].scatter(list(errors_uniform[i].keys()),list(errors_uniform[i].values()),color='tab:orange',alpha=0.8,marker='^')

        # Specs
        if True:
            for i in np.arange(5):
                for j in np.arange(3):
                    axes[j][i].set_xlim([-1,1])
                    axes[j][i].set_xlim([-1,1])
                    axes[j][i].set_xlim([-1,1])
                    axes[j][i].tick_params(labelsize=7)
                axes[1][i].set_ylim([-0.1,1.1])
                axes[2][i].set_ylim([-0.1,1.1])
                axes[3][i].set_ylim([0.001,0.1])
                axes[3][i].set_xlim([0,11000])
                axes[0][i].set_xticks([-1,0,1])
                axes[1][i].set_xticks([-1,0,1])
                axes[2][i].set_xticks([-1,0,1])
                axes[1][i].set_yticks([0,0.5,1])
                axes[2][i].set_yticks([0,0.5,1])
                axes[3][i].set_xticks([1000, 10000])
                axes[3][i].ticklabel_format(axis='x',style='sci')
                axes[0][i].set_xlabel("x")
                axes[1][i].set_xlabel("x")
                axes[2][i].set_xlabel("x")
                axes[3][i].set_xlabel("N")
                axes[3][i].set_yscale('log')
                axes[3][i].tick_params(labelsize=7)
                #axes[3][i].set_xscale('log')
            axes[0][0].set_ylim([-0.5,0.5])
            axes[0][1].set_ylim([-1,1])
            axes[0][2].set_ylim([-1.2,1.2])
            axes[0][3].set_ylim([-1.2,1.2])
            axes[0][4].set_ylim([-0.2,2.2])
            axes[3][1].set_ylim([0.01,1])

            axes[0][0].set_ylabel('g(x)')
            axes[1][0].set_ylabel('NSR(x)')
            axes[2][0].set_ylabel('ISR(x)')
            axes[3][0].set_ylabel('Error')

            axes[0][0].set_title('Constant\n+ uniform noise', fontsize=8)
            axes[0][1].set_title('Constant\n+ linear noise', fontsize=8)
            axes[0][2].set_title('Peak\n+ uniform noise', fontsize=8)
            axes[0][3].set_title('Tanh\n+ uniform noise', fontsize=8)
            axes[0][4].set_title('Lorentz\n+ multipl. noise', fontsize=8)

            # for j in np.arange(1,5):
            #     axes[1][j].set_yticklabels([])
            #     axes[2][j].set_yticklabels([])
            #     axes[3][j].set_yticklabels([])
            # for i in np.arange(5):
            #     axes[0][i].set_xticklabels([])
            #     axes[1][i].set_xticklabels([])
            plt.subplots_adjust(wspace=0.3,hspace=0.4)

        if save_plots:
            plt.savefig(fig_name+'.pdf',dpi=300,bbox_inches='tight')
        else:
            plt.show()

    if return_learners:
        return learners, errors, errors_uniform
    else:
        return

def test_NSR_animation(max_samples, return_learners=False, fps=10, samples_per_frame=100, show_anim=True, save_anim=False, **learner_kwargs):
    '''Calculates the NSR versus x for all the test functions.
       ---Input---
            max_samples: maximum number of samples (int)
            return_learners: set to True to return all the learners;
                             set to False to return nothing (bool)
            fps: frames per second (int)
            samples_per_frame: number of samples taken per frame in
                               the animation (int)
            show_anim: set to True to show animation (bool)
            save_anim: set to True to save animation as .gif (bool)'''

    learner1 = adaptive.AverageLearner1D(partial(const, a=0, sigma=0.1),
                                         bounds=(-1,1), **learner_kwargs)
    learner2 = adaptive.AverageLearner1D(partial(const, a=0, sigma=0.1,
                                                 sigma_end=0.5, bounds=(-1,1)),
                                         bounds=(-1,1), **learner_kwargs)
    learner3 = adaptive.AverageLearner1D(partial(peak, peak_width=0.01,
                                                 offset=0, sigma=0.1),
                                         bounds=(-1,1), **learner_kwargs)
    learner4 = adaptive.AverageLearner1D(partial(tanh, stretching=20,
                                                 offset=0, sigma=0.1),
                                         bounds=(-1,1), **learner_kwargs)
    learner5 = adaptive.AverageLearner1D(partial(lorentz, width=0.5,
                                                 offset=0, sigma=0.3),
                                         bounds=(-1,1), **learner_kwargs)

    # To avoid compilation errors, the variables that would be created inside
    # an exec() are created now instead
    xs = [0]
    yy = 0
    NSR1 = {}
    NSR2 = {}
    NSR3 = {}
    NSR4 = {}

    for name in ['1','2','3','4']:
        for _ in range(0,samples_per_frame):
                exec('xs, _ = learner'+name+'.ask(1)')
                for xx in xs:
                    exec('yy = learner'+name+'.function(xx)')
                    exec('learner'+name+'.tell(xx,yy)')
                exec('NSR'+name+' = calculate_NSR(learner'+name+')')

    # Figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

    plot1, = ax1.plot(list(NSR1.keys()),list(NSR1.values()))
    plot2, = ax2.plot(list(NSR2.keys()),list(NSR2.values()))
    plot3, = ax3.plot(list(NSR3.keys()),list(NSR3.values()))
    plot4, = ax4.plot(list(NSR4.keys()),list(NSR4.values()))

    for name in ['1','2','3','4']:
        exec('ax'+name+'.set_xlim([-1,1])')
        exec('ax'+name+'.set_ylim([0,1])')
        exec('ax'+name+'.set_xlabel("x")')
        exec('ax'+name+'.set_ylabel("NSR(x)")')
    ax1.set_title('Constant + uniform noise')
    ax2.set_title('Constant + linear noise')
    ax3.set_title('Peak + uniform noise')
    ax4.set_title('Tanh + uniform noise')

    # Animation
    frames = int(np.ceil(max_samples/samples_per_frame))

    def update(frame_i):
        for _ in range(0,samples_per_frame):
                xs, _ = learner1.ask(1)
                for xx in xs:
                    yy = learner1.function(xx)
                    learner1.tell(xx,yy)
        NSR1 = calculate_NSR(learner1)
        plot1.set_xdata(list(NSR1.keys()))
        plot1.set_ydata(list(NSR1.values()))
        for _ in range(0,samples_per_frame):
                xs, _ = learner2.ask(1)
                for xx in xs:
                    yy = learner2.function(xx)
                    learner2.tell(xx,yy)
        NSR2 = calculate_NSR(learner2)
        plot2.set_xdata(list(NSR2.keys()))
        plot2.set_ydata(list(NSR2.values()))
        for _ in range(0,samples_per_frame):
                xs, _ = learner3.ask(1)
                for xx in xs:
                    yy = learner3.function(xx)
                    learner3.tell(xx,yy)
        NSR3 = calculate_NSR(learner3)
        plot3.set_xdata(list(NSR3.keys()))
        plot3.set_ydata(list(NSR3.values()))
        for _ in range(0,samples_per_frame):
                xs, _ = learner4.ask(1)
                for xx in xs:
                    yy = learner4.function(xx)
                    learner4.tell(xx,yy)
        NSR4 = calculate_NSR(learner4)
        plot4.set_xdata(list(NSR4.keys()))
        plot4.set_ydata(list(NSR4.values()))
        return plot1,plot2,plot3,plot4

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000/fps)
    print('All %d frames created.' % frames)

    if show_anim:
        warnings.warn("Displaying the animation in the notebook may skip the first and last frames of animation. For the full simulation, please open the gif file separately.")
        display(HTML(ani.to_html5_video()))
    if save_anim:
        print('Saving animation...')
        ani.save('test_NSR.gif', writer='imagemagick', fps=None);

    plt.close(ani._fig)
    print('Done!')

    if not return_learners:
        return ani
    else:
        return ani, learner1, learner2, learner3, learner4

#____________________________________________________________________
#______________________CALCULATE MAGNITUDES__________________________
#____________________________________________________________________
def calculate_NSR(learner):
    '''Calculates the number of samples ratio (NSR) of a learner. This is
       calculated as number_of_samples(x)/max(number_of_samples).
       ---Output---
            NSR: dictionary where the keys are the points x and the values
                 are the NSR at those points.'''
    NSR = copy.deepcopy(learner._number_samples)
    maxNS = max(NSR.values())
    NSR.update((x, y/maxNS) for x, y in NSR.items())
    return NSR

def calculate_extremal_NS(learner):
    '''Calculates maximum and minimum number of samples per point (NS) of a
       learner. This is calculated as number_of_samples(x).
       ---Output---
            NS: dictionary where the keys are the points x and the values
                 are the NS at those points.'''
    NS = copy.deepcopy(learner._number_samples)
    maxNS = max(NS.values())
    minNS = min(NS.values())
    return maxNS, minNS

def calculate_ISR(learner):
    '''Calculates the interval size ratio (ISR) of a learner. This is
       calculated as interval_size(x_i)/max(interval_size(x)). We identify each
       interval with its mid-point.
       ---Output---
            ISR: dictionary where the keys are the mid-points of the intervals
                 and the values are the ISR of the intervals.'''
    points = copy.deepcopy(learner.data)
    x, _ = zip(*sorted(points.items()))
    ISR = {}
    for i in np.arange(len(x)-1):
        x_mid = x[i] + (x[i+1]-x[i])/2
        ISR[x_mid] = x[i+1]-x[i]
    maxIS = max(ISR.values())
    ISR.update((x, y/maxIS) for x, y in ISR.items())
    return ISR

def calculate_extremal_IS(learner):
    '''Calculates the max and min values of the interval size (IS) of a
       learner. This is calculated as interval_size(x_i)/max(interval_size(x)).
       We identify each interval with its mid-point.
       ---Output---
            IS: dictionary where the keys are the mid-points of the intervals
                 and the values are the IS of the intervals.'''
    points = copy.deepcopy(learner.data)
    x, _ = zip(*sorted(points.items()))
    IS = {}
    for i in np.arange(len(x)-1):
        x_mid = x[i] + (x[i+1]-x[i])/2
        IS[x_mid] = x[i+1]-x[i]
    maxIS = max(IS.values())
    minIS = min(IS.values())
    return maxIS, minIS

def calculate_L1error(learner):
    '''Calculates the error (L1-norm) between the real function and the
       interpolated one using the samples from the learner.
       ---Output---
            error: float'''
    x, y = zip(*learner.data.items())
    f = partial(learner.function, sigma=0)

    # Create interpolators
    f_interp = interp1d(x, y, kind='linear', fill_value='extrapolate')

    def integrand(x):
        return abs(f_interp(x)-f(x,sigma=0))

    error, error_error = quad(integrand,learner.bounds[0],learner.bounds[1])

    return error

#____________________________________________________________________
#______________________RUN AND PLOT LEARNER__________________________
#____________________________________________________________________
def plot_learner(learner,equalaxes=False,ylim=None,alphafun=0.3,alphaline=1,alphabars=0.3,Nfun=100):
    '''Plot learner'''
    xfun = np.linspace(learner.bounds[0],learner.bounds[1],Nfun)
    yfun = []
    for xi in xfun:
        yfun.append(learner.function(xi))

    x, y = zip(*sorted(learner.data.items()))
    try: # AverageLearner1D
        plt.plot(xfun,yfun,alpha=alphafun,color='tab:orange')

        plt.plot(x, y, color='tab:blue', linewidth=1, alpha=alphaline)
        _, err = zip(*sorted(learner._error_in_mean.items()))
        plt.errorbar(x, y, yerr=err, linewidth=0, marker='o', color='k',
                     markersize=2, elinewidth=1, capsize=3, capthick=1, alpha=alphabars)
        plt.title('N=%d'%learner.total_samples())
    except: # Learner1D
        plt.plot(xfun,yfun,linewidth=5,alpha=alphafun,color='tab:orange')

        plt.plot(x, y, linewidth=1, color='tab:blue', marker='o', markersize=2,
                 markeredgecolor='k', markerfacecolor='k')
        plt.title('N=%d'%len(learner.data))
    plt.xlim(learner.bounds)
    if equalaxes:
        plt.gca().set_aspect('equal', adjustable='box')
    if ylim:
        plt.ylim(ylim)

def run_N(learner,N):
    '''Runs the learner until it has N samples'''
    from tqdm.notebook import tqdm
    try: # AverageLearner1D
        N0 = learner.total_samples()
        if N-N0>0:
            for _ in tqdm(np.arange(N-N0)):
                    xs, _ = learner.ask(1)
                    for x in xs:
                        y = learner.function(x)
                        learner.tell(x, y)
    except: # Learner1D
        N0 = len(learner.data)
        if N-N0>0:
            for _ in tqdm(np.arange(N-N0)):
                    xs, _ = learner.ask(1)
                    for x in xs:
                        y = learner.function(x)
                        learner.tell(x, y)

def run_N_more(learner,N):
    '''Runs the learner to obtain N more samples'''
    from tqdm.notebook import tqdm
    for _ in tqdm(np.arange(N)):
            xs, _ = learner.ask(1)
            for x in xs:
                y = learner.function(x)
                learner.tell(x, y)

def simple_liveplot(learner, goal = lambda l: l.total_samples()==500, N_batch = 100, alphafun=0.3,alphaline=1,alphabars=0.3,N_fun=200):
    import time
    import pylab as pl
    from IPython import display
    xfun = np.linspace(learner.bounds[0],learner.bounds[1],N_fun)
    yfun0 = learner.function(xfun, sigma=0)

    yfun = []
    for xi in xfun:
        yfun.append(learner.function(xi))
    try:
        while not goal(learner):
            for i in np.arange(N_batch):
                xs, _ = learner.ask(1)
                for x in xs:
                    y = learner.function(x)
                    learner.tell(x, y)
            x, y = zip(*sorted(learner.data.items()))
            plt.cla()
            try: # AverageLearner1D
                plt.xlim(learner.bounds[0],learner.bounds[1])
                plt.plot(xfun, yfun0, color='k', linewidth=1)
                plt.plot(x, y, linewidth=2, alpha=alphaline)
                plt.autoscale(False)

                _, err = zip(*sorted(learner._error_in_mean.items()))
                plt.errorbar(x, y, yerr=err, linewidth=0, marker='o', color='k',
                             markersize=2, elinewidth=1, capsize=3, capthick=1, alpha=alphabars)
                plt.title('N=%d, n=%d'%(learner.total_samples(),len(learner.data)))
                plt.plot(xfun, yfun, alpha=alphafun, color='tab:orange')
            except: # Learner1D
                plt.xlim(learner.bounds[0],learner.bounds[1])
                plt.plot(xfun, yfun, alpha=alphafun ,color='tab:orange')
                plt.plot(x, y, linewidth=1, color='tab:blue', marker='o', markersize=2,
                         markeredgecolor='k', markerfacecolor='k')
                plt.title('N=%d'%len(learner.data))
            display.clear_output(wait=True)
            display.display(plt.gcf())
    except KeyboardInterrupt:
        display.clear_output(wait=True)
        display.display(pl.gcf())
        plot_learner(learner)


#____________________________________________________________________
#____________________________FUNCTIONS_______________________________
#____________________________________________________________________
def plot_fun(function,xlim,N=200,title=None,ylim=None,**function_kwargs):
    '''Plots a symbolic function within a specific interval.
       ---Inputs---
            function: function to plot (callable)
            xlim: bounds of the interval in which the function can be
                      evaluated (tuple)
            N: number of points (int)
            title: optional, title for the plot (string)
            ylim: optional, vertical limits of the plot (tuple)'''
    import matplotlib.pyplot as plt
    x = np.linspace(xlim[0],xlim[1],N)
    y = []
    for xi in x:
        y.append(function(xi,**function_kwargs))
    plt.plot(x,y)
    plt.xlim(xlim[0],xlim[1])
    if ylim:
        plt.ylim(ylim[0],ylim[1])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    if title:
        plt.title(title)
    return

def scaling_linear_interp_error(function,bounds,max_samples):
    '''Calculates the scaling of the linear interpolation error when the function
       (without noise) is sampled by adaptive'''
    from tqdm.notebook import tqdm
    from scipy.optimize import curve_fit

    learner = adaptive.Learner1D(function, bounds=bounds)
    err_vs_n = {}

    Nvec = np.round(np.logspace(1.7,np.log10(max_samples),20))
    for N in tqdm(np.arange(max_samples)):
        xs, _ = learner.ask(1) # MAKE IT 10??
        for x in xs:
            y = learner.function(x)
            learner.tell(x, y)
        if N in Nvec:
            err_vs_n[len(learner.data)] = calculate_L1error(learner)

    def fit_fun(x,a,b):
        return a-b*x

    # To do a reliable fitting, we log-transform the data
    x = np.log10(list(err_vs_n.keys()))
    y = np.log10(list(err_vs_n.values()))
    popt, _ = curve_fit(fit_fun, x, y)

    nvec = np.linspace(min(err_vs_n.keys()),max(err_vs_n.keys()),100)
    errvec = 10**(fit_fun(np.log10(nvec), *popt))

    plt.scatter(list(err_vs_n.keys()),list(err_vs_n.values()),color='k')
    plt.plot(nvec,errvec, color='k', alpha=0.6)
    plt.xlabel('n')
    plt.ylabel('L1-error')
    plt.title('Error = %.2f*n^(-%.2f)'%(popt[0],popt[1]))
    plt.yscale('log')
    plt.xscale('log')

    popt[0] = 10**popt[0]

    return popt, nvec, errvec

def const(x,a=0,sigma=0,sigma_end=None,bounds=None,wait=False):
    '''Constant + gaussian noise.
       ---Inputs---
            x: evaluate function at this point (float)
            a: value of the constant (float)
            sigma: std of noise (float)
            sigma_end: if None, the noise is uniform; if float, the std of the
                       noise increases linearly from sigma at bounds[0] to
                       sigma_end at bounds[1] (float)
            bounds: bounds of the interval in which the function can be
                    evaluated. Only required if sigma_end is not None (tuple)
            wait: if True, pretend this is a slow function (bool)'''
    if wait:
        sleep(random())
    if not sigma_end:
        return a + np.random.normal(0,sigma)
    elif not bounds:
        raise ValueError('bounds not specified')
    else:
        c1 = (sigma_end-sigma)/(bounds[1]-bounds[0])
        c2 = sigma-c1*bounds[0]
        return a + np.random.normal(0,c1*x+c2)

def peak(x, peak_width=0.01, offset=0, sigma=0, wait=False):
    '''Peak + uniform gaussian noise.
       ---Inputs---
            x: evaluate function at this point (float)
            peak_width: width of the peak (float)
            offset: offset of the peak (float)
            sigma: std of noise (float)
            wait: if True, pretend this is a slow function (bool)'''
    if wait:
        sleep(random())
    return x + peak_width**2 / (peak_width**2 + (x - offset)**2) + np.random.normal(0,sigma)

def tanh(x, stretching=10, height=1, offset=0, sigma=0, wait=False):
    '''Tanh + uniform gaussian noise.
       ---Inputs---
            x: evaluate function at this point (float)
            stretching: larger values make the transition steeper (float)
            height: the tanh goes from -height to +height (float)
            offset: offset of the peak (float)
            sigma: std of noise (float)
            wait: if True, pretend this is a slow function (bool)'''
    if wait:
        sleep(random())
    return math.tanh((x-offset)*stretching)*height + np.random.normal(0,sigma)

def lorentz(x, width=0.5, offset=0, sigma=0, wait=False):
    '''Lorentzian + multiplicative gaussian noise.
       ---Inputs---
            x: evaluate function at this point (float)
            width: half-width at half-maximum (float)
            offset: offset of the peak (float)
            sigma: std of noise (float)
            wait: if True, pretend this is a slow function (bool)'''
    if wait:
        sleep(random())
    return (1/np.pi)*(0.5*width)/((x-offset)**2+(0.5*width)**2) * np.random.normal(1,sigma)

def lorentz_add(x, width=0.5, offset=0, sigma=0, wait=False):
    '''Lorentzian + additive gaussian noise.
       ---Inputs---
            x: evaluate function at this point (float)
            width: half-width at half-maximum (float)
            offset: offset of the peak (float)
            sigma: std of noise (float)
            wait: if True, pretend this is a slow function (bool)'''
    if wait:
        sleep(random())
    return (1/np.pi)*(0.5*width)/((x-offset)**2+(0.5*width)**2) + np.random.normal(0,sigma)

def heaviside(x, y0=0.5, sigma=0, wait=False):
    '''Heaviside + additive gaussian noise.
       ---Inputs---
            x: evaluate function at this point (float)
            y0: value of the function at x=0 (float)
            sigma: std of noise (float)
            wait: if True, pretend this is a slow function (bool)'''
    if wait:
        sleep(random())
    return np.heaviside(x,y0) - 0.5 + np.random.normal(0,sigma)

def sinusoid(x, A=1, freq=1, offset=0, chirp=False, sigma=0, wait=False):
    '''Sinus + additive gaussian noise.
       ---Inputs---
            x: evaluate function at this point (float)
            A: amplitude (float)
            freq: frequency (float)
            offset: offset of the sinus (float)
            chirp: set to False for sin(x); set to True for sin(x^2) (bool)
            sigma: std of noise (float)
            wait: if True, pretend this is a slow function (bool)'''
    if wait:
        sleep(random())
    if chirp:
        return A*np.sin((x-offset)**2*freq) + np.random.normal(0,sigma)
    else:
        return A*np.sin((x-offset)*freq) + np.random.normal(0,sigma)
