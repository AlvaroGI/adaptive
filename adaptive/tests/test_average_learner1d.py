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

def test_NSR_ISR(max_samples, sigmas = 0, return_learners=False, save_plots=False, **learner_kwargs):
    '''Generates 3x5 plots with the estimated function, the ISR and the NSR'''

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

    # To avoid compilation errors, the variables that would be created inside
    # an exec() are created now instead
    xs = [0]
    yy = 0
    NSR = [{},{},{},{},{}]
    ISR = [{},{},{},{},{}]

    for i in np.arange(5):
        while learners[i].total_samples()<max_samples:
                xs, _ = learners[i].ask(1)
                for xx in xs:
                    yy = learners[i].function(xx)
                    learners[i].tell(xx,yy)
        NSR[i] = calculate_NSR(learners[i])
        ISR[i] = calculate_ISR(learners[i])

    # Figure
    fig, axes = plt.subplots(3,5,figsize=(25/2.54,10/2.54))
    for i in np.arange(5):
        #for x in learners[i].data.keys():
        #    for y in learners[i]._data_samples[x]:
        #        axes[0][i].scatter(x, y, s=2)
        x, y = zip(*sorted(learners[i].data.items()))
        axes[0][i].plot(x,y)
        axes[1][i].plot(list(NSR[i].keys()),list(NSR[i].values()))
        axes[2][i].plot(list(ISR[i].keys()),list(ISR[i].values()))

    # Plot noisy functions
    x = np.linspace(-1,1,100)
    y = []
    for xi in x:
        y.append(const(xi, a=0, sigma=sigmas))
    axes[0][0].plot(x,y,alpha=0.3)

    x = np.linspace(-1,1,100)
    y = []
    for xi in x:
        y.append(const(xi, a=0, sigma=sigmas, sigma_end=sigmas*5, bounds=(-1,1)))
    axes[0][1].plot(x,y,alpha=0.3)

    x = np.linspace(-1,1,100)
    y = []
    for xi in x:
        y.append(peak(xi, peak_width=0.01, offset=0, sigma=sigmas))
    axes[0][2].plot(x,y,alpha=0.3)

    x = np.linspace(-1,1,100)
    y = []
    for xi in x:
        y.append(tanh(xi, stretching=20, offset=0, sigma=sigmas))
    axes[0][3].plot(x,y,alpha=0.3)

    x = np.linspace(-1,1,100)
    y = []
    for xi in x:
        y.append(lorentz(xi, width=0.5, offset=0, sigma=sigmas*3))
    axes[0][4].plot(x,y,alpha=0.3)

    # Specs
    for i in np.arange(5):
        for j in np.arange(3):
            axes[j][i].set_xlim([-1,1])
            axes[j][i].set_xlim([-1,1])
            axes[j][i].set_xlim([-1,1])
        axes[1][i].set_ylim([-0.1,1.1])
        axes[2][i].set_ylim([-0.1,1.1])
        axes[2][i].set_xlabel("x")
    axes[0][0].set_ylim([-0.5,0.5])
    axes[0][1].set_ylim([-1,1])
    axes[0][2].set_ylim([-1.2,1.2])
    axes[0][3].set_ylim([-1.2,1.2])
    axes[0][4].set_ylim([-0.2,2.2])

    axes[0][0].set_ylabel('g(x)')
    axes[1][0].set_ylabel('NSR(x)')
    axes[2][0].set_ylabel('ISR(x)')

    axes[0][0].set_title('Constant\n+ uniform noise', fontsize=8)
    axes[0][1].set_title('Constant\n+ linear noise', fontsize=8)
    axes[0][2].set_title('Peak\n+ uniform noise', fontsize=8)
    axes[0][3].set_title('Tanh\n+ uniform noise', fontsize=8)
    axes[0][4].set_title('Lorentz\n+ multipl. noise', fontsize=8)

    for j in np.arange(1,5):
#        axes[0][j].set_yticklabels([])
        axes[1][j].set_yticklabels([])
        axes[2][j].set_yticklabels([])
    for i in np.arange(5):
        axes[0][i].set_xticklabels([])
        axes[1][i].set_xticklabels([])
    plt.subplots_adjust(wspace=0.3)

    return



def test_NSR(max_samples, return_learners=False, fps=10, samples_per_frame=100, show_anim=True, save_anim=False, **learner_kwargs):
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
#______________________________MISC__________________________________
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

def tanh(x, stretching=10, offset=0, sigma=0, wait=False):
    '''Tanh + uniform gaussian noise.
       ---Inputs---
            x: evaluate function at this point (float)
            stretching: larger values make the transition steeper (float)
            offset: offset of the peak (float)
            sigma: std of noise (float)
            wait: if True, pretend this is a slow function (bool)'''
    if wait:
        sleep(random())
    return math.tanh((x-offset)*stretching) + np.random.normal(0,sigma)

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
