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
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-0.3,0.3])
    ax1.set_xlabel('x')
    ax1.set_ylabel('NSR(x)')
    ax1.set_title('Constant + uniform noise')

    plot2, = ax2.plot(list(NSR2.keys()),list(NSR2.values()))
    ax2.set_xlim([-1,1])
    ax2.set_ylim([-1,1])
    ax2.set_xlabel('x')
    ax2.set_ylabel('NSR(x)')
    ax2.set_title('Constant + linear noise')

    plot3, = ax3.plot(list(NSR3.keys()),list(NSR3.values()))
    ax3.set_xlim([-1,1])
    ax3.set_ylim([-1.2,1.2])
    ax3.set_xlabel('x')
    ax3.set_ylabel('NSR(x)')
    ax3.set_title('Peak + uniform noise')

    plot4, = ax4.plot(list(NSR4.keys()),list(NSR4.values()))
    ax4.set_xlim([-1,1])
    ax3.set_ylim([-1.2,1.2])
    ax4.set_xlabel('x')
    ax4.set_ylabel('NSR(x)')
    ax4.set_title('Tanh + uniform noise')

    # Animation
    frames = int(np.ceil(max_samples/samples_per_frame))

    def update(frame_i):
        for name in ['1','2','3','4']:
            for _ in range(0,samples_per_frame):
                    exec('xs, _ = learner'+name+'.ask(1)')
                    for xx in xs:
                        exec('yy = learner'+name+'.function(xx)')
                        exec('learner'+name+'.tell(xx,yy)')
            exec('NSR'+name+' = calculate_NSR(learner'+name+')')
            exec('plot'+name+'.set_xdata(NSR'+name+'.keys())')
            exec('plot'+name+'.set_ydata(NSR'+name+'.values())')
        #fig.canvas.draw()
        #fig.canvas.flush_events()
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

    return ani


#____________________________________________________________________
#______________________________MISC__________________________________
#____________________________________________________________________
def calculate_NSR(learner):
    '''Calculates the number of samples ratio (NSR) of a learner. This is
       calculated as number_of_samples(x)/max(number_of_samples).
       ---Output---
            NSR: dictionary where the keys are the points x and the values
                 are the NSR at those points.'''
    NSR = learner._number_samples
    maxNS = max(NSR.values())
    NSR.update((x, y/maxNS) for x, y in NSR.items())
    return NSR



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
