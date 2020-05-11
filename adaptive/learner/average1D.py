# -*- coding: utf-8 -*-

import sys
import itertools
import math
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import sortedcollections
import sortedcontainers
from scipy.stats import t as tstud
import random

from adaptive.learner.average_mixin import DataPoint, add_average_mixin
from adaptive.learner.learner1D import Learner1D, _get_neighbors_from_list, loss_manager, _get_intervals
from adaptive.notebook_integration import ensure_holoviews


class AverageLearner1D(Learner1D):
    """Learns and predicts a noisy function 'f:ℝ → ℝ^N'.

    New parameters (wrt Learner1D)
    ------------------------------
    min_samples : int
        The minimum number of samples at each point x.
    strategy : int (1-3)
        Strategy chosen to sample new points or re-sample.
        1 = Check the error in the loss.
        2 = Check the error in the mean values.
        3 = Take 'moving average' (this strategy never re-samples).
    delta : float
        The minimum value of the mean error.
    alfa : float
        The size of the interval of confidence of the estimate of the mean
        is 1-2*alfa
    M : int
        Number of points taken for the 'moving average'. Only used in strategy 3.
    """
    def __init__(self, function, bounds, loss_per_interval=None, min_samples=3, strategy=None, delta=0.1, alfa=0.025, M=2, Rn=1.5):
        super().__init__(function, bounds, loss_per_interval)

        self._data_samples = sortedcontainers.SortedDict() # This SortedDict contains all samples f(x) for each x
                                                           # in the form {x0:[f_0(x0), f_1(x0), ...]}
        self._number_samples = sortedcontainers.SortedDict() # This SortedDict contains the number of samples
                                                             # for each x in the form {x0: n0, x1: n1, etc.}
        self._undersampled_points = set() # This set contains the points x that
                                          # are undersampled
        self.min_samples = min_samples

        # Relative error between the real losses and the losses calculated
        # using extreme values of the intervals of confidence
        self._losses_diff = loss_manager(self._scale[0]) # {x: loss_diff(x)}

        self._error_in_mean = error_in_mean_initializer() # This SortedDict contains the estimation errors for
                                                          # each fbar(x) in the form {x0: estimation_error(x0)}

        if not strategy:
            raise ValueError('Strategy not specified.')
        elif strategy>10:
            raise ValueError('Incorrect strategy (should be 1, 2, 3, 4, 5, 6, 7, 9, 10)')
        else:
            self.strategy = strategy

        if self.strategy == 3:
            self._data = data_initializer_ordered()
        elif self.strategy==4:
            self.Rn = Rn
        elif self.strategy==5:
#            self._rescaling_factors = {} # {x0: r0}
            self._rescaled_error_in_mean = error_in_mean_initializer() # {xi: Delta_gi/min(Delta_g_{i},Delta_g_{i-1})}
            self._interval_sizes = error_in_mean_initializer() # {xi: xii-xi}
#            self._relative_interval_sizes = error_in_mean_initializer() # {xi: Delta_xi/Delta_gi}
        elif self.strategy==6:
            self._Rescaling_factor = 1
            self._interval_sizes = error_in_mean_initializer() # {xi: xii-xi}
        elif self.strategy==7:
            self._interval_sizes = error_in_mean_initializer() # {xi: xii-xi}
        elif self.strategy==8:
            self._error_in_mean_capped = error_in_mean_initializer()
            self._oversampled_points = error_in_mean_initializer()
            self.Rn = Rn
        elif self.strategy==9:
            self._interval_sizes = error_in_mean_initializer() # {xi: xii-xi}
        elif self.strategy==10:
            self._Rescaling_factor = 1
            self._interval_sizes = error_in_mean_initializer() # {xi: xii-xi}


        self.delta = delta

        self.alfa = alfa

        if not M:
            import warnings
            warnings.warn("M is zero")

        self.M = M

        random.seed(2)

        #print('__init__ ok!')


    def ask(self, n, tell_pending=True):
        """Return 'n' points that are expected to maximally reduce the loss."""
        assert isinstance(self._error_in_mean, dict)
        assert isinstance(self._number_samples, dict)

        if self.strategy==3:
            points, loss_improvements = self._ask_points_without_adding(n)
        # If self.data contains no points, proceed as in Learner1D
        if not self.data.__len__():
            points, loss_improvements = self._ask_points_without_adding(n)
        # If some point is under-sampled, invest the next samples on it
        elif len(self._undersampled_points):
            x = self._undersampled_points.pop()
            self._undersampled_points.add(x)
            points, loss_improvements = self._ask_for_more_samples(x,n)
        # If only 1 point was sampled, sample a new one
        elif self.data.__len__() == 1:
            points, loss_improvements = self._ask_points_without_adding(n)
        # If there are at least 2 points and the loss difference is too large,
        # sample n/(2*(nth_neighbors+1) times each point of the interval # REVIEW
        elif ((self.strategy==1) # or self.strategy==4)
          and self._losses_diff.peekitem(0)[1] > self.delta):#*max(1,self._scale[1])):
            x1, x2 = self._losses_diff.peekitem(0)[0]
            # We invest half of the samples on each point. In case n is odd,
            # in order to not introduce any asymmetric bias, we choose at random
            # which x will be sampled more times
            if random.randint(0,1):
                    points1, loss_improvements1 = self._ask_for_more_samples(x1,n//2)
                    points2, loss_improvements2 = self._ask_for_more_samples(x2,n-n//2)
            else:
                    points1, loss_improvements1 = self._ask_for_more_samples(x1,n-n//2)
                    points2, loss_improvements2 = self._ask_for_more_samples(x2,n//2)
            points = points1+points2
            loss_improvements = loss_improvements1+loss_improvements2
        # If the error on the estimate of the mean is too large at x,
        # ask for more samples at x
        elif (self.strategy==2 and self._error_in_mean.peekitem(0)[1] > self.delta):
            x = self._error_in_mean.peekitem(0)[0]
            points, loss_improvements = self._ask_for_more_samples(x,n)
        elif self.strategy==4:
            ## Strat 4 based on 1 - UNEFFICIENT METHOD ##
            try:
                i = 0
                xi, xii = self._losses_diff.peekitem(i)[0]
                oversampled = ((self._number_samples[xi] > self.Rn*self.total_samples()/len(self.data))
                                or (self._number_samples[xii] > self.Rn*self.total_samples()/len(self.data)))
                # If any of the extremes of the interval is oversampled, we do not re-sample any of them
                while oversampled:
                    i += 1
                    xi, xii = self._losses_diff.peekitem(i)[0]
                    oversampled = ((self._number_samples[xi] > self.Rn*self.total_samples()/len(self.data))
                                    or (self._number_samples[xii] > self.Rn*self.total_samples()/len(self.data)))
                if self._losses_diff.peekitem(i)[1] > self.delta:
                    x1, x2 = self._losses_diff.peekitem(i)[0]
                    if random.randint(0,1):
                            points1, loss_improvements1 = self._ask_for_more_samples(x1,n//2)
                            points2, loss_improvements2 = self._ask_for_more_samples(x2,n-n//2)
                    else:
                            points1, loss_improvements1 = self._ask_for_more_samples(x1,n-n//2)
                            points2, loss_improvements2 = self._ask_for_more_samples(x2,n//2)
                    points = points1+points2
                    loss_improvements = loss_improvements1+loss_improvements2
                else:
                    points, loss_improvements = self._ask_points_without_adding(n)
            except:
                points, loss_improvements = self._ask_points_without_adding(n)
            ## Strat 4 based on 2 ##
            # try:
            #     if self._error_in_mean_capped.peekitem(0)[1] > self.delta:
            #         x = self._error_in_mean_capped.peekitem(0)[0]
            #         points, loss_improvements = self._ask_for_more_samples(x,n)
            #     else:
            #         points, loss_improvements = self._ask_points_without_adding(n)
            # except:
            #     points, loss_improvements = self._ask_points_without_adding(n)
        # Otherwise, sample new points
        elif self.strategy==5:
            if self._rescaled_error_in_mean.peekitem(0)[1] > self.delta:
                x = self._rescaled_error_in_mean.peekitem(0)[0]
                points, loss_improvements = self._ask_for_more_samples(x,n)
            else:
                points, loss_improvements = self._ask_points_without_adding(n)
        elif (self.strategy==6 and self._error_in_mean.peekitem(0)[1]*self._Rescaling_factor > self.delta):
            x = self._error_in_mean.peekitem(0)[0]
            points, loss_improvements = self._ask_for_more_samples(x,n)
        elif self.strategy==7:
            points, loss_improvements = self._ask_points_without_adding(n)
            x = points[0]
            xi, xii = self._find_neighbors(x, self.neighbors)
            Delta_gi = self._error_in_mean[xi]
            Delta_gii = self._error_in_mean[xii]
            if random.randint(0,1):
                try:
                    if self._interval_sizes[xi] < Delta_gi:
                        points, loss_improvements = self._ask_for_more_samples(xi,n)
                    elif self._interval_sizes[xii] < Delta_gii:
                        points, loss_improvements = self._ask_for_more_samples(xii,n)
                except: # This happens when _interval_sizes[xii] does not exist
                    if self._interval_sizes[xi] < Delta_gi:
                        points, loss_improvements = self._ask_for_more_samples(xi,n)
            else:
                try:
                    if self._interval_sizes[xii] < Delta_gii:
                        points, loss_improvements = self._ask_for_more_samples(xii,n)
                    elif self._interval_sizes[xi] < Delta_gi:
                        points, loss_improvements = self._ask_for_more_samples(xi,n)
                except: # This happens when _interval_sizes[xii] does not exist
                    if self._interval_sizes[xi] < Delta_gi:
                        points, loss_improvements = self._ask_for_more_samples(xi,n)
        elif self.strategy==9:
            if self._error_in_mean.peekitem(0)[1] > self._interval_sizes.peekitem(-1)[1]:
                x = self._error_in_mean.peekitem(0)[0]
                points, loss_improvements = self._ask_for_more_samples(x,n)
            else:
                points, loss_improvements = self._ask_points_without_adding(n)
        elif self.strategy==10:
            Err_j = self._calculate_Err_newsample()
            Err_i = self._calculate_Err_resample()
            #print(Err_i,Err_j,'Resampling:',Err_i>Err_j)
            if Err_i > Err_j:
                x = self._error_in_mean.peekitem(0)[0]
                points, loss_improvements = self._ask_for_more_samples(x,n)
            else:
                points, loss_improvements = self._ask_points_without_adding(n)
        else:
            points, loss_improvements = self._ask_points_without_adding(n)

        if tell_pending:
            #print('tell_pending must be redesigned carefully')
            for p in points:
                self.tell_pending(p)

        return points, loss_improvements

    def _calculate_Err_newsample(self):
        '''Calculates the estimated increase in the error when
           a new point is sampled inside the largest loss interval'''
        # Linear error
        xj, xjj = self.losses.peekitem(0)[0]
        yj = self.data[xj]
        yjj = self.data[xjj]
        dj = ((xj-xjj)**2+(yj-yjj)**2)**0.5
        if True: # Estimate h_j
            xj_ = self.neighbors[xj][0]
            xj2 = self.neighbors[xjj][1]
            if (xj_ is None) and (xj2 is None):
                hj = np.abs(xjj-xj)/np.abs(yjj-yj)
            elif xj_ is None:
                yj2 = self.data[xj2]
                aa = (yj2-yj)/(xj2-xj)
                bb = yj - xj*aa
                hh = np.abs(aa*xjj - yjj + bb)/(aa**2+1)**0.5
                hj = hh
            elif xj2 is None:
                yj_ = self.data[xj_]
                a_ = (yjj-yj_)/(xjj-xj_)
                b_ = yj_ - xj_*a_
                h_ = np.abs(a_*xj - yj + b_)/(a_**2+1)**0.5
                hj = h_
            else:
                yj_ = self.data[xj_]
                yj2 = self.data[xj2]
                a_ = (yjj-yj_)/(xjj-xj_)
                b_ = yj_ - xj_*a_
                h_ = np.abs(a_*xj - yj + b_)/(a_**2+1)**0.5
                aa = (yj2-yj)/(xj2-xj)
                bb = yj - xj*aa
                hh = np.abs(aa*xjj - yjj + bb)/(aa**2+1)**0.5
                hj = (h_+hh)*0.5
        Err_lin = 0.5 * dj * hj

        # Noise error
        t_student = tstud.ppf(1.0 - self.alfa, df=self.min_samples-1)
        nj = self._number_samples[xj]
        var_j = sum( [(y-yj)**2 for y in self._data_samples[xj]] )/(nj-1)
        Delta_gj = self._error_in_mean[xj]
        Delta_gjj = self._error_in_mean[xjj]
        Err_Delta = 0.5 * np.abs(xjj-xj) * (t_student * (var_j / self.min_samples)**0.5 - (Delta_gj+Delta_gjj)/2)
        #return dj
        return Err_lin-Err_Delta

    def _calculate_Err_resample(self):
        '''Calculates the estimated increase in the error when
           the largest uncertainty point is re-sampled'''
        # Noise error
        xi = self._error_in_mean.peekitem(0)[0]
        ni = self._number_samples[xi]
        Delta_gi = self._error_in_mean[xi]
        xi_ = self.neighbors[xi][0]
        xii = self.neighbors[xi][1]
        if xi_ is None:
            Delta_x = np.abs(xii-xi)
        elif xii is None:
            Delta_x = np.abs(xi-xi_)
        else:
            Delta_x = np.abs(xii-xi_)
        Err_Delta = 0.5 * Delta_gi * (1-(ni/(ni+1))**0.5) * Delta_x
        #return Delta_gi
        return Err_Delta

    def _ask_for_more_samples(self,x,n):
        points = [x] * n
        loss_improvements = [0] * n
        #print('loss_improvements not implemented yet')
        return points, loss_improvements

    def tell_pending(self, x):
        # Even if x is in self._data, we can sample it again
        # if x in self._data:
        #     # The point is already evaluated before
        #     return
        self.pending_points.add(x) # Note that a set cannot contain duplicates
        if x not in self._data:
            self._update_neighbors(x, self.neighbors_combined)
            self._update_losses(x, real=False)

    def tell(self, x, y):
        if y is None:
            raise TypeError(
                "Y-value may not be None, use learner.tell_pending(x)"
                "to indicate that this value is currently being calculated"
            )

        # either it is a float/int, if not, try casting to a np.array
        if not isinstance(y, (float, int)):
            y = np.asarray(y, dtype=float)

        '''The next if/else can be alternatively included in a different
           definition of _update_data'''
        if self.strategy == 3:
            self._data_samples[x] = y
            self._update_data_moving_avg(x,y)
            self.pending_points.discard(x)
            super()._update_data_structures(x, y)
            self._number_samples[x] = 1
        # If new data point, operate as in Learner1D
        elif not self._data.__contains__(x):
            self._data_samples[x] = [y]
            self._data[x] = y
            self.pending_points.discard(x)
            super()._update_data_structures(x, y)
            self._number_samples[x] = 1
            self._undersampled_points.add(x)
            self._error_in_mean[x] = np.inf # REVIEW: should be np.inf?
            if self.strategy==1 or self.strategy==4:
                self._update_losses_diff(x)
            elif self.strategy==5:
                #self._rescaling_factors[x] = 1 # REVIEW: should be np.inf?
                self._rescaled_error_in_mean[x] = np.inf # REVIEW: should be np.inf?
                self._update_interval_sizes(x)
                self._update_rescaled_error_in_mean(x,'new')
                #self._update_relative_interval_sizes(x)
            elif self.strategy==6:
                self._update_interval_sizes(x)
            elif self.strategy==7:
                self._update_interval_sizes(x)
            elif self.strategy==8:
                self._error_in_mean_capped[x] = np.inf # REVIEW: should be np.inf?
            elif self.strategy==9:
                self._update_interval_sizes(x)
            elif self.strategy==10:
                self._update_interval_sizes(x)
                pass
        # If re-sampled data point:
        else:
            self._update_data(x,y)
            self.pending_points.discard(x)
            self._update_data_structures_resampling(x, y)
            if self.strategy==1 or self.strategy==4:
                self._update_losses_diff(x)
            elif self.strategy==8:
                self._stop_oversampling(x)

        #if (self.strategy==5 and len(self._relative_interval_sizes)):
        #    self._update_rescaling_factors()
        if (self.strategy==6 and len(self._interval_sizes)):
            self._update_Rescaling_factor()
        if self.strategy==8:
            self._reincorporate_oversampled()

    def _update_rescaled_error_in_mean(self,x,point_type):
        '''Updates self._rescaled_error_in_mean; point_type must be "new" or
           "resampled". '''
        assert point_type=='new' or point_type=='resampled', 'point_type must be "new" or "resampled"'

        xleft, xright = self.neighbors[x]
        if xleft is None and xright is None:
            return
        if xleft is None:
            dleft = self._interval_sizes[x]
        else:
            dleft = self._interval_sizes[xleft]
            xll = self.neighbors[xleft][0]
            if xll is None:
                self._rescaled_error_in_mean[xleft] = self._error_in_mean[xleft] / self._interval_sizes[xleft]
            else:
                self._rescaled_error_in_mean[xleft] = self._error_in_mean[xleft] / min(self._interval_sizes[xll],
                                                                                       self._interval_sizes[xleft])
        if xright is None:
            dright = self._interval_sizes[xleft]
        else:
            dright = self._interval_sizes[x]
            xrr = self.neighbors[xright][1]
            if xrr is None:
                self._rescaled_error_in_mean[xright] = self._error_in_mean[xright] / self._interval_sizes[x]
            else:
                self._rescaled_error_in_mean[xright] = self._error_in_mean[xright] / min(self._interval_sizes[x],
                                                                                         self._interval_sizes[xright])

        if point_type=='resampled':
            self._rescaled_error_in_mean[x] = self._error_in_mean[x] / min(dleft,dright)
        return

    def _stop_oversampling(self,x):
        if self._number_samples[x] > self.Rn*self.total_samples()/len(self.data):
            try:
                self._error_in_mean_capped.pop(x)
                self._oversampled_points[x] = self._number_samples[x]
            except:
                return

    def _reincorporate_oversampled(self):
        try:
            if self._oversampled_points.peekitem(-1)[1] < self.Rn*self.total_samples()/len(self.data):
                x_overs = self._oversampled_points.peekitem(-1)[0]
                self._error_in_mean_capped[x_overs] = self._error_in_mean[x_overs]
                self._oversampled_points.pop(x_overs)
        except:
            return

    def _update_Rescaling_factor(self):
        x_i, minimum_interval_size = self._interval_sizes.peekitem(-1)
        x_ii = self.neighbors[x_i][1]
        if (minimum_interval_size < self._error_in_mean[x_i]
            and self._error_in_mean[x_i]*self._Rescaling_factor < self.delta):
            #and self._number_samples[x_i] > self.min_samples):
            # The second condition is used to prevent decreasing the rescaling
            # factor when there are too few samples and the error is too large
            self._Rescaling_factor = self.delta/minimum_interval_size
            self._interval_sizes.pop(x_i)
        if (minimum_interval_size < self._error_in_mean[x_ii]
            and self._error_in_mean[x_ii]*self._Rescaling_factor < self.delta):
            #and self._number_samples[x_ii] > self.min_samples):
            self._Rescaling_factor = self.delta/minimum_interval_size
            try:
                self._interval_sizes.pop(x_i)
            except: # x_i may have been popped in the previous if
                pass

    def _update_rescaling_factors(self):
        x_i, minimum_interval_size = self._relative_interval_sizes.peekitem(-1)
        x_ii = self.neighbors[x_i][1]
        if (minimum_interval_size < self._error_in_mean[x_i] and self._rescaled_error_in_mean[x_i] < self.delta):
            # The second condition is used to prevent decreasing the rescaling
            # factor when there are too few samples and the error is too large
            old_rescaling_factor = self._rescaling_factors[x_i]
            self._rescaling_factors[x_i] = self.delta/minimum_interval_size #+= self.Delta_r
            self._rescaled_error_in_mean[x_i] = self._rescaled_error_in_mean[x_i] * self._rescaling_factors[x_i] / old_rescaling_factor
            self._relative_interval_sizes.pop(x_i)
        if (minimum_interval_size < self._error_in_mean[x_ii] and self._rescaled_error_in_mean[x_ii] < self.delta):
            old_rescaling_factor = self._rescaling_factors[x_ii]
            self._rescaling_factors[x_ii] = self.delta/minimum_interval_size #+= self.Delta_r
            self._rescaled_error_in_mean[x_ii] = self._rescaled_error_in_mean[x_ii] * self._rescaling_factors[x_ii] / old_rescaling_factor
            try:
                self._relative_interval_sizes.pop(x_i)
            except: # x_i may have been popped in the previous if
                pass

    def _update_interval_sizes(self,x):
        neighbors = self.neighbors[x]
        if neighbors[0] is not None:
        #    self._interval_sizes[neighbors[0]] = x-neighbors[0]
            self._interval_sizes[neighbors[0]] = ((x-neighbors[0])**2 + (self.data[x]-self.data[neighbors[0]])**2)**0.5
        if neighbors[1] is not None:
            self._interval_sizes[x] = ((neighbors[1]-x)**2 + (self.data[neighbors[1]]-self.data[x])**2)**0.5
        return

    def _update_relative_interval_sizes(self,x):
        neighbors = self.neighbors[x]
        if neighbors[0] is not None:
            err = max(self._error_in_mean[neighbors[0]],self._error_in_mean[x])
            if err is not np.inf:
                self._relative_interval_sizes[neighbors[0]] = self._interval_sizes[neighbors[0]] / err
            else:
                self._relative_interval_sizes[neighbors[0]] = self._interval_sizes[neighbors[0]]
        if neighbors[1] is not None:
            err = max(self._error_in_mean[neighbors[1]],self._error_in_mean[x])
            if err is not np.inf:
                self._relative_interval_sizes[x] = self._interval_sizes[x] / err
            else:
                self._relative_interval_sizes[x] = self._interval_sizes[x]
        return

    def _update_data_moving_avg(self,x,y):
        '''This function is only used in strategy 3'''
        x_index = self._data_samples.index(x)
        N = self.total_samples()

        if N == 1:
            self._data[x] = y
            return

        for ii in np.arange(max(0,x_index-self.M),min(N,x_index+self.M+1)):
            new_average = 0
            counter = 0
            for jj in np.arange(max(0,ii-self.M),min(N,ii+self.M+1)):
                new_average += self._data_samples.peekitem(jj)[1]
                counter += 1
            self._data[self._data_samples.peekitem(ii)[0]] = new_average/counter

    def _update_data(self,x,y):
        '''This function is only used if self._data contains x'''
        n = len(self._data_samples[x])
        new_average = self._data[x]*n/(n+1) + y/(n+1)
        self._data[x] = new_average

    def _update_data_structures_resampling(self, x, y):
        '''This function is only used if self._data already contains x'''
        # No need to check that x is inside the bounds
        # No need to update neighbors

        # We have to update _data_samples, _number_samples,
        # _undersampled_points
        self._data_samples[x].append(y)

        self._number_samples[x] = self._number_samples[x]+1
        n = self._number_samples[x]
        if (x in self._undersampled_points) and (n >= self.min_samples):
            #print(n)
            self._undersampled_points.discard(x)

        # We compute the error in the estimation of the mean as
        # the std of the mean multiplied by a t-Student factor to ensure that
        # the mean value lies within the correct interval of confidence
        y_avg = self._data[x]
        variance_in_mean = sum( [(yj-y_avg)**2 for yj in self._data_samples[x]] )/(n-1)
        t_student = tstud.ppf(1.0 - self.alfa, df=n-1)
        self._error_in_mean[x] = t_student*(variance_in_mean/n)**0.5
        if self.strategy==8:
            self._error_in_mean_capped[x] = t_student*(variance_in_mean/n)**0.5
        elif self.strategy==5:
            self._update_interval_sizes(x)
            self._update_rescaled_error_in_mean(x,'resampled')
            #self._update_relative_interval_sizes(x)
        elif self.strategy==6:
            self._update_interval_sizes(x)
        elif self.strategy==7:
            self._update_interval_sizes(x)
        elif self.strategy==9:
            self._update_interval_sizes(x)
        elif self.strategy==10:
            self._update_interval_sizes(x)

        # We also need to update scale and losses
        super()._update_scale(x, y)
        self._update_losses_resampling(x, real=True) # REVIEW

        '''Is the following necessary?'''
        # If the scale has increased enough, recompute all losses.
        if self._scale[1] > self._recompute_losses_factor * self._oldscale[1]:
            for interval in reversed(self.losses):
                self._update_interpolated_loss_in_interval(*interval)

            self._oldscale = deepcopy(self._scale)

    def _update_losses_resampling(self, x, real=True):
        """Update all losses that depend on x, whenever the new point is a
           re-sampled point"""
        # (x_left, x_right) are the "real" neighbors of 'x'.
        x_left, x_right = self._find_neighbors(x, self.neighbors)
        # (a, b) are the neighbors of the combined interpolated
        # and "real" intervals.
        a, b = self._find_neighbors(x, self.neighbors_combined)

        if real:
            for ival in _get_intervals(x, self.neighbors, self.nth_neighbors):
                self._update_interpolated_loss_in_interval(*ival)
        elif x_left is not None and x_right is not None:
            # 'x' happens to be in between two real points,
            # so we can interpolate the losses.
            dx = x_right - x_left
            loss = self.losses[x_left, x_right]
            self.losses_combined[a, x] = (x - a) * loss / dx
            self.losses_combined[x, b] = (b - x) * loss / dx

        # (no real point left of x) or (no real point right of a)
        left_loss_is_unknown = (x_left is None) or (not real and x_right is None)
        if (a is not None) and left_loss_is_unknown:
            self.losses_combined[a, x] = float("inf")

        # (no real point right of x) or (no real point left of b)
        right_loss_is_unknown = (x_right is None) or (not real and x_left is None)
        if (b is not None) and right_loss_is_unknown:
            self.losses_combined[x, b] = float("inf")

    def _update_losses_diff(self, x, real=True):
        """Update all _losses_diff that depend on x.
           The current implementation only updates real points."""
        # When we add a new point x, we should update the losses
        # (x_left, x_right) are the "real" neighbors of 'x'.
        x_left, x_right = self._find_neighbors(x, self.neighbors)

        if real:
            # We need to update all interpolated losses in the interval
            # (x_left, x), (x, x_right) and the nth_neighbors nearest
            # neighboring intervals. Since the addition of the
            # point 'x' could change their loss.
            for ival in _get_intervals(x, self.neighbors, self.nth_neighbors):
                self._update_loss_diff_in_interval(*ival)

            # Since 'x' is in between (x_left, x_right),
            # we get rid of the interval.
            self._losses_diff.pop((x_left, x_right), None)
        else:
            print('_update_losses_diff with real=False not implemented...')

    def _update_loss_diff_in_interval(self, x_left, x_right):
        if x_left is None or x_right is None:
            return
        loss_diff = self._get_loss_diff_in_interval(x_left, x_right)
        self._losses_diff[x_left, x_right] = loss_diff

    def _get_loss_diff_in_interval(self, x_left, x_right):
        assert x_left is not None and x_right is not None

        if x_right - x_left < self._dx_eps:
            return 0

        nn = self.nth_neighbors
        i = self.neighbors.index(x_left)
        start = i - nn
        end = i + nn + 2

        xs = [self._get_point_by_index(i) for i in range(start, end)]
        # Note that we apply a random sign!!
        ys = [self.data.get(x, None)+random_sign()*self._error_in_mean.get(x, None) for x in xs]

        xs_scaled = tuple(self._scale_x(x) for x in xs)
        ys_scaled = tuple(self._scale_y(y) for y in ys)

        loss_IC = self.loss_per_interval(xs_scaled, ys_scaled) # This loss is calculated with
                                                               # extremes values of ys inside
                                                               # its confidence interval

        loss = self.losses[(xs[0],xs[1])] # Recall: the keys of this dict are tuples, not lists

        return abs(loss-loss_IC)/loss



    def tell_many(self,x,y):
        #print('Not implemented yet.')
        return

    def total_samples(self):
        '''Returns the total number of samples'''
        if self.strategy == 3:
            return len(self._data_samples)
        elif not len(self._data):
            return 0
        else:
            _, ns = zip(*self._number_samples.items())
            return sum(ns)

def random_sign():
    return 1 if random.random() < 0.5 else -1

def error_in_mean_initializer():
    '''This initialization orders the dictionary from large to small value'''
    def sorting_rule(key, value):
        return -value
    return sortedcollections.ItemSortedDict(sorting_rule, sortedcontainers.SortedDict())

def data_initializer_ordered():
    '''This initialization is used with self._data in strategy 3. It orders the
       data from small to large x'''
    def sorting_rule(key, value):
        return key
    return sortedcollections.ItemSortedDict(sorting_rule, sortedcontainers.SortedDict())

    # def _ask_points_without_adding(self, n):
    #     points, loss_improvements = super()._ask_points_without_adding(n)
    #     points = [(p, 0) for p in points]
    #     return points, loss_improvements

    # def _get_neighbor_mapping_new_points(self, points):
    #     return {
    #         p: [n for n in self._find_neighbors(p, self.neighbors) if n is not None]
    #         for p in points
    #     }

    # def _get_neighbor_mapping_existing_points(self):
    #     return {k: [x for x in v if x is not None] for k, v in self.neighbors.items()}

    # def tell(self, x_seed, y):
    #     x, seed = x_seed
    #
    #     # either it is a float/int, if not, try casting to a np.array
    #     if not isinstance(y, (float, int)):
    #         y = np.asarray(y, dtype=float)
    #
    #     self._add_to_data(x_seed, y)
    #     self._remove_from_to_pending(x_seed)
    #     self._update_data_structures(x, y)

    # def tell_pending(self, x_seed):
    #     x, seed = x_seed
    #
    #     self._add_to_pending(x_seed)
    #
    #     if x not in self.neighbors_combined:
    #         # If 'x' already exists then there is not need to update.
    #         self._update_neighbors(x, self.neighbors_combined)
    #         self._update_losses(x, real=False)

    # def tell_many(self, xs, ys, *, force=False):
    #     if not force and not (len(xs) > 0.5 * len(self._data) and len(xs) > 2):
    #         # Only run this more efficient method if there are
    #         # at least 2 points and the amount of points added are
    #         # at least half of the number of points already in 'data'.
    #         # These "magic numbers" are somewhat arbitrary.
    #         for x, dp in zip(xs, ys):
    #             for seed, y in dp.items():
    #                 self.tell((x, seed), y)
    #         return
    #
    #     # Add data points
    #     self._data.update(zip(xs, ys))
    #     for x, dp in zip(xs, ys):
    #         if x in self.pending_points:
    #             seeds = dp.keys()
    #             self.pending_points[x].difference_update(seeds)
    #             if len(self.pending_points[x]) == 0:
    #                 # Remove if pending_points[x] is an empty set.
    #                 del self.pending_points[x]
    #
    #     # Below is the same as 'Learner1D.tell_many'.
    #
    #     # Get all data as numpy arrays
    #     points = np.array(list(self._data.keys()))
    #     values = np.array(list(self.data.values()))
    #     points_pending = np.array(list(self.pending_points))
    #     points_combined = np.hstack([points_pending, points])
    #
    #     # Generate neighbors
    #     self.neighbors = _get_neighbors_from_list(points)
    #     self.neighbors_combined = _get_neighbors_from_list(points_combined)
    #
    #     # Update scale
    #     self._bbox[0] = [points_combined.min(), points_combined.max()]
    #     self._bbox[1] = [values.min(axis=0), values.max(axis=0)]
    #     self._scale[0] = self._bbox[0][1] - self._bbox[0][0]
    #     self._scale[1] = np.max(self._bbox[1][1] - self._bbox[1][0])
    #     self._oldscale = deepcopy(self._scale)
    #
    #     # Find the intervals for which the losses should be calculated.
    #     intervals, intervals_combined = [
    #         [(x_m, x_r) for x_m, (x_l, x_r) in neighbors.items()][:-1]
    #         for neighbors in (self.neighbors, self.neighbors_combined)
    #     ]
    #
    #     # The the losses for the "real" intervals.
    #     self.losses = loss_manager(self._scale[0])
    #     for ival in intervals:
    #         self.losses[ival] = self._get_loss_in_interval(*ival)
    #
    #     # List with "real" intervals that have interpolated intervals inside
    #     to_interpolate = []
    #
    #     self.losses_combined = loss_manager(self._scale[0])
    #     for ival in intervals_combined:
    #         # If this interval exists in 'losses' then copy it otherwise
    #         # calculate it.
    #         if ival in reversed(self.losses):
    #             self.losses_combined[ival] = self.losses[ival]
    #         else:
    #             # Set all losses to inf now, later they might be udpdated if the
    #             # interval appears to be inside a real interval.
    #             self.losses_combined[ival] = np.inf
    #             x_left, x_right = ival
    #             a, b = to_interpolate[-1] if to_interpolate else (None, None)
    #             if b == x_left and (a, b) not in self.losses:
    #                 # join (a, b) and (x_left, x_right) → (a, x_right)
    #                 to_interpolate[-1] = (a, x_right)
    #             else:
    #                 to_interpolate.append((x_left, x_right))
    #
    #     for ival in to_interpolate:
    #         if ival in reversed(self.losses):
    #             # If this interval does not exist it should already
    #             # have an inf loss.
    #             self._update_interpolated_loss_in_interval(*ival)

    # def remove_unfinished(self):
    #     self.pending_points = {}
    #     self.losses_combined = deepcopy(self.losses)
    #     self.neighbors_combined = deepcopy(self.neighbors)

    # def plot(self, *, with_sem=True):
    #     hv = ensure_holoviews()
    #
    #     xs = sorted(self._data.keys())
    #     vdims = ["mean", "standard_error", "std", "n"]
    #     values = [[getattr(self._data[x], attr) for x in xs] for attr in vdims]
    #     scatter = hv.Scatter((xs, *values), vdims=vdims)
    #
    #     if not with_sem:
    #         plot = scatter.opts(plot=dict(tools=["hover"]))
    #     else:
    #         ys, sems, *_ = values
    #         err = [x if x < sys.float_info.max else np.nan for x in sems]
    #         spread = hv.Spread((xs, ys, err))
    #         plot = scatter * spread
    #     return plot.opts(hv.opts.Scatter(tools=["hover"]))

    # def _set_data(self, data):
    #     # change dict -> DataPoint, because the points are saved using dicts
    #     data = {k: DataPoint(v) for k, v in data.items()}
    #     self.tell_many(data.keys(), data.values())
