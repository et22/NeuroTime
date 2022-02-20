import numpy as np
from scipy import stats, optimize
import abcTau 

def abc_estimation(spike_train: np.array, bin_size: int = 50):
    """
    Exponential estimation computes the autocorrelation timescale of an 
    individual neuron by fitting an exponential function.
    -spike_train: a num_trials x num_timebins array of spikes.
    -bin_size: size of bin to use for computing autocorrelation.
    """

    # bin data into bin_size bins and compute spike count
    num_trials, num_ms_bins = spike_train.shape
    num_bins = int(np.floor(num_ms_bins/bin_size))

    # don't include extra spikes if length of spike trains is not divisible by bin_size
    spike_train = spike_train[:, :num_bins*bin_size]

    binned_spikes = np.reshape(spike_train, (num_trials, num_bins, bin_size))
    spike_counts = np.sum(binned_spikes, axis=2)

    ### ABC
    # select summary statistics metric
    summStat_metric = 'comp_ac_fft'
    ifNorm = True # if normalize the autocorrelation or PSD

    # extract statistics from real data
    deltaT = 1 # temporal resolution of data.
    binSize = 1 #  bin-size for binning data and computing the autocorrelation.
    disp = None # put the disperssion parameter if computed with grid-search
    maxTimeLag = 50 # only used when suing autocorrelation for summary statistics
    data_sumStat, data_mean, data_var, T, numTrials =  abcTau.preprocessing.extract_stats(spike_counts, deltaT, binSize,\
                                                                                    summStat_metric, ifNorm, maxTimeLag)

    # Define a uniform prior distribution over the given range
    # for a uniform prior: stats.uniform(loc=x_min,scale=x_max-x_min)
    t_min = 0.0 # first timescale
    t_max = 100.0
    priorDist = [stats.uniform(loc= t_min, scale = t_max - t_min)]  

    # select generative model and distance function
    generativeModel = 'oneTauOU'
    distFunc = 'linear_distance'


    # set fitting params
    epsilon_0 = 1  # initial error threshold
    min_samples = 5 # min samples from the posterior
    steps = 1 # max number of iterations
    minAccRate = 1 # minimum acceptance rate to stop the iterations
    parallel = False # if parallel processing
    n_procs = 1 # number of processor for parallel processing (set to 1 if there is no parallel processing)     


    # creating model object
    class MyModel(abcTau.Model):

        #This method initializes the model object.  
        def __init__(self):
            pass

        # draw samples from the prior. 
        def draw_theta(self):
            theta = []
            for p in self.prior:
                theta.append(p.rvs())
            return theta

        # Choose the generative model (from generative_models)
        # Choose autocorrelation computation method (from basic_functions)
        def generate_data(self, theta):
            # generate synthetic data
            numTrials = 2
            data_mean = 1
            data_var = 1

            if disp == None:
                syn_data, numBinData =  eval('abcTau.generative_models.' + generativeModel + \
                                            '(theta, deltaT, binSize, T, numTrials, data_mean, data_var)')
            else:
                syn_data, numBinData =  eval('abcTau.generative_models.' + generativeModel + \
                                            '(theta, deltaT, binSize, T, numTrials, data_mean, data_var, disp)')
                
            # compute the summary statistics
            syn_sumStat = abcTau.summary_stats.comp_sumStat(syn_data, summStat_metric, ifNorm, deltaT, binSize, T,\
                                            numBinData, maxTimeLag)   
            return syn_sumStat

        # Computes the summary statistics
        def summary_stats(self, data):
            sum_stat = data
            return sum_stat

        # Choose the method for computing distance (from basic_functions)
        def distance_function(self, data, synth_data):
            if np.nansum(synth_data) <= 0: # in case of all nans return large d to reject the sample
                d = 10**4
            else:
                d = eval('abcTau.distance_functions.' +distFunc + '(data, synth_data)')        
            return d

    # path for loading and saving data
    datasave_path = 'example_abc_results/'
    dataload_path = 'example_data/'

    # path and filename to save the intermediate results after running each step
    inter_save_direc = 'example_abc_results/'
    inter_filename = 'abc_intermediate_results'

    # define filename for loading and saving the results
    filename = 'OU_tau20_mean0_var1_rawData'
    filenameSave = filename
    # fit with aABC algorithm for any generative model
    abc_results, final_step = abcTau.fit.fit_withABC(MyModel, data_sumStat, priorDist, inter_save_direc, inter_filename,\
                                                 datasave_path,filenameSave, epsilon_0, min_samples, \
                                                 steps, minAccRate, parallel, n_procs, disp) 

    theta_accepted = abc_results[final_step-1]['theta accepted']
    tau1 = np.mean(theta_accepted[0])                   
    return tau1
