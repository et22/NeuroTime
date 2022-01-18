import numpy as np
from scipy import stats, optimize

def exponential_estimation(spike_train: np.array, bin_size: int = 50):
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

    corr_mtx = np.zeros((num_bins, num_bins))
    lag_list = np.zeros(int((num_bins**2-num_bins)/2))
    corr_list = np.zeros(int((num_bins**2-num_bins)/2))
    list_idx = 0

    for i in np.arange(0, num_bins):
        for j in np.arange(i+1, num_bins):
            r, p = stats.pearsonr(spike_counts[:, i], spike_counts[:, j])
            
            # corr mtx
            corr_mtx[i, j] = r

            # lag and corr lists 
            lag_list[list_idx] = j-i
            corr_list[list_idx] = r
            list_idx += 1

    f = lambda x, tau, A, B: A*(np.exp(-(bin_size*x)/tau)+B)
    popt, pcov = optimize.curve_fit(f, lag_list, corr_list)
    tau, A, B = popt

    return tau


        