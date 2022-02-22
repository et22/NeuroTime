import numpy as np
from dataclasses import dataclass
from ARMAXModel import ARMAXModel
from ModelDefinition import ModelDefinition
from data_utils import SignalTimes, SignalValues

@dataclass
class NeuronData:
    trial_start_times: list[float]
    trial_end_times: list[float]
    spike_times: np.ndarray
    signal_times: SignalTimes
    signal_values: SignalValues

    def to_armax_input(self, model: ARMAXModel, model_definition: ModelDefinition, bin_size: int = 50, save_path: str = './armax_input.npy'):
        binned_spikes, bin_times = self.__bin_trial_spikes(bin_size)
        nrows = len(np.concatenate(bin_times))
        ncols = len(model.signal_labels)
        num_trials = len(bin_times)
        values = np.nan(shape=(nrows, ncols))
        times = np.nan(shape=(nrows, ncols))

        idx = 0
        for tr in model_definition.tr_comps:
            tr_value = getattr(self.signal_values, tr.name)
            tr_time = getattr(self.signal_times, tr.name)

            trial_times = []
            trial_values = []
            for trial in range(num_trials):
                num_bins = len(binned_spikes[trial])
                trial_times[trial] = bin_times[trial]<tr_time[trial]
                trial_values[trial] = tr_value[trial]*np.ones(shape=(num_bins))
            
            values[:,idx] = np.vstack(trial_values)
            times[:,idx] = np.vstack(trial_times)

            idx+=1
        
        for ar in model_definition.ar_comps:
            trial_times = []
            trial_values = []

            for trial in range(num_trials):
                num_bins = len(binned_spikes[trial])

                trial_values[trial] = np.ones(shape=(num_bins,ar.depth))
                trial_times[trial] = np.ones(shape=(num_bins,ar.depth))
                for bin in range(num_bins):
                    if ar.name == "ar_intrinsic":
                        if bin >= ar.depth:
                            trial_values[trial][bin,:] = binned_spikes[trial][bin-ar.depth:bin]
                        else:
                            trial_values[trial][bin,:] = np.nan(shape=(ar.depth))
                    elif ar.name == "ar_seasonal":
                        if trial-ar.depth >= 0:
                            trial_values[trial][bin,:] = np.array([binned_spikes[trial-k][bin] for k in range(1,1+ar.depth)])
                        else:
                            trial_values[trial][bin,:] = np.nan(shape=(ar.depth))
                        
                    else: 
                        raise ValueError("Invalid autoregressive component name")
                
                if ar.name == "ar_intrinsic":
                    trial_times[trial] = trial_times[trial]*[np.arange(start=-ar.depth*bin_size, stop=0, step=bin_size)]
                elif ar.name == "ar_seasonal":
                    sts = np.array(self.trial_start_times)
                    diff = np.nanmean(sts[1:]-sts[:-1])
                    trial_times[trial] = trial_times[trial]*[np.arange(start=-ar.depth*diff, stop=0, step=diff)]

                
            values[:,idx:idx+ar.depth] = np.vstack(trial_values)
            times[:,idx:idx+ar.depth] = np.vstack(trial_times)

            idx += ar.depth
        
        for exo in model_definition.exo_comps:
            exo_value = getattr(self.signal_values, exo.name)
            exo_time = getattr(self.signal_times, exo.name)

            trial_times = []
            trial_values = []
            for trial in range(num_trials):
                num_bins = len(binned_spikes[trial])
                if trial-exo.depth >= 0:
                    trial_times[trial] = exo_time[trial-exo.depth:trial]
                    trial_values[trial] = exo_value[trial-exo.depth:trial]
                else:
                    trial_times[trial] = np.nan(shape=(exo.depth))
                    trial_values[trial] = np.nan(shape=(exo.depth))

                trial_times[trial] = np.tile(trial_times[trial], (num_bins, 1))
                trial_values[trial] = np.tile(trial_values[trial], (num_bins, 1))

                trial_times[trial] = bin_times[trial] - trial_times[trial]

            values[:,idx:idx+exo.depth] = np.vstack(trial_values)
            times[:,idx:idx+exo.depth] = np.vstack(trial_times)

            idx += exo.depth            
        
        intrinsic_size = bin_size

        sts = np.array(self.trial_start_times)
        diff = np.nanmean(sts[1:]-sts[:-1])
        seasonal_size = diff

        y = np.hstack(binned_spikes)
        
        values[:,-1] = np.nanmean(y)

        X = np.hstack((values, times))
        
        return X, y, intrinsic_size, seasonal_size

    def __bin_trial_spikes(self, bin_size: int):
        """
        Converts spike_times into binned spike times for each trial based on the start and end of the trial. 
        
        Parameters:
        bin_size : int
            size of bin in milliseconds for binning spikes

        Return:
        binned_spikes : list[np.ndarray]
            binned_spikes is a list of length num_trials of arrays of length num_bins/trial with # spikes in bin
        bin_times : list[np.ndarray]
            bin_times is a list of length num_trials of arrays of length num_bins/trial with time of bin onset
                
        """
        binned_spikes = []
        bin_times = []
        for start_time, end_time in zip(self.trial_start_times, self.trial_end_times):
            trial_spikes = self.spike_times[np.logical_and(self.spike_times > start_time, self.spike_times < end_time)]
            num_bins = np.ceil((end_time - start_time)/bin_size)
            range = (start_time, start_time + num_bins*bin_size)
            hist, bin_edges = np.histogram(trial_spikes, bins = num_bins, range = range)
            binned_spikes.append(hist)
            bin_times.append(bin_edges)

        return binned_spikes, bin_times