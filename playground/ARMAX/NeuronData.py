class NeuronData: 
    # Contain all the input information we want to create the model
        #
        #   neuronData (Struct)
        #       .spikeTime:         All the spike time for entire experiment
        #
        #       .signalTime:        Structure contain absolute time of all
        #                           signal for each trial.
        #                           Name of each field represents name of each
        #                           signal
        #
        #       .signalValue:       Structure contain values of all the possible
        #                           signal for each trial. There might be signals
        #                           with signalTime but without signalValue
        #                           Name of each field represents name of each
        #                           signal
        #
        #       .endOfTrialSignal:  Name of the signal that indicates the end of the
        #                           trial. There might be a time distance
        #                           before/after "endOfTrialSignal" to the real end
        #                           of the trial which can be set by
        #                           "endOfTrialDist". (String)
        #
        #       .endOfTrialDist:    Time distanse from "endOfTrialSignal" that
        #                           indicated the real end of the trial. (msec)
        #
        #       .maxTrialLen:       Maximum time for each trial. Useful for
        #                           arranging data into bins. (msec)
        #                           The real value for trial length would be the
        #                           caculated by
        #                               "(signalTime.(endOfTrialSignal)(currTrial) +
        #                               endOfTrialDist) - ...
        #                               (signalTime.(endOfTrialSignal)(currTrial-1) +
        #                               endOfTrialDist);"
        #
        #       .beginOfTrialSignal:    Name of the signal that indicates the begin of the
        #                               trial. (String)
        #
        #       .binSize:           Size of each bin (msec)
        #
        #       .trialRange:        Range of trials; Total number of trials.
        
    def __init__(self):
        pass