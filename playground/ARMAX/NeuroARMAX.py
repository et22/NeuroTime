import numpy as np
from numpy.matlib import repmat

from playground.ARMAX.ModelComponent import ModelComponent


class ARMAXNeuro:
    """
    ARMAXNEURO is a class for estimating short, seasonal, and exogenous
    (reward and choice) memories plus exogenous selctivity of neurons.
       
    To construct an instance use "ARMAXNeuro(neuronData)"
    
    Copyright 2021 Ethan Trepka (ethan.trepka@gmail.com), Mehran M. Spitmaan (mehran.m.spitman@gmail.com).
    """
    def __init__(self, neuronData, display = 1, saveAddress='./MyNeuralDataStorage/', saveNamePrefix='MyNeuralModel'): 
        ## Setting parameters
        self.neuronData = neuronData
        self.display = display                        # Display messages in output
        self.saveAddress = saveAddress                    # Saving folder address
        self.saveNamePrefix = saveNamePrefix                 # Prefix of saving file name

        # default properties
        self.ITI = []                       # Storage for ITI
        self.timeIntervalsAbsolute = None   # Absolute time values for each bin and trials
        self.startPoint = 2                 # starting point for trials
        
        self.firingRateMat = None            # Total Firing rate for trial and bins
        self.maxOrder = 0                   # maximum memory order of component
        self.boundries = None                     # boudries for trials and bins
        self.totalDataPoint = None                # Total number of datapoint
        self.firingRateMatMean = None              # Total Firing rate for bins (neural profile)
        self.firingRateMatMeanArranged = None      # Total Firing rate for bins (neural profile) rearranged for all data points   
        
        self.fittingModelData = None               # Final Data needed for fitting the model
        self.paramSet = None                     # Data structure for parameter set and their boundries
        self.fittingSet = None                     # Data structure for fitting procedure
        self.fittingResults = None                 # Data structure for fitting results
        
        self.zscoring = 1                   # Indicates needs for zscoring the data
        
        ## Data Structure for Model       
        self.component = ModelComponent()
    
    # Set whether model should display messages or not
    def setDisplay(self, display):
        self.display = display
    
    # Initializing Data
    def initData(self):
        #   Initiates the data and make it ready for further
        #   computation: 
        #   - Computing trial_alignment based on signals
        #   - Computing the firing rates
        #   - Computing ITI
        
        self.displayMsg('Start initiation process...')
        
        # Computing ITI
        self.compITI()
        
        # Compute absolute time intevals
        startPointEst = self.neuronData.maxTrialLen-self.neuronData.endOfTrialDist
        alignSignTime = getattr(self.neuronData.signalTime, self.neuronData.endOfTrialSignal)
        
        binsPerTrial = np.arange(-startPointEst, self.neuronData.endOfTrialDist+1, self.neuronData.binSize).T
        binsPerTrial = np.expand_dims(binsPerTrial, axis=0)
        self.timeIntervalsAbsolute = repmat(binsPerTrial,alignSignTime.shape[0], 1) + repmat(alignSignTime, 1, binsPerTrial.shape[1])

        # Calculating Fairing Rate Matrix
        self.firingRateMat = []
        
        trialFilter = np.arange(self.startPoint, self.timeIntervalsAbsolute.shape[0]) # removing one trial on begening and one at the  for the sake of binning the spike trian
        
        for cntTrial in trialFilter:           
            # All spike in current trial
            allSpikes = self.neuronData.spikeTime
            
            trialtimeIntervalsAbsolute = self.timeIntervalsAbsolute[cntTrial,:]
            
            spikeCount, _ = np.histogram(allSpikes,trialtimeIntervalsAbsolute)
            
            # make the spike count equal to nan, if the bin is in
            # previous trial
            idx = (trialtimeIntervalsAbsolute+self.neuronData.binSize) < alignSignTime(cntTrial-1)+self.neuronData.endOfTrialDist
            spikeCount[idx] = np.nan
            
            # storing firing rate
            self.firingRateMat = self.firingRateMat.append(spikeCount/(self.neuronData.binSize/1000))                
        
        # triming absolute times based on "trialFilter"
        self.timeIntervalsAbsolute = self.timeIntervalsAbsolute[trialFilter,1:]
        
        # triming signal value based on "trialFilter"
        signalValueNames = self.neuronData.signalValue.keys()
        for name in signalValueNames:
            curr_name = getattr(self.neuronData.signalValue, 'name')
            setattr(self.neuronData.signalValue, 'name', curr_name[trialFilter])
        
        # triming signal times based on "trialFilter"
        signalTimeNames = self.neuronData.signalTime.keys()
        for name in signalTimeNames:
            curr_name = getattr(self.neuronData.signalTime, 'name')
            setattr(self.neuronData.signalTime, 'name', curr_name[trialFilter])

        
        # calculate mean firing pattern in profile
        self.firingRateMatMean = np.nanmean(self.firingRateMat,0)                      

        # calculating boudries
        self.calcTrialBinBoudries()
        
        # calculating model components
        self.initMemShort()
        self.initMemSeason()
        self.initMemExo()
        self.initExoSignal()

        self.displayMsg('Initiation process is done!')
            
    # Setup fitting procedure information
    def setFittingSet(self, iterNum = 20, crossPer = 0.2, crossIter = 10, modelSelection = 1):            
        self.displayMsg('Setup fitting parameters...')
        
        self.fittingSet.iterNum = iterNum
        self.fittingSet.crossPer = crossPer
        self.fittingSet.crossIter = crossIter           
        self.fittingSet.modelSelection = modelSelection        
        
        self.fittingSet.selectedComponents.seqOrderName = dict() # a dictonary for converting seq order num to name of components
        self.fittingSet.selectedComponents.totalPermutationNumber = [] # total number of model permutations
        self.fittingSet.selectedComponents.currentSelectedSeq = [] # The current selected sequence of components (binary seq)
        
        # Selection of ExoSignals
        self.fittingSet.selectedComponents.exoSignal.selected = len(self.component.exoSignal.keys()) != 0
        seqOrderLen = len(self.fittingSet.selectedComponents.seqOrderName.keys())
        self.fittingSet.selectedComponents.seqOrderName[seqOrderLen] = 'exoSignal'
        self.fittingSet.selectedComponents.exoSignal.seqNumber = seqOrderLen+1
        
        # Selection of short and seasonal memories
        self.fittingSet.selectedComponents.memShort.selected = self.component.memShort.defined
        self.fittingSet.selectedComponents.seqOrderName{+1} = 'memShort'
        self.fittingSet.selectedComponents.memShort.seqNumber = length(self.fittingSet.selectedComponents.seqOrderName)
        
        self.fittingSet.selectedComponents.memSeason.selected = self.component.memSeason.defined
        self.fittingSet.selectedComponents.seqOrderName{+1} = 'memSeason'
        self.fittingSet.selectedComponents.memSeason.seqNumber = length(self.fittingSet.selectedComponents.seqOrderName)
        

        # Calculating exogenous memory selection
        if ~isempty(fieldnames(self.component.memExo))
            exoMemNames = fieldnames(self.fittingModelData.memExo)
            
            for cntExoMem = 1:length(exoMemNames)
                exoMemNameTemp = exoMemNames{cntExoMem}
                self.fittingSet.selectedComponents.memExo.(exoMemNameTemp).selected = 1
                self.fittingSet.selectedComponents.seqOrderName{+1} = ['memExo.',exoMemNameTemp]
                self.fittingSet.selectedComponents.memExo.(exoMemNameTemp).seqNumber = length(self.fittingSet.selectedComponents.seqOrderName)
            
        
                    
        self.fittingSet.selectedComponents.totalPermutationNumber = 2.^length(self.fittingSet.selectedComponents.seqOrderName)
        self.fittingSet.selectedComponents.currentSelectedSeq = ones(1,length(self.fittingSet.selectedComponents.seqOrderName))
    
    
    # Fit the model
    def fit(self):     
        self.saveModel()
        
        if self.fittingSet.modelSelection
            self.displayMsg('Start Model Selection Process...')
            
            for cntModel = 0:self.fittingSet.selectedComponents.totalPermutationNumber-1
                self.displayMsg(['Running model [',num2str(cntModel),']...'])
                                    
                # Set current selection Mask
                self.fittingSet.selectedComponents.currentSelectedSeq = dec2bin(cntModel,length(self.fittingSet.selectedComponents.currentSelectedSeq))
                self.fittingSet.selectedComponents.currentSelectedSeq = self.fittingSet.selectedComponents.currentSelectedSeq(:-1:1)
                
                self.fittingSet.currentModel = cntModel+1                    
                
                # Setup parameter set and boundries
                self.calcParamSet()
        
                # Fit a single model
                fit_singleModel(self)
                
                self.displayMsg([' of model [',num2str(cntModel),']...'])
            
            
            self.chooseBestModel()                                
            
        else
            self.displayMsg('Running General Model...')
            cntModel = 0
            self.fittingSet.currentModel = cntModel+1
            
            # Set current selection Mask
            self.fittingSet.selectedComponents.currentSelectedSeq = dec2bin(self.fittingSet.selectedComponents.totalPermutationNumber-1,length(self.fittingSet.selectedComponents.currentSelectedSeq))
            self.fittingSet.selectedComponents.currentSelectedSeq = self.fittingSet.selectedComponents.currentSelectedSeq(:-1:1)
            
            # Setup parameter set and boundries
            self.calcParamSet()
            
            # Fit a single model
            fit_singleModel(self)
            self.displayMsg(' of General Model...')
        
            
        self.saveModelSnapshot(1)
        self.saveModel(1)
    
    
    # Fit a single model
    def fit_singleModel(self)           
        dataMask = []
        dataMask.mask = logical(ones(size(self.fittingModelData.output)))'
        minFunc = @(v,d) self.costdef(v,d.mask)
        positiveFlag = self.paramSet.limit.sorted(:,1) >=0
        limitSet = {}
        for cntV = 1:size(self.paramSet.limit.sorted,1)
            limitSet{cntV} = self.paramSet.limit.sorted(cntV,:)
        
        
        [vars, stat, ret] = Fitting_def(minFunc, dataMask, size(self.paramSet.limit.sorted,1),...
            self.fittingSet.iterNum, self.fittingSet.crossPer, self.fittingSet.crossIter,...
            positiveFlag, limitSet, {'mask'}, 1)           
        
        self.fittingResults.models{self.fittingSet.currentModel}.res.vars = vars
        self.fittingResults.models{self.fittingSet.currentModel}.res.stat = stat
        self.fittingResults.models{self.fittingSet.currentModel}.res.ret = ret
        
        self.fittingResults.models{self.fittingSet.currentModel}.paramSet = self.paramSet
        
        self.saveModelSnapshot()
    

    def displayMsg(self, msg):
        if self.display
            display(msg)
        
    
    
    # Compute InterTrial Intervals
    def compITI(self)
        #   Computing ITI for all trials
        self.displayMsg('Computing ITI for all trials...')
        
        for cntTemp= 2:self.neuronData.trialRange()
            ITITemp(cntTemp) = self.neuronData.signalTime.(self.neuronData.beginOfTrialSignal)(cntTemp) - self.neuronData.signalTime.(self.neuronData.endOfTrialSignal)(cntTemp-1) # All the ITIs
        
        self.ITI = ITITemp
                    
    
    # Compute Short term memory
    def initMemShort(self)
        if (self.component.memShort.defined)
            #   Setup the neural data for Short memory
            self.displayMsg('Setup the neural data for Short memory...')
            
            dataPointIdx = 0
            self.fittingModelData.memShort.Rate = zeros(self.totalDataPoint, self.component.memShort.order)
            self.fittingModelData.memShort.Time = zeros(self.totalDataPoint, self.component.memShort.order)
            
            for cntTrials = self.boundries.trial
                for cntBins = self.boundries.bin
                    # clc
                    if ~isnan(self.firingRateMat(cntTrials,cntBins))
                        
                        # calculating the stating bin number of non nan
                        # data
                        startPointNoNAN = sum(isnan(self.firingRateMat(cntTrials,:)))+1
                        
                        # calculating the sliding windeo negative values
                        # indicate the value from previouse trial
                        binsSlidingBox = [cntBins-self.component.memShort.order:cntBins-1]
                        binsSlidingBox = [binsSlidingBox(binsSlidingBox<startPointNoNAN)-startPointNoNAN+1 binsSlidingBox(binsSlidingBox>=startPointNoNAN)]
                        
                        #
                        #                         yTemp = spikeCountsGeneral(cntTrials,cntBins)
                        #                         yMeanTemp = spikeCountsGeneralMean(cntBins)
                        #
                        #                         yTempZS = spikeCountsGeneralZS(cntTrials,cntBins)
                        #                         yMeanTempZS = spikeCountsGeneralMeanZS(cntBins)
                        
                        maxBinLen = size(self.firingRateMat,2)
                        
                        # Initializing data storages
                        ARSTemp = zeros(1,self.component.memShort.order)
                        timeS = zeros(1,self.component.memShort.order)
                        
                        cntId = 1
                        for cntSlidingBox = binsSlidingBox
                            if cntSlidingBox>0
                                ARSTemp(cntId) = self.firingRateMat(cntTrials,cntSlidingBox)
                                timeS(cntId) = self.timeIntervalsAbsolute(cntTrials,cntSlidingBox)
                                cntId = cntId + 1
                            else
                                ARSTemp(cntId) = self.firingRateMat(cntTrials-1,maxBinLen+cntSlidingBox)
                                timeS(cntId) = self.timeIntervalsAbsolute(cntTrials-1,maxBinLen+cntSlidingBox)
                                cntId = cntId + 1
                            
                        
                        
                        for cntBinTimeS = 1:length(timeS)-1
                            timeS(cntBinTimeS) = timeS(cntBinTimeS+1)-timeS(cntBinTimeS)
                        
                        timeS() = self.neuronData.binSize
                        
                        # Inverse the vectors
                        ARSTemp = ARSTemp(:-1:1)
                        timeS = timeS(:-1:1)
                        
                        dataPointIdx = dataPointIdx + 1
                        self.fittingModelData.memShort.Rate(dataPointIdx,:) = ARSTemp
                        self.fittingModelData.memShort.Time(dataPointIdx,:) = timeS
                        
                    
                    
                
            
            
            self.fittingModelData.memShort.mask = ones(self.totalDataPoint, self.component.memShort.order)
            if ~isempty(self.component.memShort.effectiveTimeWin)
                self.fittingModelData.memShort.mask = self.clcTimWin(self.component.memShort.effectiveTimeWin, self.fittingModelData.memShort.mask)
            
        
           
    
    # Compute Seasonal memory
    def initMemSeason(self)
        if (self.component.memSeason.defined)
            #   Setup the neural data for Seasonal memory
            self.displayMsg('Setup the neural data for Seasonal memory...')
            
            dataPointIdx = 0
            self.fittingModelData.memSeason.Rate = zeros(self.totalDataPoint, self.component.memSeason.order)
            self.fittingModelData.memSeason.Time = zeros(self.totalDataPoint, self.component.memSeason.order)
            for cntTrials = self.boundries.trial
                for cntBins = self.boundries.bin
                    # clc
                    if ~isnan(self.firingRateMat(cntTrials,cntBins))
                        
                        ARLTemp = self.firingRateMat(cntTrials-self.component.memSeason.order:cntTrials-1,cntBins)
                        timeL = self.timeIntervalsAbsolute(cntTrials-self.component.memSeason.order:cntTrials-1,cntBins)
                        
                        # Inverse the vectors
                        ARLTemp = ARLTemp(:-1:1)
                        timeL = timeL(:-1:1)
                        
                        dataPointIdx = dataPointIdx + 1
                        self.fittingModelData.memSeason.Rate(dataPointIdx,:) = ARLTemp
                        self.fittingModelData.memSeason.Time(dataPointIdx,:) = timeL
                        
                    
                    
                
            
            
            self.fittingModelData.memSeason.mask = ones(self.totalDataPoint, self.component.memSeason.order)
            if ~isempty(self.component.memSeason.effectiveTimeWin)
                self.fittingModelData.memSeason.mask = self.clcTimWin(self.component.memSeason.effectiveTimeWin, self.fittingModelData.memSeason.mask)
            
        
    
    
    # Compute exogenous memories
    def initMemExo(self)
        if ~isempty(fieldnames(self.component.memExo))
            #   Setup the neural data for exo memories
            self.displayMsg('Setup the neural data for exo memories...')
            
            exoMemNames = fieldnames(self.component.memExo)
            
            for cntExoMem = 1:length(exoMemNames)
                exoMemNameTemp = exoMemNames{cntExoMem}
                exoMemTemp = self.component.memExo.(exoMemNameTemp)
                
                if ~exoMemTemp.interaction
                    if ~exoMemTemp.exoSelEff
                        
                        dataPointIdx = 0
                        self.fittingModelData.memExo.(exoMemNameTemp).Signal = zeros(self.totalDataPoint, exoMemTemp.memOrder)
                        self.fittingModelData.memExo.(exoMemNameTemp).Time = zeros(self.totalDataPoint, exoMemTemp.memOrder)
                        for cntTrials = self.boundries.trial
                            for cntBins = self.boundries.bin
                                # clc
                                if ~isnan(self.firingRateMat(cntTrials,cntBins))
                                    
                                    # clc
                                    # current time bigger than the signal
                                    if self.timeIntervalsAbsolute(cntTrials,cntBins) > (self.neuronData.signalTime.(exoMemTemp.signalTime)(cntTrials))
                                        exoSigTemp = self.neuronData.signalValue.(exoMemTemp.signalName)(cntTrials-exoMemTemp.memOrder+1:cntTrials)
                                        exoTimeTemp = self.timeIntervalsAbsolute(cntTrials,cntBins) -...
                                            self.neuronData.signalTime.(exoMemTemp.signalTime)(cntTrials-exoMemTemp.memOrder+1:cntTrials)
                                        
                                    else # current time smaller than the signal
                                        exoSigTemp = self.neuronData.signalValue.(exoMemTemp.signalName)(cntTrials-exoMemTemp.memOrder:cntTrials-1)
                                        exoTimeTemp = self.timeIntervalsAbsolute(cntTrials,cntBins) -...
                                            self.neuronData.signalTime.(exoMemTemp.signalTime)(cntTrials-exoMemTemp.memOrder:cntTrials-1)
                                    
                                    
                                    # Inverse the vectors
                                    exoSigTemp = exoSigTemp(:-1:1)
                                    exoTimeTemp = exoTimeTemp(:-1:1)
                                    
                                    dataPointIdx = dataPointIdx + 1
                                    self.fittingModelData.memExo.(exoMemNameTemp).Signal(dataPointIdx,:) = exoSigTemp
                                    self.fittingModelData.memExo.(exoMemNameTemp).Time(dataPointIdx,:) = exoTimeTemp
                                    
                                
                            
                        
                        
                    else
                        
                        dataPointIdx = 0
                        self.fittingModelData.memExo.(exoMemNameTemp).Signal = zeros(self.totalDataPoint, exoMemTemp.memOrder)
                        self.fittingModelData.memExo.(exoMemNameTemp).Time = zeros(self.totalDataPoint, exoMemTemp.memOrder)
                        for cntTrials = self.boundries.trial
                            for cntBins = self.boundries.bin
                                # clc
                                if ~isnan(self.firingRateMat(cntTrials,cntBins))
                                    
                                    exoSigTemp = self.neuronData.signalValue.(exoMemTemp.signalName)(cntTrials-exoMemTemp.memOrder:cntTrials-1)
                                    exoTimeTemp = self.timeIntervalsAbsolute(cntTrials,cntBins) -...
                                        self.neuronData.signalTime.(exoMemTemp.signalTime)(cntTrials-exoMemTemp.memOrder:cntTrials-1)
                                    
                                    # Inverse the vectors
                                    exoSigTemp = exoSigTemp(:-1:1)
                                    exoTimeTemp = exoTimeTemp(:-1:1)
                                    
                                    dataPointIdx = dataPointIdx + 1
                                    self.fittingModelData.memExo.(exoMemNameTemp).Signal(dataPointIdx,:) = exoSigTemp
                                    self.fittingModelData.memExo.(exoMemNameTemp).Time(dataPointIdx,:) = exoTimeTemp
                                    
                                
                            
                        
                        
                    
                else
                    if ~exoMemTemp.exoSelEff
                        
                        dataPointIdx = 0
                        self.fittingModelData.memExo.(exoMemNameTemp).Signal = zeros(self.totalDataPoint, exoMemTemp.memOrder)
                        self.fittingModelData.memExo.(exoMemNameTemp).Time = zeros(self.totalDataPoint, exoMemTemp.memOrder)
                        for cntTrials = self.boundries.trial
                            for cntBins = self.boundries.bin
                                # clc
                                if ~isnan(self.firingRateMat(cntTrials,cntBins))
                                    
                                    # clc
                                    # current time bigger than the signal
                                    if self.timeIntervalsAbsolute(cntTrials,cntBins) > (self.neuronData.signalTime.(exoMemTemp.signalTime)(cntTrials))
                                        for cntSignals = 1:length(exoMemTemp.signalName)
                                            exoSigTemp(:,cntSignals) = self.neuronData.signalValue.(exoMemTemp.signalName{cntSignals})(cntTrials-exoMemTemp.memOrder+1:cntTrials)
                                        
                                        
                                        exoSigTemp = self.interactionCalc(exoSigTemp, exoMemTemp.interactionType)
                                        exoTimeTemp = self.timeIntervalsAbsolute(cntTrials,cntBins) -...
                                            self.neuronData.signalTime.(exoMemTemp.signalTime)(cntTrials-exoMemTemp.memOrder+1:cntTrials)
                                        
                                    else # current time smaller than the signal
                                        for cntSignals = 1:length(exoMemTemp.signalName)
                                            exoSigTemp(:,cntSignals) = self.neuronData.signalValue.(exoMemTemp.signalName{cntSignals})(cntTrials-exoMemTemp.memOrder:cntTrials-1)
                                        
                                        
                                        exoSigTemp = self.interactionCalc(exoSigTemp, exoMemTemp.interactionType)
                                        exoTimeTemp = self.timeIntervalsAbsolute(cntTrials,cntBins) -...
                                            self.neuronData.signalTime.(exoMemTemp.signalTime)(cntTrials-exoMemTemp.memOrder:cntTrials-1)
                                    
                                    
                                    # Inverse the vectors
                                    exoSigTemp = exoSigTemp(:-1:1)
                                    exoTimeTemp = exoTimeTemp(:-1:1)
                                    
                                    dataPointIdx = dataPointIdx + 1
                                    self.fittingModelData.memExo.(exoMemNameTemp).Signal(dataPointIdx,:) = exoSigTemp
                                    self.fittingModelData.memExo.(exoMemNameTemp).Time(dataPointIdx,:) = exoTimeTemp
                                    
                                
                            
                        
                        
                    else
                        
                        dataPointIdx = 0
                        self.fittingModelData.memExo.(exoMemNameTemp).Signal = zeros(self.totalDataPoint, exoMemTemp.memOrder)
                        self.fittingModelData.memExo.(exoMemNameTemp).Time = zeros(self.totalDataPoint, exoMemTemp.memOrder)
                        for cntTrials = self.boundries.trial
                            for cntBins = self.boundries.bin
                                # clc
                                if ~isnan(self.firingRateMat(cntTrials,cntBins))
                                    
                                    for cntSignals = 1:length(exoMemTemp.signalName)
                                        exoSigTemp(:,cntSignals) = self.neuronData.signalValue.(exoMemTemp.signalName{cntSignals})(cntTrials-exoMemTemp.memOrder:cntTrials-1)
                                    
                                    
                                    exoSigTemp = self.interactionCalc(exoSigTemp, exoMemTemp.interactionType)
                                    exoTimeTemp = self.timeIntervalsAbsolute(cntTrials,cntBins) -...
                                        self.neuronData.signalTime.(exoMemTemp.signalTime)(cntTrials-exoMemTemp.memOrder:cntTrials-1)
                                    
                                    # Inverse the vectors
                                    exoSigTemp = exoSigTemp(:-1:1)
                                    exoTimeTemp = exoTimeTemp(:-1:1)
                                    
                                    dataPointIdx = dataPointIdx + 1
                                    self.fittingModelData.memExo.(exoMemNameTemp).Signal(dataPointIdx,:) = exoSigTemp
                                    self.fittingModelData.memExo.(exoMemNameTemp).Time(dataPointIdx,:) = exoTimeTemp
                                    
                                
                            
                        
                        
                    
                
                
                self.fittingModelData.memExo.(exoMemNameTemp).meanData = self.firingRateMatMeanArranged
                
                self.fittingModelData.memExo.(exoMemNameTemp).mask = ones(self.totalDataPoint, self.component.memExo.(exoMemNameTemp).memOrder)
                if ~isempty(self.component.memExo.(exoMemNameTemp).effectiveTimeWin)
                    self.fittingModelData.memExo.(exoMemNameTemp).mask = self.clcTimWin(self.component.memExo.(exoMemNameTemp).effectiveTimeWin, self.fittingModelData.memExo.(exoMemNameTemp).mask)
                
                
            
        
    
    
    # Compute exogenous signals
    def initExoSignal(self)
        if ~isempty(fieldnames(self.component.exoSignal))
            #   Setup the neural data for exo signals
            self.displayMsg('Setup the neural data for exo signals...')
            
            exoSignalNames = fieldnames(self.component.exoSignal)
            
            for cntExoSignal = 1:length(exoSignalNames)
                exoSignalNameTemp = exoSignalNames{cntExoSignal}
                exoSignalTemp = self.component.exoSignal.(exoSignalNameTemp)
                
                if ~exoSignalTemp.interaction
                    dataPointIdx = 0
                    self.fittingModelData.exoSignal.(exoSignalNameTemp).Signal = zeros(self.totalDataPoint, 1)
                    
                    for cntTrials = self.boundries.trial
                        for cntBins = self.boundries.bin
                            # clc
                            if ~isnan(self.firingRateMat(cntTrials,cntBins))
                                
                                # clc
                                # current time bigger than the signal
                                if self.timeIntervalsAbsolute(cntTrials,cntBins) > (self.neuronData.signalTime.(exoSignalTemp.signalName)(cntTrials))
                                    exoSigTemp = self.neuronData.signalValue.(exoSignalTemp.signalName)(cntTrials-1+1:cntTrials)
                                else # current time smaller than the signal
                                    exoSigTemp = self.neuronData.signalValue.(exoSignalTemp.signalName)(cntTrials-1:cntTrials-1)
                                
                                
                                # Inverse the vectors
                                exoSigTemp = exoSigTemp(:-1:1)
                                dataPointIdx = dataPointIdx + 1
                                self.fittingModelData.exoSignal.(exoSignalNameTemp).Signal(dataPointIdx,:) = exoSigTemp
                            
                        
                    
                    
                else
                    
                    dataPointIdx = 0
                    self.fittingModelData.exoSignal.(exoSignalNameTemp).Signal = zeros(self.totalDataPoint, 1)
                    
                    for cntTrials = self.boundries.trial
                        for cntBins = self.boundries.bin
                            # clc
                            if ~isnan(self.firingRateMat(cntTrials,cntBins))
                                
                                # clc
                                for cntSignals = 1:length(exoSignalTemp.signalName)
                                    # current time bigger than the signal
                                    if self.timeIntervalsAbsolute(cntTrials,cntBins) > (self.neuronData.signalTime.(exoSignalTemp.signalName{cntSignals})(cntTrials))
                                        exoSigTemp(:,cntSignals) = self.neuronData.signalValue.(exoSignalTemp.signalName{cntSignals})(cntTrials-1+1:cntTrials)
                                    else # current time smaller than the signal
                                        exoSigTemp(:,cntSignals) = self.neuronData.signalValue.(exoSignalTemp.signalName{cntSignals})(cntTrials-1:cntTrials-1)
                                    
                                
                                
                                exoSigTemp = self.interactionCalc(exoSigTemp, exoSignalTemp.interactionType)
                                
                                # Inverse the vectors
                                exoSigTemp = exoSigTemp(:-1:1)
                                dataPointIdx = dataPointIdx + 1
                                self.fittingModelData.exoSignal.(exoSignalNameTemp).Signal(dataPointIdx,:) = exoSigTemp
                            
                        
                    
                    
                
                
                self.fittingModelData.exoSignal.(exoSignalNameTemp).mask = ones(self.totalDataPoint, 1)
                if ~isempty(self.component.exoSignal.(exoSignalNameTemp).effectiveTimeWin)
                    self.fittingModelData.exoSignal.(exoSignalNameTemp).mask = self.clcTimWin(self.component.exoSignal.(exoSignalNameTemp).effectiveTimeWin, self.fittingModelData.exoSignal.(exoSignalNameTemp).mask)
                
            
        
           
    
    # Compute trial boundries
    def calcTrialBinBoudries(self)
        # calculate maximum order
        allOrders = self.component.memSeason.order
        exoMemNames = fieldnames(self.component.memExo)
        for cntExoMems = 1:length(exoMemNames)
            allOrders(+1) = self.component.memExo.(exoMemNames{cntExoMems}).memOrder
        
        
        self.maxOrder = max(allOrders)
        
        # calculating boudries
        self.boundries.trial = self.maxOrder+2:size(self.firingRateMat,1)
        self.boundries.bin = 1:size(self.firingRateMat,2)
        
        # Calculating total datapoints
        self.totalDataPoint = 0
        self.fittingModelData.output = []
        self.firingRateMatMeanArranged = []
        for cntTrials = self.boundries.trial
            for cntBins = self.boundries.bin
                
                if ~isnan(self.firingRateMat(cntTrials,cntBins))
                    self.totalDataPoint = self.totalDataPoint + 1
                    self.fittingModelData.output(+1) = self.firingRateMat(cntTrials,cntBins)
                    self.firingRateMatMeanArranged(+1) = self.firingRateMatMean(cntBins)
                
            
        
    
    
    # Compute time windows for time mask
    def tempMask = clcTimWin(self, timeWin, tempMask)
        # initiate a temp mask
        dataPointIdx = 0
#             tempMask = ones(self.totalDataPoint, self.component.memShort.order)
        
        if timeWin.type == 1
            for cntTrials = self.boundries.trial
                for cntBins = self.boundries.bin
                    if ~isnan(self.firingRateMat(cntTrials,cntBins))
                        dataPointIdx = dataPointIdx + 1
                        
                        if ~ismember(cntBins,timeWin.binWindow) 
                            tempMask(dataPointIdx,:) = tempMask(dataPointIdx,:) * 0
                        
                    
                
            
        else
            
            for cntTrials = self.boundries.trial
                for cntBins = self.boundries.bin
                    # clc
                    if ~isnan(self.firingRateMat(cntTrials,cntBins))
                        dataPointIdx = dataPointIdx + 1
                        # clc
                        # current time smaller than the mask begin time
                        if self.timeIntervalsAbsolute(cntTrials,cntBins) < (self.neuronData.signalTime.(timeWin.beginSig_Name)(cntTrials)+timeWin.beginSig_TimeDist)
                            tempMask(dataPointIdx,:) = tempMask(dataPointIdx,:) * 0
                        # current time biger than the mask  time
                        elseif self.timeIntervalsAbsolute(cntTrials,cntBins) > (self.neuronData.signalTime.(timeWin.endSig_Name)(cntTrials)+timeWin.endSig_TimeDist) 
                            tempMask(dataPointIdx,:) = tempMask(dataPointIdx,:) * 0
                        
                        
                    
                
            
            
        
        
    
    
    # Compute interaction data
    def data = interactionCalc(self, data, interactionType)
        # Assumption is that size(data,2) shows the number of
        # parameters involved in interaction and values can be either 1
        # or -1.
        #
        # def retuns a unique value correspond to the binary value of
        # input between -(paramNum^2)/2 and (paramNum^2)/2, excluding 0
        
        if interactionType == 1
            data = prod(data,2)      
        elseif interactionType == 2
            paramNum = size(data,2)
            dataPos = (data+1)/2
            tempOutput = bin2dec(num2str(dataPos))+1
            if tempOutput <= (paramNum^2)/2
                tempOutput = tempOutput - ((paramNum^2)/2 + 1)
            else
                tempOutput = tempOutput - ((paramNum^2)/2)
            
        
        data = tempOutput
    
    
    # Setup parameter set and boundries
    def calcParamSet(self)
        
        self.displayMsg('Setup parameter settings...')
        
        curly = @(x, varargin) x{varargin{:}}
        
        self.paramSet.size.bias          = 1
        self.paramSet.size.exoSignal     = 1
        self.paramSet.size.memShort      = self.component.memShort.order
        self.paramSet.size.memSeason     = self.component.memSeason.order
        self.paramSet.size.memExo        = 2
        self.paramSet.size.mean          = 1
        
        
        self.paramSet.flag.bias          = []
        self.paramSet.flag.exoSignal     = []
        self.paramSet.flag.memShort      = []
        self.paramSet.flag.memSeason     = []
        self.paramSet.flag.memExo        = []
        self.paramSet.flag.mean          = []
        
        self.paramSet.effect.bias        = 0
        self.paramSet.effect.exoSignal   = 0
        self.paramSet.effect.memShort    = 0
        self.paramSet.effect.memSeason   = 0
        self.paramSet.effect.memExo      = 0
        self.paramSet.effect.mean        = 0
                    
        
        self.paramSet.limit.bias         = [-inf inf]
        self.paramSet.limit.exoSignal    = [-inf inf]
        self.paramSet.limit.memShort     = [-inf inf]
        self.paramSet.limit.memSeason    = [-inf inf]
        self.paramSet.limit.memExo       = [-inf inf]            
        self.paramSet.limit.mean         = [-inf inf]
        
        
        self.paramSet.flag.bias = 1
        currFlag = 1
        self.paramSet.effect.bias = 1
        
        self.paramSet.limit.sorted       = []
        self.paramSet.limit.sorted(1,:)  = [-inf inf]

        
        if ~isempty(fieldnames(self.component.exoSignal)) && ...
                str2num(self.fittingSet.selectedComponents.currentSelectedSeq(self.fittingSet.selectedComponents.exoSignal.seqNumber))
            self.paramSet.flag.exoSignal = [currFlag+1:currFlag+self.paramSet.size.exoSignal*(length(fieldnames(self.component.exoSignal)))]
            currFlag = self.paramSet.flag.exoSignal()
            self.paramSet.effect.exoSignal = 1
            
            for cntN = 1:length(fieldnames(self.component.exoSignal))                    
                self.paramSet.limit.exoSignal((cntN-1)*self.paramSet.size.exoSignal+1:cntN*self.paramSet.size.exoSignal,:) = ...
                    self.component.exoSignal.(curly(fieldnames(self.component.exoSignal),cntN)).paramLim
                self.paramSet.limit.sorted(+1:+self.paramSet.size.exoSignal,:)  = ...
                    self.component.exoSignal.(curly(fieldnames(self.component.exoSignal),cntN)).paramLim
            
                                
        
        if (self.component.memShort.defined) && ...
                str2num(self.fittingSet.selectedComponents.currentSelectedSeq(self.fittingSet.selectedComponents.memShort.seqNumber))
            self.paramSet.flag.memShort = [currFlag+1:currFlag+self.paramSet.size.memShort]
            currFlag = self.paramSet.flag.memShort()
            self.paramSet.effect.memShort = 1
            self.paramSet.limit.memShort(1:self.paramSet.size.memShort,:) = repmat(self.component.memShort.paramLim,[self.paramSet.size.memShort,1])
            self.paramSet.limit.sorted(+1:+self.paramSet.size.memShort,:)  = repmat(self.component.memShort.paramLim,[self.paramSet.size.memShort,1])
        
        
        if (self.component.memSeason.defined) && ...
                str2num(self.fittingSet.selectedComponents.currentSelectedSeq(self.fittingSet.selectedComponents.memSeason.seqNumber))
            self.paramSet.flag.memSeason = [currFlag+1:currFlag+self.paramSet.size.memSeason]
            currFlag = self.paramSet.flag.memSeason()
            self.paramSet.effect.memSeason = 1
            self.paramSet.limit.memSeason(1:self.paramSet.size.memSeason,:) = repmat(self.component.memSeason.paramLim,[self.paramSet.size.memSeason,1])
            self.paramSet.limit.sorted(+1:+self.paramSet.size.memSeason,:)  = repmat(self.component.memSeason.paramLim,[self.paramSet.size.memSeason,1])
        
        
        if ~isempty(fieldnames(self.component.memExo))
            exoMemNames = fieldnames(self.fittingModelData.memExo)
            exoEffeciveNumTemp = 0 # store the numbers of effective exo memory components
            for cntN = 1:length(exoMemNames)
                exoMemNameTemp = exoMemNames{cntN}
                if str2num(self.fittingSet.selectedComponents.currentSelectedSeq(self.fittingSet.selectedComponents.memExo.(exoMemNameTemp).seqNumber))
                    self.paramSet.limit.memExo((cntN-1)*self.paramSet.size.memExo+1:cntN*self.paramSet.size.memExo,:) = ...
                        self.component.memExo.(curly(fieldnames(self.component.memExo),cntN)).paramLim
                    self.paramSet.limit.sorted(+1:+self.paramSet.size.memExo,:)  = ...
                        self.component.memExo.(curly(fieldnames(self.component.memExo),cntN)).paramLim
                    
                    exoEffeciveNumTemp = exoEffeciveNumTemp + 1
                
            
            
            if exoEffeciveNumTemp>0
                self.paramSet.flag.memExo = [currFlag+1:currFlag+self.paramSet.size.memExo*exoEffeciveNumTemp]
                currFlag = self.paramSet.flag.memExo()
                self.paramSet.effect.memExo = 1
            
            
            
            # Add mean flag
            self.paramSet.flag.mean = [currFlag+1]
            currFlag = self.paramSet.flag.mean
            self.paramSet.effect.mean = 1
            self.paramSet.limit.sorted(+1,:) = self.paramSet.limit.mean
        

                    
    
    # Compute output of the model
    def ret = modelOutput(self, params, dataMask)    
        if nargin < 3
            dataMask = logical(ones(size(self.fittingModelData.output)))'
        
        
        ex              = @(x ,t) x(1).*(exp(-t./x(2)))
        
        anonymTools
        
        # Validate the parameters
        rejectParams = 0
        
        interupt = 0
        
        if size(params,2) ~= size(self.paramSet.limit.sorted,1)
            interupt = 1
        
                    
        for cntP = 1:size(self.paramSet.limit.sorted,1)
            if (params(cntP) < self.paramSet.limit.sorted(cntP,1)) | (params(cntP) > self.paramSet.limit.sorted(cntP,2))
                interupt = 1
            
                               
        
        # Run the model
        if interupt
            totOutputTemp = ones(size(self.fittingModelData.output(dataMask'))) * 10^45
        else
#                 ex              = @(x ,t) x(1).*(exp(-t./x(2))+x(3))
#                 ex_rew          = @(x ,t) x(1).*(exp(-t./x(2)))# + (abs(x(1))>4 | x(2)>(4*20)) * 10^45
#                 ex_choice       = @(x ,t) x(1).*(exp(-t./x(2)))
#                 #     ex_rew          = @(x ,t) x(1).*(exp(-t./x(2)))# + (abs(x(1))>4 | x(2)>(4*20)) * 10^45
#                 ex_l            = @(x ,t) x(1).*(exp(-t./x(2))+x(3))# + (abs(x(1))>4 | x(2)>(4*20)) * 10^45
            
            totInput        =   min(self.totalDataPoint, sum(dataMask))
            
            biasPart        =   0
            exoSignalPart   =   zeros(totInput,1)
            memShortPart    =   zeros(totInput,1)
            memSeasonPart   =   zeros(totInput,1)
            memExoPart      =   zeros(totInput,1)
            ARMeanPart      =   zeros(totInput,1)
            
            # zscoring if needed
            if self.zscoring == 1
                self.zscoreData(dataMask)
            
            
            # Calculating bias component
            if self.paramSet.effect.bias
                biasPart    =   params(self.paramSet.flag.bias)
            
            
            # Calculating exogenous signal(s) component
            if self.paramSet.effect.exoSignal
                exoSignalNames = fieldnames(self.fittingModelData.exoSignal)
                
                for cntExoSignal = 1:length(exoSignalNames)
                    exoSignalNameTemp = exoSignalNames{cntExoSignal}
#                         exoSignalTemp = self.fittingModelData.exoSignal.(exoSignalNameTemp)
                    exoSignalPart(:,cntExoSignal)   =   (self.fittingModelData.snapShot.exoSignal.(exoSignalNameTemp).mask .* self.fittingModelData.snapShot.exoSignal.(exoSignalNameTemp).Signal * params(self.paramSet.flag.exoSignal(cntExoSignal)))
                  
                
                exoSignalPart   =   sum(exoSignalPart,2)
            
            
            # Calculating short memory component
            if self.paramSet.effect.memShort                                       
                self.fittingModelData.snapShot.memShort.Rate(isnan(self.fittingModelData.snapShot.memShort.Rate))             = 0
                memShortPart     =   sum(self.fittingModelData.snapShot.memShort.mask .* self.fittingModelData.snapShot.memShort.Rate .*  repmat(params(self.paramSet.flag.memShort),[totInput,1]),2)
            
            
            # Calculating long memory component
            if self.paramSet.effect.memSeason
                self.fittingModelData.snapShot.memSeason.Rate(isnan(self.fittingModelData.snapShot.memSeason.Rate))             = 0
                memSeasonPart     =   sum(self.fittingModelData.snapShot.memSeason.mask .* self.fittingModelData.snapShot.memSeason.Rate .*  repmat(params(self.paramSet.flag.memSeason),[totInput,1]),2)
            
            
            # Calculating exogenous memory component
            if self.paramSet.effect.memExo
                exoMemNames = fieldnames(self.fittingModelData.memExo)
                cntExoMemTemp = 0
                for cntExoMem = 1:length(exoMemNames)
                    exoMemNameTemp = exoMemNames{cntExoMem}
                    if str2num(self.fittingSet.selectedComponents.currentSelectedSeq(self.fittingSet.selectedComponents.memExo.(exoMemNameTemp).seqNumber))
                        exoMemTemp = self.fittingModelData.snapShot.memExo.(exoMemNameTemp)
                        cntExoMemTemp = cntExoMemTemp + 1
                        
                        memExoPart(:,cntExoMemTemp) = sum(repmat(exoMemTemp.meanData',[1, size(exoMemTemp.Signal,2)]) .*...
                            self.component.memExo.(exoMemNameTemp).memorydef(params(self.paramSet.flag.memExo((cntExoMemTemp-1)*2+1:(cntExoMemTemp-1)*2+2)),...
                            exoMemTemp.Time/1000) .*...
                            exoMemTemp.Signal .* ...
                            exoMemTemp.mask,2)
                        
                        if params(self.paramSet.flag.memExo((cntExoMemTemp-1)*2+2))<0
                            rejectParams = 1
                        
                    
                                                       
                
                memExoPart = sum(memExoPart,2)
                
                ARMeanPart   =   exoMemTemp.meanData' * params(self.paramSet.flag.mean())
                                                          
            
            memShortPart(isnan(memShortPart))       = 0
            memSeasonPart(isnan(memSeasonPart))     = 0
            memExoPart(isnan(memExoPart))           = 0
            
            totOutputTemp = (biasPart + ARMeanPart + exoSignalPart + memShortPart + memSeasonPart + memExoPart)'
            
            if rejectParams
                totOutputTemp = ones(size(totOutputTemp))*10^45
            
            ret = totOutputTemp
            # toc
        

    
    
    # Compute the costdef
    def ret = costdef(self, params, dataMask)
        if nargin < 3
            dataMask = logical(ones(size(self.fittingModelData.output)))'
        
        out_temp = self.modelOutput(params, dataMask)            
        ret = self.fittingModelData.output(dataMask') - out_temp
      
    
    # ZScoreData
    def zscoreData(self, dataMask)
        if nargin < 2
            dataMask = logical(ones(size(self.fittingModelData.output)))'
        
        # Exo Signals
        exoSignalNames = fieldnames(self.fittingModelData.exoSignal)
        
        for cntExoSignal = 1:length(exoSignalNames)
            exoSignalNameTemp = exoSignalNames{cntExoSignal}
            self.fittingModelData.snapShot.exoSignal.(exoSignalNameTemp).Signal = nanzscore(self.fittingModelData.exoSignal.(exoSignalNameTemp).Signal(dataMask))                
            self.fittingModelData.snapShot.exoSignal.(exoSignalNameTemp).mask = self.fittingModelData.exoSignal.(exoSignalNameTemp).mask(dataMask)
        
        
        # Short Memory
        self.fittingModelData.snapShot.memShort.Rate = nanzscore(self.fittingModelData.memShort.Rate(dataMask,:))
        self.fittingModelData.snapShot.memShort.mask = (self.fittingModelData.memShort.mask(dataMask,:))
        self.fittingModelData.snapShot.memShort.Time = (self.fittingModelData.memShort.Time(dataMask,:))
        
        # Seasonal Memory
        self.fittingModelData.snapShot.memSeason.Rate = nanzscore(self.fittingModelData.memSeason.Rate(dataMask,:))
        self.fittingModelData.snapShot.memSeason.mask = (self.fittingModelData.memSeason.mask(dataMask,:))
        self.fittingModelData.snapShot.memSeason.Time = (self.fittingModelData.memSeason.Time(dataMask,:))
        
        # Exogenous Memory
        exoMemNames = fieldnames(self.fittingModelData.memExo)
        
        for cntExoMem = 1:length(exoMemNames)
            exoMemNameTemp = exoMemNames{cntExoMem}
            self.fittingModelData.snapShot.memExo.(exoMemNameTemp).Signal = nanzscore(self.fittingModelData.memExo.(exoMemNameTemp).Signal(dataMask,:))
            self.fittingModelData.snapShot.memExo.(exoMemNameTemp).meanData = nanzscore(self.fittingModelData.memExo.(exoMemNameTemp).meanData(dataMask))
            self.fittingModelData.snapShot.memExo.(exoMemNameTemp).Time = (self.fittingModelData.memExo.(exoMemNameTemp).Time(dataMask,:))
            self.fittingModelData.snapShot.memExo.(exoMemNameTemp).mask = (self.fittingModelData.memExo.(exoMemNameTemp).mask(dataMask,:))
        
    
    
    def saveModelSnapshot(self,noDateTime)
        if nargin<2
            noDateTime = 0
        
        [~,~,~] = mkdir(self.saveAddress)
        fittingResults = self.fittingResults
        if noDateTime
            save([self.saveAddress, self.saveNamePrefix,'_snapShot_Model_[',num2str(self.fittingSet.currentModel),'].mat'],'fittingResults')
        else
            save([self.saveAddress, self.saveNamePrefix,'_snapShot_Model_[',num2str(self.fittingSet.currentModel),']_',datestr(datetime),'.mat'],'fittingResults')
        
    
    
    def saveModel(self,noDateTime)
        if nargin<2
            noDateTime = 0
        
        [~,~,~] = mkdir(self.saveAddress)
        if noDateTime
            save([self.saveAddress, self.saveNamePrefix,'_all.mat'],'self')
        else
            save([self.saveAddress, self.saveNamePrefix,'_all_',datestr(datetime),'.mat'],'self')
        
    
    
    def chooseBestModel(self)
        comparingFactor = 'RSS'
        comparingdef = @max
        comparingSet = []
        for modelCnt = 1:length(self.fittingResults.models)
            comparingSet(+1) = self.fittingResults.models{modelCnt}.res.stat.test{1}.(comparingFactor)
        
        [V I] = comparingdef(comparingSet)
        
        self.fittingResults.bestModel = self.fittingResults.models{I}
        self.fittingResults.comparingSet = comparingSet
        self.fittingResults.bestModelNumber = I
        
        # Convert best model into general format
        self.fittingResults.bestModelGeneralFormat.components = dec2bin(I-1,length(self.fittingSet.selectedComponents.currentSelectedSeq))
        self.fittingResults.bestModelGeneralFormat.components = self.fittingResults.bestModelGeneralFormat.components(:-1:1)
        self.fittingResults.bestModelGeneralFormat.params = nan(1,self.fittingResults.models{}.paramSet.flag.mean)
        self.fittingResults.bestModelGeneralFormat.P_test = nan(1,self.fittingResults.models{}.paramSet.flag.mean)
        
        cellNames = fieldnames(self.fittingResults.bestModel.paramSet.flag)
        for cntCell = [1:4 6]
            cellNameTemp = cellNames{cntCell}
            if ~isempty(self.fittingResults.bestModel.paramSet.flag.(cellNameTemp))
                self.fittingResults.bestModelGeneralFormat.params(self.fittingResults.models{}.paramSet.flag.(cellNameTemp)) =...
                    self.fittingResults.bestModel.res.vars(self.fittingResults.bestModel.paramSet.flag.(cellNameTemp))
                
                self.fittingResults.bestModelGeneralFormat.P_test(self.fittingResults.models{}.paramSet.flag.(cellNameTemp)) =...
                    self.fittingResults.bestModel.res.stat.all.P_test(self.fittingResults.bestModel.paramSet.flag.(cellNameTemp))
            
        
        
        cellNameTemp = 'memExo'
        exoMemNames = fieldnames(self.fittingModelData.memExo)
        exoEffeciveNumTemp = 1 # store the numbers of effective exo memory components
        
        for cntN = 1:length(exoMemNames)
            exoMemNameTemp = exoMemNames{cntN}
            if str2num(self.fittingResults.bestModelGeneralFormat.components(self.fittingSet.selectedComponents.memExo.(exoMemNameTemp).seqNumber))
                memExoTempIdx_Geneal = (cntN-1)*self.paramSet.size.memExo+1:cntN*self.paramSet.size.memExo                    
                
                memExoTempIdx_Special = (exoEffeciveNumTemp-1)*self.paramSet.size.memExo+1:exoEffeciveNumTemp*self.paramSet.size.memExo
                
                self.fittingResults.bestModelGeneralFormat.params(self.fittingResults.models{}.paramSet.flag.(cellNameTemp)(memExoTempIdx_Geneal)) = ...
                    self.fittingResults.bestModel.res.vars(self.fittingResults.bestModel.paramSet.flag.(cellNameTemp)(memExoTempIdx_Special))
                
                self.fittingResults.bestModelGeneralFormat.P_test(self.fittingResults.models{}.paramSet.flag.(cellNameTemp)(memExoTempIdx_Geneal)) = ...
                    self.fittingResults.bestModel.res.stat.all.P_test(self.fittingResults.bestModel.paramSet.flag.(cellNameTemp)(memExoTempIdx_Special))
                
                exoEffeciveNumTemp = exoEffeciveNumTemp + 1
                                            
        
                    
    


methods(Access = public)
    
    def tm = timeMaskBin(self, binWindow)
        #   creates and returns timeMask (tm) selfect based on bin info
        
        tm.type = 1  # 1: bin 2: signal dependent            
        tm.binWindow = binWindow
    
    
    def tm = timeMaskSignal(self, beginSig_Name, beginSig_TimeDist, endSig_Name, endSig_TimeDist)
        #   creates and returns timeMask (tm) selfect based on signals
        
        tm.type = 2  # 1: bin 2: signal dependent            
        tm.beginSig_Name = beginSig_Name
        tm.beginSig_TimeDist = beginSig_TimeDist
        
        tm.endSig_Name = endSig_Name
        tm.endSig_TimeDist = endSig_TimeDist
    
    
    def addMemShort(self, order, PACFTrans, effWin, paramLim)
        #   Initialize data structure for short memory
        #       .order:                 Indicates the memory order
        #
        #       .effectiveTimeWin:      Time window that this memory is
        #                               effective (refer to timeMask def)
        #
        #       .PACF_transfer:         Whether transfer AR Coeffs into
        #                               PACF (Partial AutoCorrelation def) domain (Boolean)  
        #
        #       .paramLim:              Parameter fitting boundries
        
        if nargin < 2
            warning('Not enough arguments!')
            return
        elseif nargin < 3
            effWin = []
            PACFTrans = false
            paramLim = [-inf inf]
        elseif nargin < 4
            effWin = []
            paramLim = [-inf inf]
        elseif nargin < 5                
            paramLim = [-inf inf]
        
        
        proceedFlag = false
        
        if self.component.memShort.defined
            str = input('You already have a short memory component in your model.\n Do you want to rewrite the short memory? y/n [y]: ','s')
            if isempty(str)
                str = 'y'
            
            
            if strcmp(str,'y')
                proceedFlag = true
            
        else
            proceedFlag = true
        
        
        if proceedFlag
            self.component.memShort.defined = true
            self.component.memShort.order = order
            self.component.memShort.effectiveTimeWin = effWin
            self.component.memShort.PACF_transfer = PACFTrans
            self.component.memShort.paramLim = paramLim
        
    
    
    def addMemSeason(self, order, PACFTrans, effWin, paramLim)
        #   Initialize data structure for seasonal memory
        #       .order:                 Indicates the memory order
        #
        #       .effectiveTimeWin:      Time window that this memory is
        #                               effective (refer to timeMask def)
        #
        #       .PACF_transfer:         Whether transfer AR Coeffs into
        #                               PACF (Partial AutoCorrelation def) domain (Boolean)       
        #
        #       .paramLim:              Parameter fitting boundries
        
        if nargin < 2
            warning('Not enough arguments!')
            return
        elseif nargin < 3
            effWin = []
            PACFTrans = false
            paramLim = [-inf inf]
        elseif nargin < 4
            effWin = []
            paramLim = [-inf inf]
        elseif nargin < 5                
            paramLim = [-inf inf]
        
        
        proceedFlag = false
        
        if self.component.memSeason.defined
            str = input('You already have a seasonal memory component in your model.\n Do you want to rewrite the seasonal memory? y/n [y]: ','s')
            if isempty(str)
                str = 'y'
            
            
            if strcmp(str,'y')
                proceedFlag = true
            
        else
            proceedFlag = true
        
        
        if proceedFlag
            self.component.memSeason.defined = true
            self.component.memSeason.order = order
            self.component.memSeason.effectiveTimeWin = effWin
            self.component.memSeason.PACF_transfer = PACFTrans
            self.component.memSeason.paramLim = paramLim
        
    
                    
    def addMemExo(self, name, signalNames, interaction, interactionType, signalTime, expo, expoOrder, memOrder, exoSelEff, effectiveTimeWin, memorydef, paramLim)
        #   Add and initialize data structure for exo memory
        #
        #           .name:                  Name of the exogenous memory
        #
        #           .signalName:        A set contains name(s) of the
        #                               signals involve in this memory       
        #
        #           .interaction:       Whether memory is based on
        #                               interaction of signals
        #
        #           .interactionType:   Type of interactio
        #                               1) simple product
        #                               2) seperate values
        #
        #           .signalTime:        Name of the 'signalTime'
        #                               corresponding for this exo memory                                      
        #
        #           .expo:              Whether memory def is a exponential
        #                               def
        #
        #           .expoOrder:         Order of exponential def
        #
        #           .memOrder:          Indicates the memory order                
        #
        #           .exoSelEff:         Whether consider the effect of exo
        #                               signal (Boolean)
        #                               We should have similar name exo
        #                               signal in "exoSignal" component
        #
        #           .effectiveTimeWin:  Time window that this memory is
        #                               effective (refer to timeMask def)
        # 
        #           .paramLim:          Parameter fitting boundries

        if nargin < 8
            warning('Not enough arguments!')
            return
        elseif nargin < 10
            effectiveTimeWin = []
            exoSelEff = true
            memorydef = @(x ,t) x(1).*(exp(-t./x(2)))
            paramLim = [-inf inf -inf inf]
        elseif nargin < 11
            effectiveTimeWin = []
            memorydef = @(x ,t) x(1).*(exp(-t./x(2)))
            paramLim = [-inf inf -inf inf]
        elseif nargin < 12
            memorydef = @(x ,t) x(1).*(exp(-t./x(2)))
            paramLim = [-inf inf -inf inf]
        elseif nargin < 13
            paramLim = [-inf inf -inf inf]
        
        
        if isempty(memorydef)
            memorydef = @(x ,t) x(1).*(exp(-t./x(2)))
        
        
        proceedFlag = false                       
                    
        if sum(strcmp(fieldnames(self.component.memExo),name)) > 0
            str = input(['You already have a exo memory component with the name of "',name ,'" in your model.\n Do you want to rewrite this exo memory? y/n [y]: '],'s')
            if isempty(str)
                str = 'y'
            
            
            if strcmp(str,'y')
                proceedFlag = true
            
        else
            proceedFlag = true
        
        
        if proceedFlag
            self.component.memExo.(name).signalName = signalNames
            self.component.memExo.(name).interaction = interaction
            self.component.memExo.(name).interactionType = interactionType
            self.component.memExo.(name).signalTime = signalTime
            self.component.memExo.(name).expo = expo
            self.component.memExo.(name).expoOrder = expoOrder
            self.component.memExo.(name).memOrder = memOrder
            self.component.memExo.(name).exoSelEff = exoSelEff
            self.component.memExo.(name).effectiveTimeWin = effectiveTimeWin
            self.component.memExo.(name).memorydef = memorydef
            self.component.memExo.(name).paramLim = paramLim
        
    
                    
    def addExoSignal(self, name, signalNames, interaction, interactionType, effectiveTimeWin, paramLim)
        #   Add and initialize data structure for exo signal
        #       .name:                  Name of the exogenous memory
        #
        #           .signalName:        A set contains name(s) of the
        #                               signals involve in this memory
        #
        #           .interaction:       Whether memory is based on
        #                               interaction of signals
        #
        #           .interactionType:   Type of interactio
        #                               1) simple product
        #                               2) seperate values
        #
        #           .effectiveTimeWin:  Time window that this memory is
        #                               effective (refer to timeMask def)
        #
        #           .paramLim:          Parameter fitting boundries
        
        if nargin < 4
            warning('Not enough arguments!')
            return
        elseif nargin < 5
            interactionType = 2
            effectiveTimeWin = []
            paramLim = [-inf inf]
        elseif nargin < 6
            effectiveTimeWin = []
            paramLim = [-inf inf]
        elseif nargin < 7
            paramLim = [-inf inf]
        
        
        proceedFlag = false                       
                    
        if sum(strcmp(fieldnames(self.component.exoSignal),name)) > 0
            str = input(['You already have a exo signal component with the name of "',name ,'" in your model.\n Do you want to rewrite this exo signal? y/n [y]: '],'s')
            if isempty(str)
                str = 'y'
            
            
            if strcmp(str,'y')
                proceedFlag = true
            
        else
            proceedFlag = true
        
        
        if proceedFlag
            self.component.exoSignal.(name).signalName = signalNames
            self.component.exoSignal.(name).interaction = interaction
            self.component.exoSignal.(name).interactionType = interactionType
            self.component.exoSignal.(name).effectiveTimeWin = effectiveTimeWin
            self.component.exoSignal.(name).paramLim = paramLim
        
    


