class ModelComponent: 
    # Contain all the input information we want to create the model
    # Components contains all 3 possible memory types and exogenous
        # selectivity, each have their on data structure. 
        #
        #   .memShort (Struct for short memory)
        #       .defined:               Whether memory is defined or not
        #                               (Boolean)
        #
        #       .order:                 Indicates the memory order
        #
        #       .effectiveTimeWin:      Time window that this memory is
        #                               effective (refer to timeMask function)
        #
        #       .PACF_transfer:         Whether transfer AR Coeffs into
        #                               PACF (Partial AutoCorrelation Function) domain (Boolean)
        #
        #       .varIdx:                Index of variables in total vriable
        #                               vector
        #
        #       .Data:                  Data stored for this memory
        #               .allARCoeff:    All real values of AR Coeffs
        #               .allPAACFCoeff: All real values of PACF Coeffs
        #               .allTau:        All real values of converted taus
        #               .meanTau:       Mean of converted taus
        #               .maxTau:        Max of converted taus
        #
        #                   .p:         Each field has p-val
        #                   .val:       Each field has real value
        #                   .stat:      Each field has stat (t-val, etc.)
        #
        #        
        #
        #   .memSeason (Struct for seasonal memory)
        #       .defined:               Whether memory is defined or not
        #                               (Boolean)
        #
        #       .order:                 Indicates the memory order
        #
        #       .effectiveTimeWin:      Time window that this memory is
        #                               effective (refer to timeMask function)
        #
        #       .PACF_transfer:         Whether transfer AR Coeffs into
        #                               PACF (Partial AutoCorrelation Function) domain (Boolean)
        #
        #       .varIdx:                Index of variables in total vriable
        #                               vector
        #
        #       .Data:                  Data stored for this memory
        #               .allARCoeff:    All real values of AR Coeffs
        #               .allPAACFCoeff: All real values of PACF Coeffs
        #               .allTau:        All real values of converted taus
        #               .meanTau:       Mean of converted taus
        #               .maxTau:        Max of converted taus
        #
        #                   .p:         Each field has p-val
        #                   .val:       Each field has real value
        #                   .stat:      Each field has stat (t-val, etc.)
        #
        #
        #
        #   .memExo (Struct(s) for Exogenous memory)
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
        #           .signalTime:        Name of the 'signalTime'
        #                               corresponding for this exo memory                                      
        #
        #           .memFunc:           Type of memory function
        #               .expo:          Whether it is a exponential
        #                               function
        #               .expoOrder:     Order of exponential function
        #
        #           .memOrder:          Indicates the memory order
        #
        #           .effectiveTimeWin:  Time window that this memory is
        #                               effective (refer to timeMask function)
        #
        #           .exoSelEff:         Whether consider the effect of exo
        #                               signal (Boolean)
        #                               We should have similar name exo
        #                               signal in "exoSignal" component
        #
        #
        #
        #
        #   .exoSignal (Struct(s) for Exogenous signals)
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
        #                               effective (refer to timeMask function)
        #

    def __init__(self):
        # Initializing components
        
        # Short Memory
        component.memShort.defined = false;
        component.memShort.order = 0;
        component.memShort.effectiveTimeWin = nan;
        component.memShort.PACF_transfer = false;
        component.memShort.varIdx = [];
        
        lblTemp = {'allARCoeff','allPAACFCoeff','allTau','meanTau','maxTau'};
        for cntLBL = 1:length(lblTemp)
            lblCurr =  lblTemp{cntLBL};
            component.memShort.Data.(lblCurr).p = [];
            component.memShort.Data.(lblCurr).val = [];
            component.memShort.Data.(lblCurr).stat = [];
        end
        
        # Seasonal Memory
        component.memSeason.defined = false;
        component.memSeason.order = 0;
        component.memSeason.effectiveTimeWin = nan;
        component.memSeason.PACF_transfer = false;
        component.memSeason.varIdx = [];
        
        lblTemp = {'allARCoeff','allPAACFCoeff','allTau','meanTau','maxTau'};
        for cntLBL = 1:length(lblTemp)
            lblCurr =  lblTemp{cntLBL};
            component.memSeason.Data.(lblCurr).p = [];
            component.memSeason.Data.(lblCurr).val = [];
            component.memSeason.Data.(lblCurr).stat = [];
        end
                    
        # Exo Memory 
        # #################################################
        # ############ Need Update ########################
        # #################################################
        component.memExo = struct;   
        
        # Exo Signal
        # #################################################
        # ############ Need Update ########################
        # ################################################# 
        component.exoSignal = struct; 
        
        obj.component = component;
        pass