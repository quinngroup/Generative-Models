Directory making discussion:
 - create a directory called experimentName in models/runs/ where experimentName is an informative abbreviation for the name of the experiment and the date in which it was performed
    For example, models/runs/firstExperiment-10-17-19
 - title the logging template log.txt within this directory
 - also include the tensorboard integration values within this directory

Template for logging experiments in log.txt:

[Experiment Name] Experiment Log:
=======================================================================================
Date: [Date experiment organized/performed]
Model Architecture: [Name of model architecture being trained/tested]
Non-default arguments: [Description of how experiment changes default hyperparamater and configuration values]
[Description of generated file structure]
=======================================================================================
Final loss: [final loss]
Computation Time: [computation time]
Number trainable paramaters: [number trainable parameters (if can find)]
[Analytic description of results, i.e. more nuanced quantitative and qualitiative perspective] 
Conclusion: [Conclusion, or ideas for next experiment]





Log Example:

First VtPVAE Experiment Log
=======================================================================================
Date: 10/16/2019
Model Architecture: VtPVAE.py
Non-default arguments: patience = (-1,5,10)
Weights for model with patience ## are stored as firstVtPVAE##.h5
=======================================================================================
Final loss: 7.564
Computation time: 1243.3 seconds
Number trainable paramaters: 60,000
The model seemed to plateau after about the 20th epoch
We should try to run this model again with greater tolerance value
