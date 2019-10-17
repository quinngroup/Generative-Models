Directory making discussion:
 - create a directory called exp[n] in models/runs/ where n is new greatest number out of all exp directories
 - title the logging template log.txt within this directory
 - also include the tenserboard integration vlaues within this directory

Template for logging expiriments in log.txt:

Expiriment [Expiriment Name] Log:
=======================================================================================
Date: [Date expiriment organized/performed]
Model Architecture: [Name of model architecture being trained/tested]
Non-default arguments: [Description of how expiriment changes default hyperparamater and configuration values]
[Description of generated file structure]
=======================================================================================
Final loss: [final loss]
Computation Time: [computation time]
Number trainable paramaters: [number trainable parameters (if can find)]
[Analytic description of results, i.e. more nuanced quantitative and qualitiative perspective] 
Conclusion: [Conclusion, or ideas for next expiriment]





Log Example:

Experiment First VTP VAE log
=======================================================================================
Date: 10/16/2019
Model Architecture: vtpvae.py
Non-default arguments: patience = (-1,5,10)
Weights for model with patience ## are stored as firstVtPVAE##.h5
=======================================================================================
Final loss: 7.564
Computation time: 1243.3 seconds
Number trainable paramaters: 60,000
The model seemed to plateau after about the 20th epoch
We should try to run this model again with greater tolerance value
