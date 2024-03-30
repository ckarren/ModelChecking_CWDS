# Model checking water distribution systems files and models
This repo contains all the files necessary to reproduce the methods and findings of the research of "Identifying Cyber-Attack Vulnerabilities of Water Distribution Systems using Model Checking Karrenberg et al. 
## Python files for linearization and calibration
`linear_regression.py` can be used to perform the linear regression of the case study water distribution system, MiniTown. The results of the regression can be directly used in the linearized model. Alternatively, the linear regression can be performed in any other regression tool (such as Excel, R, or Stata), and the values from the regression can be passed into the linear model (through `set_pump_eqns()`).  
`linear_model.py` contains all the utility and helper functions to perform hyudraulic simulations (using WNTR), linear simulations, and discretized linear simulations. 
`compare_linear_hydraulic.py` can be used to visualize the performance of the linear and discretized case study water distribution system compared to a hydraulic simulation using `WNTR`. 
`calc_RMSE.py` can be used to find the number of discrete states of the target water distribution system that minimizes the root mean square error when compared with a hydraulic simulation. 
## nuXmv model files
Files with the extension `.smv` are infinite state models of the MiniTown CWDS written in nuXmv. These files can be opened in in the nuXmv model checker available for download from https://nuxmv.fbk.eu/.   
## LTSA model files
Files with the extension `.lts` are state machine models of the MiniTown CWDS written in FSP. These files can be opened in the Labelled Transistion System Analyser (LTSA) which is available for download from https://www.doc.ic.ac.uk/ltsa/. 
