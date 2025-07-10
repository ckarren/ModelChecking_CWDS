# Model checking water distribution systems files and models
This repo contains all the files necessary to reproduce the methods and findings of the research of "Identifying Cyber-Attack Vulnerabilities of Water Distribution Systems using Model Checking Karrenberg et al. 
## Python files for linearization and calibration
- `linear_regression.py` performs the linear regression of the case study water distribution system, MiniTown. The results of the regression can be directly used in the linearized model. Alternatively, the linear regression can be performed in any other regression tool (such as Excel, R, or Stata), and the values from the regression can be passed into the linear model (through `set_pump_eqns()`).  
- `linear_model.py` contains all the utility and helper functions to perform hyudraulic simulations (using WNTR), linear simulations, and discretized linear simulations. 
- `compare_linear_hydraulic.py` produces plots to visualize the performance of the linear and discretized case study water distribution system compared to a hydraulic simulation using `WNTR` 
- `calc_RMSE.py` finds the number of discrete states of the target water distribution system that minimizes the root mean square error (RMSE) when compared with a hydraulic simulation
## nuXmv model files
Files with the extension `.smv` are infinite state models written in nuXmv. These files can be opened in in the nuXmv model checker available for download from https://nuxmv.fbk.eu/
- `door.smv` is a small example to illustrate the nuXmv language. This file contains an infinite state model of a door
- `tanks.smv` contains the nuXmv infinite state model of water tanks to illustrate the nuXmv language and how cyberphysical water distribution system components can be modelled in nuXmv
- `MiniTownInfinite.smv' is an infinite state model of the MiniTown CWDS and can be used to perform model checking and identify security vulnerabilities using the nuXmv model checker.   
## LTSA model files
Files with the extension `.lts` are state machine models of the MiniTown CWDS written in FSP. These files can be opened in the Labelled Transistion System Analyser (LTSA) which is available for download from https://www.doc.ic.ac.uk/ltsa/
- `minitown7d13t.lts` is the finite state machine model of the MiniTown CWDS
- `minitown_7d13t.lts` is the finite state machine model of the MiniTown CWDS that asserts the safety of the system and ensures that are no attacks are comopromises in the system
- `minitown7d13t_plc1_injection_attack.lts` is the finite state machine model of the MiniTown CWDS with an injection attack targeting PLC 1
- `minitown7d13t_plc2_compromised.lts` is the finite state machine model of the MiniTown CWDS with a compromised PLC 2
- `minitown7d13t_pump_actuator_attack.lts` is the finite state machine model of the MiniTown CWDS with an attack on the pump actuator
- `minitown7d13t_tank_sensor_attack.lts` is the finite state machine model of the MiniTown CWDS with an attack on the tank sensor
## .inp file
`minitown_map.inp` contains the EPANET compatible file of the hydraulic model of the MiniTown CWDS
## Other files
-`minitown_patterns.csv` is a CSV file that contains all the demand patterns used in the hydraulic simulation of the MiniTown CWDS
-`demand.txt` is a TXT file that contains the demand curve for a 168 hour hydraulic simulatino of the MiniTown CWDS
