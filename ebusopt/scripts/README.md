# scripts
This directory includes scripts used for various stages of analyses, including processing data and running case studies for optimization models.

- `dissertation_case_study.py`: scripts for running case study instances from the final chapter of my dissertation. This includes both optimization models and using the simulation platform to evaluate performance across various scenarios.
- `king_county_charger_location.py`: This script was originally used for the case study in our *Transportation Research Part C* paper. Some functions it calls were revised during my dissertation work and the script would need to be updated to run the same analysis again.
- `scheduling_case_study.py`: This script is for running charging scheduling case studies on the simple notional network and King County Metro network.
- `script_helpers.py`: This module contains helper functions used by various scripts that run test cases, to help with reproducibility. These functions help with processing data and providing inputs to optimization methods in the expected format.
- `sensitivity.py`: Sensitivity analysis from TR Part C paper.