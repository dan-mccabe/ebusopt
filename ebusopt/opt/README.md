# opt
This directory contains optimization code for the `ebusopt` project. The modules are organized as follows:

- `benders_charge_scheduling.py`
  - This module contains functions that implement Combinatorial Benders decomposition, used to solve the charging scheduling problem.
- `charger_location.py`
  - The BEB Optimal Charger Location (BEB-OCL) model implementation, via the `ChargerLocationModel` class.
- `heuristic_charge_scheduling.py`
  - Heuristic algorithms for solving the charging scheduling problem, including our 3S heuristic.
- `simulation.py`
  - This module contains simulation and evaluation code, most importantly a discrete event simulation model, to test the performance of the optimization models across various scenarios.