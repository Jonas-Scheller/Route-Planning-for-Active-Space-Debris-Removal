# Route Planning for Active Space Debris Removal
Algorithms for multi-rendezvous space trajectory optimization in the context of active space removal, based on a travelling salesman problem variant. The following algorithms were developed during a master thesis project, inspired by [a similar approach from ESA](https://esa.github.io/pygmo/examples/example7.html).

<p align="center">
<img src="https://github.com/Tshabalutzi/Route-Planning-for-Active-Space-Debris-Removal/blob/master/images/tour.png" width="60%" />
</p>

## Usage guide
Run `pip install -r requirements.txt` to install the requirements for the project. Furthermore, an installation and license of [gurobi](https://www.gurobi.com/) is required. Examples on how to use the code are given in the jupyter notebook *example.ipynb*.

## Problem Description
With an increasing amount of human-made objects in space, more and more space debris emerges that threatens future missions and the safety of the space around earth. Space agencies are aware of this issue, ESA announced the [world's first mission for active space debris removal](https://www.esa.int/Safety_Security/Clean_Space/ESA_commissions_world_s_first_space_debris_removal) in  2019. Although this mission targets to remove a single debris chunk, there are plans to remove multiple debris pieces in a single missions. In order to find efficient routes, we will transform this problem into two travelling salesman problem variants: The static city-selection travelling salesman problem (**CS-TSP**) and the dynamic city selection travelling salesman problem (**CS-DTSP**). The static variant considers the costs of the transfers between the debris-pieces to be the same regardless of the time of transfer, whereas the dynamic variant allows a change of transfer costs over time, e.g., due to orbital pertubations.

## Algorithms
Most of the proposed algorithms formulate a mixed integer program, that will be solved with the commercial solver gurobi. Since there exists a variety of mixed integer problem formulations, various formulations have been tested for solving the CS-TSP and CS-DTSP. For the CS-TSP, we propose flow-based formulations **F1** and **F3**, as well as different time-based formulations **T1, T2 and seq** (implementation details in *optimization/mips_static.py*). For the dynamic problem formulation CS-DTSP, we first present a way to find a solution built on top of any mixed integer program for the CS-TSP. Since this often results to be inefficient, we delevoped two more flow based formulations for the CS-DTSP called **DF1** and **DF2**, as well as the time-based formulation **DT1** (implementation details in *optimization/mips_dynamic.py*). Finally, a genetic algorithm has been developed in order to solve the CS-DTSP.

## Data
The satellite data can be downloaded from [celestrak](https://celestrak.com/). In order to compute the costs for a maneuver, the data is extracted from the [Two Line Element set](https://celestrak.com/columns/v04n03/). For the approximation of values for a removal in a mission, data from the [satellite catalog (SATCAT](https://celestrak.com/satcat/search.php) is used.
