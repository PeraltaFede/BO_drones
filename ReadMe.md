# Bayesian Optimization ASV system

Bayesian Optimization-based multi Autonomous Surface Vehicle system for Environmental Monitoring.

Consists on a system-based approach for defining surface vehicles in charge of efficiently measuring water quality profiles of lakes and lagoons. This repository corresponds to the code written for works [1] and [2] in Python 3.8.4.

There are different implementations based on our current research objectives. As for april 2021, the following implementations are available and have been aproved for usage in real environments:

* [Single ASV with single water quality parameter acquisition](bin/v2/Examples/example.py) [1].
* [Multiples ASV with single water quality parameter acquisition](bin/v2/Examples/max_example2.py) [2].
* [Single ASV with multiple water quality parameter acquisitions](bin/v2/Examples/gym_example.py) [3].

## Installation

Please install the following python libraries before use:

```commandline
pip install matplotlib scikit-optimize scikit-learn numpy json paho-mqtt pyyaml deap
```

## Use

The code includes an [example folder](bin/v2/Examples) where numerous example of use are found.

## Citing

Please cite the work as [1], [2] or [3].

Any questions fperalta@us.es

## References
[1] F. P. Samaniego, D. G. Reina, S. L. T. Marín, M. Arzamendia and D. O. Gregor, "A Bayesian Optimization Approach for Water Resources Monitoring Through an Autonomous Surface Vehicle: The Ypacarai Lake Case Study," in IEEE Access, vol. 9, pp. 9163-9179, 2021, doi: 10.1109/ACCESS.2021.3050934.

[2] F. Peralta, S. Yanes, D. G. Reina, S. L. T. Marín, "Monitoring Water Resources through a Bayesian Optimization-based Approach using Multiple Surface Vehicles: The Ypacarai Lake Case Study," 2021.

[3] Peralta, F., Reina, D.G., Toral, S., Arzamendia, M. and Gregor, D., 2021. A Bayesian Optimization Approach for Multi-Function Estimation for Environmental Monitoring Using an Autonomous Surface Vehicle: Ypacarai Lake Case Study. Electronics, 10(8), p.963.
