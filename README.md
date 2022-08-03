<img align="left" src="docs/missingdescent.png"> &nbsp; &nbsp;

[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
                                                               
# Learning High-dimensional Gaussians from Censored Data

 This is an implementation of the following paper:
 
[Arnab Bhattacharyya](https://www.comp.nus.edu.sg/~arnab/), [Constantinos Daskalakis](http://people.csail.mit.edu/costis/), [Themis Gouleakis](https://www.mit.edu/~tgoule/), [Vo Vinh Thanh](https://vothanhvinh.github.io/), [Wang Yuhao](https://yohannawang.com/)

"[Learning High-dimensional Gaussians from Censored Data]()" arXiv preprint arXiv (2022).

## Background
The missingness mechanism are as follows:
1. Missing Completely At Random, value is missing with some probability \alpha;
2. Missing At Random. One fully observed variable lead to the missingness of another variable.
3. Missing Not At Random. Hidden variable(s) lead to the missingness of a fully observed variable. 


MCAR        | MAR
:--------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:
<img width="400" alt="characterization" src="/docs/mcar.png" >  |  <img width="400" alt="characterization" src="/docs/mar.png" >
Self-masking MNAR       | General MNAR
<img width="400" alt="characterization" src="/docs/mnar_self_masking.png" >  |  <img width="400" alt="characterization" src="/docs/mnar_logistic.png" >
''


## Introduction
Assume the censoring model is MNAR, we study two settings
1. [**Self-censoring**]: Assume self-censoring mechanism, we developed a distribution learning algorithm (Algorithm 1 below) tha learns $ N(\mu^*, \Sigma^*)$ up to TV distance $\varepsilon$.
2. [**Convex masking**]: When the missingness mechanisms are in general, we design an efficient mean estimation algorithm from a d-dimensional Gaussian $N{\mu^*, \Sigma}$, assuming that the observed missingness pattern is not very rare conditioned on the values of the observed coordinates, and that any small subset of coordinates is observed with sufficiently high probability.

## Related work
1. [Recent Advances in Algorithmic High-Dimensional Robust Statistics](https://arxiv.org/abs/1911.05911)
2. [Robustly Learning a Gaussian: Getting Optimal Error, Efficiently](https://arxiv.org/pdf/1704.03866.pdf)
3. [Workshop](https://simons.berkeley.edu/workshops/schedule/16121)
https://github.com/YohannaWANG/Missing-Data-Literature

## Prerequisites

- **Python 3.6+**
  - `networkx`
  - `argpase`
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `torch`
  
## Contents

- `data.py` - generate synthetic chain graph data, including graph simulation and data simulation
- `evaluate.py` - algorithm accuracy evaluation.
- `utils.py` - simulation parameters, such as selecte graph type, node number, data type, graph degree, etc.  
- `utils.R` - prune, regression, etc.
- `main.py` - main algorihtm.

## Parameters

| Parameter    | Type | Description                      | Options            |
| -------------|------| ---------------------------------|  :----------:      |
| `n`          |  int |  number of nvariables            |      -             |
| `a`          |  int |  average node degree             |      -             |
| `d`          |  int |  number of samples               |      -             |
| `plot`       |  Bool |  plot chain graph or not        |  -                 |
| `algorithm`  |  str |  choice which algorithm          |   `self-censoring`, `convex-masking` |


## Running a simple demo

The simplest way to try out MissingDescent is to run a simple example:
```bash
$ git clone https://github.com/YohannaWANG/MissingDescent.git
$ cd MissingDescent/
$ python $ cd MissingDescent/main.py
```

## Runing as a command

Alternatively, if you have a CSV data file `X.csv`, you can install the package and run the algorithm as a command:
```bash
$ pip install git+git://github.com/YohannaWANG/MissingDescent
$ cd MissingDescent
$ python main.py --algorithm self-censoring --n 50 --s 1000 --d 4 
```

## Algorithms

- ![](https://via.placeholder.com/15/f03c15/000000?text=+)  **Algorithm 1**  [Truncation_PSGD] Distribution recovery given access to an
oracle that generates samples with incomplete data;
   <img width="800" align ="center" alt="characterization" src="/docs/algo1.PNG" >
- ![](https://via.placeholder.com/15/c5f015/000000?text=+)  **Algorithm 2**  [MissingDescent] Mean recovery given access to an oracle that generates
samples with incomplete data. 
   <img width="800" align ="center" alt="characterization" src="/docs/algo2.PNG">    
- ![](https://via.placeholder.com/15/c5f015/000000?text=+)  **Algorithm 3**  [Initialize] Initialization for the main algorithm.
   <img width="800" align ="center" alt="characterization" src="/docs/algo3.PNG">    
- ![](https://via.placeholder.com/15/c5f015/000000?text=+)  **Algorithm 4**  [SampleGradient] Sampler for $\nabla \ell(\bm{\mu})$. 
   <img width="800" align ="center" alt="characterization" src="/docs/algo4.PNG">    
- ![](https://via.placeholder.com/15/c5f015/000000?text=+)  **Algorithm 5**  [ProjectToDomain] The function that projects a current guess back to the domain 
onto the $\ball_{\bm{\Sigma}}$ ball.
   <img width="800" align ="center" alt="characterization" src="/docs/algo5.PNG">    
   
## Performance

[Truncation_PSGD] (Mean absolute percentage error (MAPE) and KL divergence)   
<img width="800" alt="characterization" src="/docs/errors-over-N.jpg" > 
:--------------------------------------------------------------------:

[Truncation_PSGD] We fixed N=20,000 and varied the percentage of missing from 10% to 80%.
<img width="800" alt="characterization" src="/docs/errors-over-missing.jpg" > 

:--------------------------------------------------------------------:

[Truncation_PSGD] Running time on synthetic data.
<img width="700" alt="characterization" src="/docs/running-time-over-missing.jpg" > 


[Truncation_PSGD] Semi-synthetic dataset.
<img width="800" alt="characterization" src="/docs/errors-Turbine.jpg" > 


## Related Works
 One paragraph in our related work section gives almost a complete history of work done on them! We summarized most of the related works below, it will also be updated accordingly. 
 
 [https://github.com/YohannaWANG/Missing-Data-Literature](https://github.com/YohannaWANG/Missing-Data-Literature)

## Citation


## Contacts

[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/YohannaWANG/DCOV/MisisngDescent)
Please feel free to contact us if you meet any problem when using this code. We are glad to hear other advise and update our work. 
We are also open to collaboration if you think that you are working on a problem that we might be interested in it.
Please do not hestitate to contact us!



