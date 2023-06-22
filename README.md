# MarChie: a Compact Open Source Tool for Analyzing Discrete **Mar**kov **Ch**ains

[![Generic badge](https://img.shields.io/badge/PyPI-0.1-green.svg)](https://pypi.org/project/MarChie/)
[![Generic badge](https://img.shields.io/badge/GitHub-Source-red.svg)](https://github.com/maxschmaltz/MarChie)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)


----------

### Contents:

* [Intro](#intro)
* [Quick Start](#quick-start)
* [Acknowledgements](#acknowledgements)
* [License](#license)

----------


## Intro

Markov Chain is a model of a system with $N$ states, that assumes that
transition from one state to another is independent of the history of transitions
and is strictly defined by the probability of such transition (Markov assumption).

At the beginning moment of time (step $0$), the probabilities of the states are defined by the initial states distribution vector.
On each next time step $k$, the system goes from a state $\xi_k = i$ to a state $\xi_{k + 1} = j$ with a probability $P(\xi_{k + 1} = j| \xi_k = i) \equiv p_{ij}$, while the history of the transitions does not change this probability (Markov assumption):

$$
P(\xi_{k + 1} =j| \xi_0 = i_0, \xi_1 = i_1, ...,  \xi_{k - 1} = i_{i - 1}, \xi_k = i) = P(\xi_{k + 1} =j| \xi_k = i) \equiv p_{ij}
$$

Thus, a discrete Markov Chain is described with 2 components:

1. Initial probability distribution vector $\pi = (\pi_0, \pi_1, ..., \pi_n)$, where $n$ is the number of states in the system, and $\pi_i$ is the probability that the system starts in the state $\xi_0 = i$.

2. Transition probability matrix $P = (p_{ij})$, where $p_{ij}$ is the probability to go to the state $\xi_{k + 1} = j$ from the state $\xi_k = i$ in one step.


The vector of an initial states distribution, as well as the rows of a transition matrix, are stochastic (the probabilities should add up to 1) as the events of transitions are mutually exclusive, while the system must make a transition at each step, so the sum of the probabilities of all the transitions must be 1 altogether.

From Markov assumption it follows:

$$
\forall n \geq 1, \forall i_k: \quad
P(\xi_0 = i_0, \xi_1 = i_1, ..., \xi_n = i_n) = \pi_i^{(0)} p_{i_{0} i_{1}} p_{i_{1} i_{2}} ... p_{i_{n - 1} i_{n}}
$$

<br>

Given transition probability matrix and (optional) initial state probability distribution,
`MarChie` does the following:

* computes adjacency matrix;
* computes reachability matrix (using adjacency matrix);
* computes transposed reachability matrix (using reachability);
* computes communication matrix (using reachability and transposed reachability matrices);
* computes communication matrix complement (using communication);
* computes classification matrix (using reachability and communication matrix complement);
* computes classification matrix extension (using classification matrix);
* computes equivalency classes matrix (using communication matrix and classification matrix extension);
* defines essential and inessential states;
* defines equivalency classes and their cyclic subclasses;
* builds the structure of the chain from where all its states, classes and subclasses are easily accessible;
* defines properties of the chain;
* classifies the chain (based on the properties);
* defines end behavior of the chain (based on the classification).


## Quick Start

`March` is a `pip`-installable package. You can access it directly from PyPI:

```bash
pip install MarChie
```

The main object that is really intended to be used is `class MarChie.marchie.March`. The class requires only transition probability matrix and (optionally) initial state probability distribution vector as arguments; if you provide no initial distribution vector, it will be generated.

```python

>>> import numpy as np
>>> from March.marchie import March

>>> trans_mat = np.array([
    [1,     0,     0  ],
    [0.8,   0.2,   0  ],
    [0.3,   0.5,   0.2]
])
>>> init_distr = np.array(
    [0.3,   0.2,   0.5]    
)
>>> marchie = MarChie(
    init_distr=init_distr,
    trans_mat=trans_mat
)

>>> marchie
```

Output:
```text
Monoergodic Absorbing Markov Chain
|
|___Essential States
    |
    |___Acyclic Equivalency Class 0 : states 0
|
|___Inessential States: states 1, 2

[+reducible][-polyergodic][+regular][+absorbing][+strong_convergence]
```

All the information is accessible in class / instance variables, which you will learn in details in API_reference.html. Also, make sure to take a look at the demo notebook for relevant examples.


## Acknowledgements

I would like to kindly thank Elena Ilyushina (MSU, Faculty of Mechanics and Mathematics) for the course "Markov Chains and their linguistic applications" (2021). Even though at the moment, the course was far above my educational stage, I was kindly accepted to take it and, with Elena Ilyushina's help, I was able to understand the basic concepts of Markov Chains Theory thoroughly and solidly, learn to apply the Theory in real-life tasks and found my programming skills useful for efficient Markov Chains analysis.


## License

> Copyright 2023 Max Schmaltz: @maxschmaltz
> 
> Licensed under the Apache License, Version 2.0 (the "License"); <br>
> you may not use this file except in compliance with the License. <br>
> You may obtain a copy of the License at <br>
> 
>    http://www.apache.org/licenses/LICENSE-2.0 <br>
> 
> Unless required by applicable law or agreed to in writing, software <br>
> distributed under the License is distributed on an "AS IS" BASIS, <br>
> WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. <br>
> See the License for the specific language governing permissions and <br>
> limitations under the License.
