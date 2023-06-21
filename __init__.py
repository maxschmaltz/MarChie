'''
### Module for computing and storing information of a discrete Markov Chain.
Markov chain is a model of a system with \(n\) states, that assumes that
transition from one state to another is independent of the history of transitions
and is strictly defined by the probability of such transition (Markov assumption).

Thus, a discrete Markov Chain is described with 2 components:

1. Initial probability distribution vector \(π = (π_0, π_1, ..., π_n)\), where \(n\) is the number 
of states in the system, and \(π_i\) is the probability that the system starts in the state \(i\).

2. Transition probability matrix \(P = (p_{ij})\), where \(p_{ij}\) is the probability to go
to the state \(j\) from the state \(i\) in one step.

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
'''

author_mail = 'schmaltzmax@gmail.com'