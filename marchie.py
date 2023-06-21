# Copyright 2023 Max Schmaltz: @maxschmaltz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ************************************************************************


#region Imports

import numpy as np
import graphviz
from typing import Tuple, Dict

from structure import *
from end_behavior import *

#endregion

__pdoc__ = {
    'March._matrix': True
}


class MarChie:
    
    '''
    ### Class for computing and storing information of a discrete Markov Chain.
    Markov chain is a model of a system with \(n\) states, that assumes that
    transition from one state to another is independent of the history of transitions
    and is strictly defined by the probability of such transition (Markov assumption).

    Thus, a discrete Markov Chain is described with 2 components:

    1. Initial probability distribution vector \(\pi = (\pi_0, \pi_1, ..., \pi_n)\), where \(n\) is the number 
    of states in the system, and \(\pi_i\) is the probability that the system starts in the state \(i\).

    2. Transition probability matrix \(P = (p_{ij})\) where \(p_{ij}\) is the probability to go
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

    Parameters
    ----------
    init_distr : `numpy.ndarray` of shape (`n_states`, ), optional, defaults to `None`
        initial probability distribution vector; if not provided, generated automatically

    trans_mat : `numpy.ndarray` of shape (`n_states`, `n_states`)
        transition probabilities matrix

    Raises
    ------
    AssertionError
        if chain components are not valid

    Notes
    -----
    The vector of initial state distribution should add up to \(1\) as the system 
    must begin at some state anyway, so the sum of the probabilities of all the states must be \(1\);
    otherwise the vector is invalid.

    The rows of the transition matrix should add up to \(1\) as from each step, the system must be able to go
    to some state anyway, so the sum of the probabilities of the transitions will be \(1\);
    otherwise the matrix is invalid.
    '''

    class _matrix(np.ndarray):

        '''
        ### Wrapper class for Markov Chain matrices.
        Inherits from `numpy.ndarray`; the main feature is implementing
        `canonical` property that rebuilds the input matrix
        in accordance with canonical numbering.

        Parameters
        ----------
        input_matrix : `numpy.ndarray` of shape `input_shape`
            input matrix to wrap around

        canonical_mapping: `Dict[int, int]`
            mapping between state and its canonical number

        Returns
        -------
        obj : `_matrix` of shape `input_shape`
            wrapped matrix
        
        Notes
        -----
        In general, class adds 3 properties:

        canonical_mapping: `Dict[int, int]`
            mapping between state and its canonical number
        
        dim: `int`
            dimentionality of matrix

        canonical: `numpy.ndarray` of shape `input_shape`
            input matrix where the states renumbered canonically
        '''

        def __new__(cls, input_matrix: np.ndarray, canonical_mapping: Dict[int, int]) -> np.ndarray:
            obj = np.asarray(input_matrix).view(cls)
            obj.canonical_mapping = canonical_mapping
            obj.original = input_matrix
            obj.dim = len(obj.shape)
            return obj

        @property
        def canonical(self) -> np.ndarray:

            '''
            ### Rebuilds the input matrix in accordance with canonical numbering.

            Returns
            -------
            can_matrix: `numpy.ndarray` of shape `input_shape`
                input matrix where the states renumbered canonically
            '''

            can_matrix = self.copy()
            for i, row in enumerate(self):
                if self.dim == 1: # for state distribution
                    i_ = self.canonical_mapping[i]
                    can_matrix[i_] = row
                else: # for 2D matrices
                    for j, value in enumerate(row):
                        i_ = self.canonical_mapping.get(i)
                        j_ = self.canonical_mapping.get(j)
                        if i_ is None or j_ is None: continue # for extension
                        can_matrix[i_, j_] = value
            if self.dim == 1: can_matrix = can_matrix.squeeze()
            return can_matrix

    init_distr: _matrix
    '''
    ### Initial state distribution vector.
    Vector \(\pi = (\pi_0, \pi_1, ..., \pi_n)\), where \(n\) is the number 
    of states in the system, and \(\pi_i\) is the probability that the system starts in the state \(i\).

    Returns
    -------
    init_distr : `March._matrix` of shape (`n_states`, )
        distribution vector
    '''

    trans_mat: _matrix
    '''
    ### Transition probability matrix
    Matrix \(P = (p_{ij})\) where \(p_{ij}\) is the probability to go
    to the state \(j\) from the state \(i\) in one step.

    Returns
    -------
    trans_mat : `March._matrix` of shape (`n_states`, `n_states`)
        transition matrix
    '''

    adjacency_mat: _matrix
    '''
    ### Adjacency matrix of the chain.
    Adjacency matrix is a matrix that indicates existence of one-step paths between states,
    where a one-step path from the state \(i\) to the state \(j\) exists 
    if there a non-zero probability transition between them.

    Returns
    -------
    adjacency_mat : `March._matrix` of shape (`n_states`, `n_states`)
        boolean-like matrix indicating one-step paths between states,
        where `adjacency_mat[i, j]` is \(1\) if the probability of tranition 
        from the state `i` to the state `j` is not zero, else \(0\)

    Notes
    -----
    Adjacency matrix is calculated as shown below:

    $$
    A = (a_{ij}) | a_{ij} =
    \\begin{cases}
        1, & p_{ij} > 0     \\\\
        0, & p_{ij} = 0
    \\end{cases}
    $$

    where \(A\) is adjacency matrix, \((p_{ij})\) is transition matrix.
    '''

    reachability_mat: _matrix
    '''
    ### Reachability matrix of the chain.
    Reachability matrix is a matrix that indicates existence of paths between states,
    where a path from the state \(i\) to the state \(j\) exists if there a sequence 
    of non-zero probability transitions between states such that it leads from \(i\) to \(j\)
    (e.g. \(i\) → \(h\) → \(k\) → ... → \(j\)).

    Returns
    -------
    reachability_mat : `March._matrix` of shape (`n_states`, `n_states`)
        boolean-like matrix indicating paths between states,
        where `reachability_mat[i, j]` is \(1\) if there is a path 
        from the state `i` to the state `j`, else \(0\)

    Notes
    -----
    Reachability matrix is calculated as shown below:

    $$
    D = (d_{ij}) | d_{ij} =
    \\begin{cases}
        1, & \\exists i → j        \\\\
        0, & \\neg \exists i → j
    \\end{cases}
    $$

    where \(D\) is reachability matrix, \(→\) marks a path.
    '''

    reachability_mat_tr: _matrix
    '''
    ### Transposed reachability matrix of the chain.
    Reachability matrix is a matrix that indicates existence of inverse paths between states,
    where a path from the state \(i\) to the state \(j\) exists if there a sequence 
    of non-zero probability transitions between states such that it leads from \(i\) to \(j\)
    (e.g. \(i\) → \(h\) → \(k\) → ... → \(j\)).

    Returns
    -------
    reachability_mat_tr : `March._matrix` of shape (`n_states`, `n_states`)
        boolean-like matrix indicating paths between states,
        where `reachability_mat_tr[i, j]` is \(1\) if there is a path 
        from the state `j` to the state `i`, else \(0\)

    Notes
    -----
    Transposed reachability matrix is calculated as shown below:

    $$
    D^T = (d^T_{ij}) | d^T_{ij} =
    \\begin{cases}
        1, & \\exists j → i        \\\\
        0, & \\neg \exists j → i
    \\end{cases}
    $$

    where \(D^T\) is transposed reachability matrix, \(→\) marks a path.
    '''

    communication_mat: _matrix
    '''
    ### Communication matrix of the chain.
    Communication matrix is a matrix that indicates existence of bidirectional paths between states,
    where a bidirectional path between the states \(i\) and \(j\) exists if there a sequence 
    of non-zero probability transitions from \(i\) to \(j\) (e.g. \(i\) → \(h\) → \(k\) → ... → \(j\))
    and from \(j\) to \(i\) (not exactly the same way back).

    Returns
    -------
    communication_mat : `March._matrix` of shape (`n_states`, `n_states`)
        boolean-like matrix indicating bidirectional paths between states,
        where `communication_mat[i, j]` is \(1\) if there is a path 
        from the state `j` to the state `i` and from `j` to `i`, else \(0\)

    Notes
    -----
    Communication matrix is calculated as shown below:

    $$
    C = (c_{ij}) = D \\times D^T | c_{ij} = 
    \\begin{cases}
        1, & d_{ij} = d_{ji} = 1        \\\\
        0, & \\neg (d_{ij} = d_{ji} = 1)
    \\end{cases}
    $$

    where \(C\) is communication matrix, \(D\) is reachability matrix,
    \(D^T\) is transposed reachability matrix, \(→\) marks a path.
    '''

    communication_mat_comp: _matrix
    '''
    ### Communication matrix complement. 
    Communication matrix complement is a matrix that indicates nonexistence of bidirectional paths between states,
    where a bidirectional path between the states \(i\) and \(j\) exists if there a sequence 
    of non-zero probability transitions from \(i\) to \(j\) (e.g. \(i\) → \(h\) → \(k\) → ... → \(j\))
    and from \(j\) to \(i\) (not exactly the same way back).

    Returns
    -------
    communication_mat_comp : `March._matrix` of shape (`n_states`, `n_states`)
        boolean-like matrix indicating absence of bidirectional paths between states,
        where `communication_mat[i, j]` is \(1\) if there is no path 
        from the state `j` to the state `i` and from `j` to `i`, else 0

    Notes
    -----
    Communication matrix complement is calculated as follows:

    $$
    \\overline C = (\\overline c_{ij}) | \\overline c_{ij} = 
    \\begin{cases}
        1, & c_{ij} = 0     \\\\
        0, & c_{ij} = 1
    \\end{cases}
    $$

    where \(\\overline C\) is communication matrix complement, \(c_{ij}\) is communication matrix.
    '''

    classification_mat: _matrix
    '''
    ### Classification matrix of the chain.
    Classification matrix is a step for building extended classification matrix.

    Returns
    -------
    classification_mat : `March._matrix` of shape (`n_states`, `n_states`)
        boolean-like matrix where `classification_mat[i, j]` is \(1\) if there is a path 
        from the state `i` to the state `j` but states `j` and `i` do not communicate, 0 otherwise

    Notes
    -----
    Classification matrix is calculated as follows:

    $$
    T = D \\times \\overline C = (t_{ij}) | t_{ij} = 
    \\begin{cases}
        1, & d_{ij} = \\overline c_{ij} = 1      \\\\
        0, & \\neg (d_{ij} = \\overline c_{ij} = 1)
    \\end{cases}
    $$

    where \(T\) is classification matrix, \(d_{ij}\) is reachability matrix, 
    \((\\overline c_{ij})\) is communication matrix complement.
    '''

    classification_mat_ext: _matrix
    '''
    ### Classification matrix extension. 
    Classification matrix extension indicates essential and inessential states.
    Essential states are the ones that communicate with each (!) state they have a path to;
    respectively, the ineccential are the ones for which this statement is not true.

    Returns
    -------
    classification_mat_ext : `March._matrix` of shape (`n_states`, `n_states` + 1)
        boolean-like matrix where the last column indicates essentiality of states:
        if classification_mat_ext[`i`, `n_states` + 1] equals 1, the state `i`
        is inessential, if 0, is essential

    Notes
    -----
    Classification matrix extention is defined below:

    $$
    T_{ext} = T|(t_{i(N + 1)}) | t_{i(N + 1)} = 
    \\begin{cases}
        1, & \\exists j t_{ij} = 1        \\\\
        0, & \\neg \\exists j t_{ij} = 1
    \\end{cases}
    $$

    where \(T_{ext}\) is classification matrix extension, \(T\) is classification matrix,
    \(N\) is the number of states in the system.
    '''

    equivalency_cls_mat: _matrix
    '''
    ### Equivalency matrix of the chain.

    Returns
    -------
    equivalency_cls_mat : `March._matrix` of shape (`n_classes`, `n_essential`)
        matrix showing belonging of each essential (!) state to its equivalence class;
        `equivalency_cls_mat[i, j]` indicates that the state `j` is in `i`th equivalence class

    Notes
    -----
    Matrix of equivalency classes is defined as follows:

    $$
    K = (k_{ij}) | k_{it} \\Longleftrightarrow i \\in EC_t
    $$

    where \(K\) is equivalency matrix.
    '''

    structure: ChainStructure
    '''
    ### Structure of the chain.
    That includes:

    * Essential and inessential states:
    Essential state is such a state \(i\) that communicates with every (!) state \(j\)
    it has a path to; respectively, a state is inessential if that is not true.

    * Cyclic subclasses of equivalence classes:
    Each equivalence classes has a period \(d\): it equals the GCD of the lengths
    of all the contours going through a state \(i\) belonging to the equivalence class.
    Contour is a cyclic path from a state to itself (e.g. \(i\) → \(h\) → \(k\) → ... → \(i\)), 
    its length is the number of the transitions within it.
    It is proven that all the states of an equivalence class have the same period \(d\).
    If \(d = 1\), the equivalence class is acyclic.
    If \(d \geq 2\), it can be split into \(d\) cyclic subclasses. Cyclic subclass \(C_r\) of equivalence class \(EC_n\)
    is a subset of states of \(EC_n\) such that 
    
    * the states are not adjacent (i.e. there is no one-step path between them);
    * from any of the states there is a one-step path only to a state of the next cyclic subclass \(C_{r + 1}\);
    * the states belong to an only cyclic subclass;
    * the equivalence class is fully distributed between cyclic subclasses (no states left outside of the subclasses).

    That means the cyclic subclasses are closed and form a cycle themselves:

    \(C_0 → C_1 → C_2 → ... → C_{d - 1} → C_0\)

    * Canonical numbers of the states in the system structure:
    Canonical numbering is a numbering that goes in strict order through each state in each cyclic subclass
    of each equivalency class, followed only then by inessential states. That allows to transform the transition matrix
    into a block matrix that represents probabilities of transition between cyclic subclasses.

    Returns
    -------
    structure : `ChainStructure`
        class containing attributes for chain structure

    Notes
    -----
    Essentiality of states if defined as follows:
    $$
    i \\in S_e \\iff 
    \\forall j: \\quad i → j \\Longrightarrow j → i
    $$
    $$
    i \\in S_{ie} \\iff 
    \\neg \\forall j: \\quad i → j \\Longrightarrow j → i
    $$

    Where \(S_e\) is the set of essential states, \(S_{ie}\) is the set of inessential states.

    Formally, the period of an equivalency class is found as:

    $$
    d = \\text{GCD}(|(i → i)_1|, ..., |(i → i)_r|)
    $$
        
    where \((i → i)\) is a contour, \(|(i → i)_1|\) is a length of a contour

    If \(d\) equials 1, the equivalence class is acyclic, which means it has no cyclic subclasses.

    Canonical numbering goes as follows:

    \(EC_1\):

    \(C_0: \\quad 1, \\quad 2, \\quad 3, \\quad ..., \\quad r_0\)

    \(C_1: \\quad r_0 + 1, \\quad r_0 + 2, \\quad ..., \\quad r_1\)

    \(...\)

    \(C_{d - 1}: r_0 + r_1 + ... + r_{d - 1} + 1, \\quad r_0 + r_1 + ... + r_{d - 1} + 2, 
    \\quad ..., \\quad r_0 + r_1 + ... + r_{d - 1} + r_{d} = N_{EC_1}\)

    \(EC_2\): (same way)

    \(...\)

    \(EC_L\): (same way)

    \(S_{ie}: \\quad N_e + 1, \\quad N_e + 2, \\quad ..., \\quad N_e + N_{ie} = N\)

    where \(EC_k\) is equivalence class \(k\), \(C_s\) is a cyclic subclass \(s\) of the equivalence class,
    \(N_{EC_k}\) is number of states in equivalence class \(k\), \(N_e\) is number of essential states in the system,
    \(N\) is number of states in the system.
    '''

    properties: ChainProperties
    '''
    Markov Chain classification:

    1. \(S\) = \(Se\) = \(EC\) ⇒ ergodic / irreducible Markov Chain

        1.1. \(EC\) is regular ⇒ regular irreducible Markov Chain

        \(\quad\) 1.1.1. \(EC\) is absorbing ⇒ absorbing irreducible Markov Chain

        1.2. \(EC\) is cyclic ⇒ cyclic irreducible Markov Chain

    2. \(Sie ≠ ∅\) ⇒ reducible Markov Chain

        2.1. a single \(EC\) ⇒ monoergodic Markov Chain

        \(\quad\)2.1.1. the single \(EC\) is regular ⇒ regular monoergodic Markov Chain

        \(\qquad\) 2.1.1.1. the single \(EC\) is absorbing ⇒ absorbing monoergodic Markov Chain

        \(\quad\) 2.1.2. the single \(EC\) is cyclic ⇒ cyclic monoergodic Markov Chain

        2.2. more than one \(EC\)s ⇒ polyergodic Markov Chain

        \(\quad\) 2.2.1. all the \(EC\)s are regular ⇒ regular polyergodic Markov Chain

        \(\qquad\) 2.2.1.1. all the \(EC\)s are absorbing ⇒ absorbing polyergodic Markov Chain

        \(\quad\) 2.2.2. all the \(EC\)s are cyclic ⇒ cyclic polyergodic Markov Chain

        \(\quad\) 2.2.3. there are both regular and cyclic \(EC\)s ⇒ mixed polyergodic Markov Chain,

    where \(S\) is set of states, \(Se\) is set of essential states, \(Sie\) is set of inessential states,
    \(EC\) stands for equivalence class.

    Returns
    -------
    structure : `ChainProperties`
        class containing attributes for chain properties

    Notes
    -----
    Note that in our classification we won't just return a single label of the class, but rather
    use a more "linguistic" notation: we will define a set of properties, each of which can be 
    either `True`, or `False`, or `None` (if not applicable). Thus, we will be able to 
    split the space of classes uniquely, yet be able to access chain features in a more convenient way.

    Reminder: a \(EC\) is regular if it has a period \(d\) = 1, i.e. has no cyclic subclasses, otherwise cyclic.
    Reminder: a \(EC\) is absorbing if it contains a single state.
    
    The properties are:

    * `reducible` (if there are inessential states)
    * `polyergodic` (if there are more than one `EC`)
    * `regular` (cyclic if `False`, mixed if `None`)
    * `absorbing` (may be `True` only if regular)
    * `strong_convergence` (if the matrix strongly converges to a final matrix in a great number of steps)

    The matrix converges strongly if it's regular, weakly otherwise.

    Thus, we'll have a "linguistic-like" notation, e.g. a cyclic polyergodic Markov Chain
    could be notated as ```[+reducible][+polyergodic][-regular][∅absorbing][-strong_convergence]```.
    '''

    end_behavior: EndBehavior
    '''
    End behavior of a Markov chain can be described with 3 properties:

    * end state distribution vector: what are the probabilities of
    the system to be in each state in a very large (limits to infinity) number of steps;
    accessible in `end_state_distr` property

    * end transition matrix: what are the probabilities of
    the transitions in a very large (limits to infinity) number of steps;
    accessible in `end_trans_mat` property

    * end time percentage: how much time the system will spend in each of the states
    during a very long (limits to infinity) period of time;
    accessible in `time_percentage` property

    Returns
    -------
    end_behavior : `EndBehavior`
        class containing attributes for end behavior components

    Notes
    -----
    that only regular chains converge to those probabilities strongly;
    cyclic and mixed chains show different empirical end behavior.
    '''

    def __init__(self, *, init_distr: np.ndarray=None, trans_mat: np.ndarray) -> None:

        if init_distr is None: init_distr, _ = self.generate_chain(n_states=trans_mat.shape[0])
        
        self._check_chain(init_distr, trans_mat) # check if the components are valid

        self.init_distr = init_distr # initial states distribution
        self.trans_mat = trans_mat # transition matrix, i.d. P(1)
        self.adjacency_mat = self._adjacency_mat() # adjacency matrix, i.e. A
        self.reachability_mat = self._reachability_mat() # reachability matrix, i.e. D
        self.reachability_mat_tr = self._reachability_mat_tr() # transposed reachability matrix, i.e. DT
        self.communication_mat = self._communication_mat() # communication matrix, i.e. C
        self.communication_mat_comp = self._communication_mat_comp() # communication matrix complement, i.e. Ccomp
        self.classification_mat = self._classification_mat() # classification matrix, i.e. T
        self.classification_mat_ext = self._classification_mat_ext() # classification matrix extension, i.e. Text
        
        self.equivalency_cls_mat = self._build_structure() # builds `structure``, equivalency classes matrix

        # canonize matrices: add them `canonical` property
        cm = {'canonical_mapping': self.canonical_mapping}
        def canonize(matrix): return self._matrix(input_matrix=matrix, **cm)
        self.init_distr = canonize(self.init_distr) # initial states distribution
        self.trans_mat = canonize(self.trans_mat) # transition matrix, i.d. P(1)
        self.adjacency_mat = canonize(self.adjacency_mat) # adjacency matrix, i.e. A
        self.reachability_mat = canonize(self.reachability_mat) # reachability matrix, i.e. D
        self.reachability_mat_tr = canonize(self.reachability_mat_tr) # transposed reachability matrix, i.e. DT
        self.communication_mat = canonize(self.communication_mat) # communication matrix, i.e. C
        self.communication_mat_comp = canonize(self.communication_mat_comp) # communication matrix complement, i.e. Ccomp
        self.classification_mat = canonize(self.classification_mat) # classification matrix, i.e. T
        self.classification_mat_ext = canonize(self.classification_mat_ext) # classification matrix extension, i.e. Text
        self.equivalency_cls_mat = canonize(self.equivalency_cls_mat) # equivalency classes matrix

        self._classify() # builds `properties` property

        if not self.properties.reducible: end_behavior = ErgodicChainEndBehavior
        elif not self.properties.polyergodic: end_behavior = MonoergodicChainEndBehavior
        elif not self.properties.absorbing: end_behavior = PolyergodicChainEndBehavior
        else: end_behavior = PolyergodicAbsorbingChainEndBehavior

        self.end_behavior = end_behavior(self) # define end behavior

    #region Properties

    @property
    def n_states(self) -> int:

        '''
        ### Returns the number of states of the system.

        Returns
        -------
        n_states : `int`
            number of states
        '''

        n_states, n_states = self.trans_mat.shape
        return n_states

    @property
    def n_essential_states(self) -> int:

        '''
        ### Returns the number of essential states of the system.

        Returns
        -------
        n_essential_states : `int`
            number of essential states
        '''

        n_essential_states = len(self.structure.essential_states)
        return n_essential_states
    
    @property
    def n_inessential_states(self) -> int:

        '''
        ### Returns the number of inessential states of the system.

        Returns
        -------
        n_inessential_states : `int`
            number of inessential states
        '''

        n_inessential_states = len(self.structure.inessential_states)
        return n_inessential_states
    
    @property
    def n_equivalency_classes(self) -> int:

        '''
        ### Returns the number of equivalency classes of the system.

        Returns
        -------
        n_eq_classes : `int`
            number of equivalency classes
        '''

        n_eq_classes = len(self.structure.equivalency_classes)
        return n_eq_classes
    
    @property
    def n_cyclic_subclasses(self) -> int:

        '''
        ### Returns the number of all cyclic subclasses (from all equivalency classes) of the system.

        Returns
        -------
        n_subclasses : `int`
            number of equivalency classes
        '''

        n_subclasses = len(self.structure.cyclic_subclasses)
        return n_subclasses

    @property
    def shape(self) -> Tuple[int, int]:

        '''
        ### Returns the shape of the transition matrix.

        Returns
        -------
        shape : `tuple[int, int]`
            shape of the transition matrix
        '''

        shape = (self.n_states, self.n_states)
        return shape
    
    @property
    def canonical_mapping(self) -> Dict[int, int]:

        '''
        ### Returns mapping between original states and canonical numbers.

        Returns
        -------
        mapping : `dict[int, int]`
            mapping between original states and canonical numbers
        '''
        
        # return {
        #     state.id: state.canonical_number
        #     for state in self.structure.states
        # }
        mapping = dict(zip(
            self.structure.original_states,
            self.structure.canonical_states
        ))
        return mapping

    #endregion

    #region Utils

    @classmethod
    def generate_chain(cls, n_states: int) -> Tuple[np.ndarray]:

        '''
        ### Generates a transition matrix and an initial distribution.
        
        Parameters
        ----------
        n_states : `dict`
            number of states in the chain 

        Returns
        -------
        init_distr : `numpy.ndarray` of shape (`n_states`, )
            vector of initial distribution of probabilities of the states

        trans_mat : `numpy.ndarray` of shape (`n_states`, `n_states`)
            transition matrix, i.e. matrix of probabilities of transitions between states,
            where `trans_mat[i, j]` is probability of transition from the `i`th state to the `j`th state in a single step
        '''

        init_distr = np.random.rand(n_states)
        init_distr /= init_distr.sum()

        trans_mat = np.random.rand(n_states, n_states)
        for row in trans_mat: row /= row.sum()

        return init_distr, trans_mat

    def _check_chain(self, init_distr: np.ndarray, trans_mat: np.ndarray) -> None:

        '''
        ### Checks if matrices are valid components of a Markov chain.
        A Markov chain should have 2 components a transition matrix i.e. matrix of probabilities of transitions 
        between states and an initial distribution of probabilities of the states. Since the transition matrix 
        is a matrix of probabilities, the sum of the probabilities in each row should add up to \(1\); 
        the initial distribution probabilities should do so altogether.

        Raises
        ------
        AssertionError
            if any of the probabilities is out of the range \([0; 1]\).

        AssertionError
            if the shape of the initial distribution vector is not `(n, )`
            or the shape of the transition matrix is not `(n, n)`

        AssertionError
            if the probabilities of the initial distribution do not add up to \(1\)

        AssertionError
            if the probabilities in a row of the transition matrix do not add up to \(1\)

        AssertionError
            if the initial distribution vector and the transition matrix have different number of states
        '''

        assert np.all(init_distr >= 0), 'Probabilities can\'t be negative.'
        assert np.all(trans_mat >= 0), 'Probabilities can\'t be negative.'
        assert np.all(init_distr <= 1), 'Probabilities can\'t be grater than 1.'
        assert np.all(trans_mat <= 1), 'Probabilities can\'t be grater than 1.'

        assert len(init_distr.shape) == 1, 'The initial distribution vector should be a 1D array.' 
        assert np.isclose(init_distr.sum(), 1), 'The probabilities of initial distribution should add up to 1.'

        assert len(trans_mat.shape) == 2, 'The transition matrix should be a 1D array.' 
        assert trans_mat.shape[0] == trans_mat.shape[1], 'Transition matrix should be a square matrix.'
        for row in trans_mat: assert np.isclose(row.sum(), 1), 'The probabilities of all transitions from a state should add up to 1.'

        assert init_distr.shape[0] == trans_mat.shape[0], 'The components of the chain should have the same number of states.'

    @property
    def _zero_mask(self):

        '''
        ### Creates a zero matrix of the same shape the transition matrix has.

        Returns
        -------
        zero_mat : `numpy.ndarray` of shape (`n_states`, `n_states`)
            zero matrix of the transition matrix shape
        '''

        zero_mat = np.zeros(self.shape)
        return zero_mat
    
    @property
    def _all_ones_mask(self):

        '''
        ### Creates a matrix of ones of the same shape the transition matrix has.

        Returns
        -------
        ones_mat : `numpy.ndarray` of shape (`n_states`, `n_states`)
            matrix of ones of the transition matrix shape
        '''

        ones_mat = np.ones(self.shape)
        return ones_mat
    
    def _vertex_image(self, vertices: List[int]) -> List[int]:

        '''
        ### Finds the image of the given set ot vertices (i.e. states).
        Image of a state `i` is a set of all states such that there is a one-step path from `i` to them.
        Image of a set of states contains all states one-step reachable from one of the states in this set.

        Returns
        -------
        image : list[int]
            image of the given set of vertices (i.e. states)

        Raises
        ------
        AssertionError
            if there is an unknown to the system state in `vertices`

        Notes
        -----
        Formally, image is defined as follows:

        $$
        Г(i) = {j | a[i, j] = 1}
        $$

        $$
        Г(B) = union(Г(i)), i ∈ B
        $$

        where \(Г(i)\) / \(Г(B)\) is image of vertex / set of vertices, \((a[i, j])\) is the adjacency matrix
        '''

        states = set(range(self.n_states))
        assert set(vertices).issubset(states), f'The system does not have the following states: {set(vertices) - states}'

        _, image = np.where(self.trans_mat[vertices])
        image = list(set(image))
        image.sort() # set sorts it weirdly
        return image

    #endregion

    #region Iterations

    def trans_insteps(self, n_steps: int, use_canonical=True) -> np.ndarray:

        '''
        ### Calculates matrix of transitions in `n_steps` steps.
        Conceptually, that represents the probabilities that having begun in the state \(i\), the system
        will end up in the state \(j\) having made `n_steps` steps.

        Parameters
        ----------
        n_steps : `int`
            number of steps, where on each step, a transition takes place

        use_canonical : `bool`
            whether to use canonical numbering of the transition matrix

        Returns
        -------
        trans_mat_n : `numpy.ndarray` of shape (`n_states`, `n_states`)
            matrix of probabilities of transitions between states in `n_steps` steps,
            where `trans_mat_n[i, j]` is probability of transition from the state \(i\) to the state \(j\)
            in exactly `n_steps` steps (i.e. `n_steps` transitions)

        Notes
        -----
        Matrix of transitions in `n` step is calculated as follows:

        $$
        P^{(n)} = P^n
        $$

        where \(P(n)\) is matrix of transitions in \(n\) steps, \(n\) is the number of steps,
        \(P\) is matrix of transitions in \(1\) step (i.e. \(P^{(1)}\))

        NB! The rows of the matrix should add up to \(1\) as from each step, the system must be able to go
        to some state anyway, so the sum of the probabilities of the transitions will be 1.
        '''

        matrix = self.trans_mat if not use_canonical else self.trans_mat.canonical
        trans_mat_n = np.linalg.matrix_power(matrix, n=n_steps)
        return trans_mat_n
    
    def states_insteps(self, n_steps: int, use_canonical=True) -> np.ndarray:

        '''
        ### Calculates state distribution vector in `n_steps` steps.
        Conceptually, that represents the probabilities of system to be in the state \(j\) after `n_steps` steps.

        Parameters
        ----------
        n_steps : `int`
            number of steps, where on each step, a transition takes place

        use_canonical : `bool`
            whether to use canonical numbering of the state distribution vector

        Returns
        -------
        state_distr_n : `numpy.ndarray` of shape (`n_states`, )
            state probability distribution in `n_steps` steps (i.e. `n_steps` transitions)

        Notes
        -----
        State distribution in `n` step is calculated as:

        $$
        \pi^{(n)} = \pi^{(0)}P^{(n)}
        $$

        where \(\pi^{(n)}\) is state probability distribution in \(n\) steps,
        \(\pi^{(0)}\) is the initial state distribution

        NB! The vector should add up to \(1\) as the system will be at some state anyway, so the sum
        of the probabilities of all the states will be \(1\).
        '''

        init_distr = self.init_distr if not use_canonical else self.init_distr.canonical
        state_distr_n = np.matmul(self.init_distr, self.trans_insteps(n_steps=n_steps, use_canonical=use_canonical))
        return state_distr_n

    def first_trans_insteps(self, n_steps: int, use_canonical=True) -> np.ndarray:

        '''
        ### Calculates matrix of first transitions in `n_steps` steps.
        Conceptually, that represents the probabilities that having begun in the state \(i\), the system
        will reach the state \(i\) in `n_steps` steps for the first (!) time.

        Parameters
        ----------
        n_steps : `int`
            number of steps, where on each step, a transition takes place

        use_canonical : `bool`
            whether to use canonical numbering of the state distribution vector

        Returns
        -------
        first_trans_mat : `numpy.ndarray` of shape (`n_states`, `n_states`)
            matrix of probabilities of first transitions between states,
            where `first_trans_mat[i, j]` is probability that the system 
            will reach the `j`th state from the `i`th state for the first (!) time in exactly `n_steps` steps

        Notes
        -----
        Matrix of first transitions in `n` step is calculated recursively as follows:

        $$
        \\hat P^{(n)} = (\\hat p_{ij}^{(n)}) | \\hat p^{(n)}_{ij} = \\sum _{k \\neq j}p_{ik}\\hat p^{(n - 1)}_{kj}
        $$
        
        where \(\\hat P^{(n)}\) is matrix of first transitions in \(n\) steps, \(n\) is the number of steps,
        \((p_{ik})\) is the transition matrix (i.e. \(P(1)\)).

        In other words, the probability to get to the state \(j\) from the state \(i\) in \(n\) steps for the first time
        equals the sum of probabilities to get from \(i\) to any state \(h\) (which is not \(j\)) multiplied by
        probability to get to \(j\) from \(h\) for the first time in \(n - 1\) steps.

        NB! Note that this matrix's rows should not add up to \(1\). With a large enough number of steps,
        this matrix might become a zero matrix as it'll be impossible to get to some state for the first time
        in such a big amount of steps.
        '''

        trans_mat = self.trans_mat if not use_canonical else self.trans_mat.canonical
        first_trans_mat = trans_mat.copy() # `Phat(1)`
        for _ in range(n_steps - 1): # -1 because on each step we'll add the probability to finally reach the state
            # calculate recursively: on each step `m`, fill out matrix `Phat(m)`
            local_step = self._zero_mask
            for i in range(self.n_states):
                for j in range(self.n_states):
                    # from `i` to `j` the first time
                    not_j_states = np.delete(trans_mat[i], j) # now we should reach `j` from a not-`j` state so skip it
                    not_j_indices = np.delete(range(self.n_states), j)
                    # finally reach `j` from a not-`j` state: probabilities of path to a not-`j` state * to go from the not-`j`
                    # state to `j` for the first time in `m` - 1 steps (we refer to `phat` for that)
                    probs = not_j_states * first_trans_mat[not_j_indices, j]
                    prob = probs.sum() # all the paths from `i` to `j` where `j` is reached only at the very end
                    local_step[i, j] = prob
            first_trans_mat = local_step # now we'll refer to Phat(m) at the next step 
        return first_trans_mat
    
    def first_trans_nolaterthan(self, n_steps: int, use_canonical=True) -> np.ndarray:

        '''
        ### Calculates matrix of first transitions in no more than `n_steps` steps.
        Conceptually, that represents the probabilities that having begun in the state \(i\), the system
        will reach the state \(j\) in `n_steps` or less steps for the first (!) time.

        Parameters
        ----------
        n_steps : `int`
            number of steps, where on each step, a transition takes place

        use_canonical : `bool`
            whether to use canonical numbering of the state distribution vector

        Returns
        -------
        first_trans_mat_nolater : `numpy.ndarray` of shape (`n_states`, `n_states`)
            matrix of probabilities of first transitions between states,
            where `first_trans_mat_nolater[i, j]` is probability that the system 
            will reach the `j`th state from the `i`th state for the first (!) time in `n_steps` steps or less

        Notes
        -----
        Matrix of first transitions in no more than `n` step is calculated as sum of the 
        matrices of first transitions in exactly \(1\), \(2\), ..., `n` steps:

        $$
        \\hat P^{(\\leq n)} = \\sum _{t=1}^{n}\\hat P^t
        $$

        where \(\\hat P^{(\leq n)}\) is matrix of first transitions in \(n\) steps or less,
        \(\\hat P^{(n)}\) is matrix of first transitions in \(n\) steps, \(n\) is the number of steps.

        NB! Note that this matrix's rows should not add up to \(1\). With a large enough number of steps,
        the sum might be more that \(1\) because in a little number of steps,
        it might be possible to get to several states for the first time with a high probability.
        '''

        first_trans_mat_nolater = self._zero_mask
        for n in range(1, n_steps): first_trans_mat_nolater += self.first_trans_insteps(n_steps=n, use_canonical=use_canonical)
        return first_trans_mat_nolater

    #endregion

    #region Reachability

    def _adjacency_mat(self) -> np.ndarray:

        '''
        ### Builds adjacency matrix of the chain.
        Adjacency matrix is a matrix that indicates existence of one-step paths between states,
        where a one-step path from the state \(i\) to the state \(j\) exists 
        if there a non-zero probability transition between them.

        Returns
        -------
        adjacency_mat : `March._matrix` of shape (`n_states`, `n_states`)
            boolean-like matrix indicating one-step paths between states,
            where `adjacency_mat[i, j]` is \(1\) if the probability of tranition 
            from the state `i` to the state `j` is not zero, else \(0\)

        Notes
        -----
        Adjacency matrix is calculated as shown below:

        $$
        A = (a_{ij}) | a_{ij} =
        \\begin{cases}
            1, & p_{ij} > 0     \\\\
            0, & p_{ij} = 0
        \\end{cases}
        $$

        where \(A\) is adjacency matrix, \((p_{ij})\) is transition matrix.
        '''

        adjacency_mat = self._all_ones_mask
        adjacency_mat *= (self.trans_mat > 0)
        return adjacency_mat.astype(int)
    
    def _reachability_mat(self) -> np.ndarray:

        '''
        ### Builds reachability matrix following the Warshall's Algorithm. 
        Reachability matrix is a matrix that indicates existence of paths between states,
        where a path from the state \(i\) to the state \(j\) exists if there a sequence 
        of non-zero probability transitions between states such that it leads from \(i\) to \(j\)
        (e.g. \(i\) → \(h\) → \(k\) → ... → \(j\)).

        Returns
        -------
        reachability_mat : `March._matrix` of shape (`n_states`, `n_states`)
            boolean-like matrix indicating paths between states,
            where `reachability_mat[i, j]` is \(1\) if there is a path 
            from the state `i` to the state `j`, else \(0\)

        Notes
        -----
        Reachability matrix is calculated as shown below:

        $$
        D = (d_{ij}) | d_{ij} =
        \\begin{cases}
            1, & \\exists i → j        \\\\
            0, & \\neg \exists i → j
        \\end{cases}
        $$

        where \(D\) is reachability matrix, \(→\) marks a path.
        '''

        reachability_mat = self._zero_mask
        curr_adj_mat = self.adjacency_mat.copy() # start with the adjacency matrix
        for state in range(0, self.n_states):
            # states from where `state` is accessible in the current adjacency matrix
            adj_states, = np.where(curr_adj_mat[:, state])
            # for each of those states, update the row as such:
            # `A[i] := A[i] v A[state]`, where `i` ∈ `adj_states`
            curr_adj_mat[adj_states] |= curr_adj_mat[state]
        reachability_mat = curr_adj_mat
        return reachability_mat

    def _reachability_mat_tr(self) -> np.ndarray:

        '''
        ### Builds transposed reachability matrix. 
        Reachability matrix is a matrix that indicates existence of inverse paths between states,
        where a path from the state \(i\) to the state \(j\) exists if there a sequence 
        of non-zero probability transitions between states such that it leads from \(i\) to \(j\)
        (e.g. \(i\) → \(h\) → \(k\) → ... → \(j\)).

        Returns
        -------
        reachability_mat_tr : `March._matrix` of shape (`n_states`, `n_states`)
            boolean-like matrix indicating paths between states,
            where `reachability_mat_tr[i, j]` is \(1\) if there is a path 
            from the state `j` to the state `i`, else \(0\)

        Notes
        -----
        Transposed reachability matrix is calculated as shown below:

        $$
        D^T = (d^T_{ij}) | d^T_{ij} =
        \\begin{cases}
            1, & \\exists j → i        \\\\
            0, & \\neg \exists j → i
        \\end{cases}
        $$

        where \(D^T\) is transposed reachability matrix, \(→\) marks a path.
        '''

        reachability_mat_tr = self.reachability_mat.T
        return reachability_mat_tr
    
    def _communication_mat(self) -> np.ndarray:

        '''
        ### Builds communication matrix. 
        Communication matrix is a matrix that indicates existence of bidirectional paths between states,
        where a bidirectional path between the states \(i\) and \(j\) exists if there a sequence 
        of non-zero probability transitions from \(i\) to \(j\) (e.g. \(i\) → \(h\) → \(k\) → ... → \(j\))
        and from \(j\) to \(i\) (not exactly the same way back).

        Returns
        -------
        communication_mat : `March._matrix` of shape (`n_states`, `n_states`)
            boolean-like matrix indicating bidirectional paths between states,
            where `communication_mat[i, j]` is \(1\) if there is a path 
            from the state `j` to the state `i` and from `j` to `i`, else \(0\)

        Notes
        -----
        Communication matrix is calculated as shown below:

        $$
        C = (c_{ij}) = D \\times D^T | c_{ij} = 
        \\begin{cases}
            1, & d_{ij} = d_{ji} = 1        \\\\
            0, & \\neg (d_{ij} = d_{ji} = 1)
        \\end{cases}
        $$

        where \(C\) is communication matrix, \(D\) is reachability matrix,
        \(D^T\) is transposed reachability matrix, \(→\) marks a path.
        '''

        communication_mat = self.reachability_mat & self.reachability_mat_tr
        return communication_mat
    
    def _communication_mat_comp(self) -> np.ndarray:

        '''
        ### Builds communication matrix complement. 
        Communication matrix complement is a matrix that indicates nonexistence of bidirectional paths between states,
        where a bidirectional path between the states \(i\) and \(j\) exists if there a sequence 
        of non-zero probability transitions from \(i\) to \(j\) (e.g. \(i\) → \(h\) → \(k\) → ... → \(j\))
        and from \(j\) to \(i\) (not exactly the same way back).

        Returns
        -------
        communication_mat_comp : `March._matrix` of shape (`n_states`, `n_states`)
            boolean-like matrix indicating absence of bidirectional paths between states,
            where `communication_mat[i, j]` is \(1\) if there is no path 
            from the state `j` to the state `i` and from `j` to `i`, else 0

        Notes
        -----
        Communication matrix complement is calculated as follows:

        $$
        \\overline C = (\\overline c_{ij}) | \\overline c_{ij} = 
        \\begin{cases}
            1, & c_{ij} = 0     \\\\
            0, & c_{ij} = 1
        \\end{cases}
        $$

        where \(\\overline C\) is communication matrix complement, \(c_{ij}\) is communication matrix.
        '''

        communication_mat = self.communication_mat.copy().astype(bool)
        communication_mat_comp = np.invert(communication_mat).astype(int)
        return communication_mat_comp
    
    def _classification_mat(self) -> np.ndarray:

        '''
        ### Builds classification matrix. 
        Classification matrix is a step for building extended classification matrix.

        Returns
        -------
        classification_mat : `March._matrix` of shape (`n_states`, `n_states`)
            boolean-like matrix where `classification_mat[i, j]` is \(1\) if there is a path 
            from the state `i` to the state `j` but states `j` and `i` do not communicate, 0 otherwise

        Notes
        -----
        Classification matrix is calculated as follows:

        $$
        T = D \\times \\overline C = (t_{ij}) | t_{ij} = 
        \\begin{cases}
            1, & d_{ij} = \\overline c_{ij} = 1      \\\\
            0, & \\neg (d_{ij} = \\overline c_{ij} = 1)
        \\end{cases}
        $$

        where \(T\) is classification matrix, \(d_{ij}\) is reachability matrix, 
        \((\\overline c_{ij})\) is communication matrix complement.
        '''

        classification_mat = self.reachability_mat & self.communication_mat_comp
        return classification_mat
    
    def _classification_mat_ext(self) -> np.ndarray:

        '''
        ### Builds classification matrix extension. 
        Classification matrix extension indicates essential and inessential states.
        Essential states are the ones that communicate with each (!) state they have a path to;
        respectively, the ineccential are the ones for which this statement is not true.

        Returns
        -------
        classification_mat_ext : `March._matrix` of shape (`n_states`, `n_states` + 1)
            boolean-like matrix where the last column indicates essentiality of states:
            if classification_mat_ext[`i`, `n_states` + 1] equals 1, the state `i`
            is inessential, if 0, is essential

        Notes
        -----
        Classification matrix extention is defined below:

        $$
        T_{ext} = T|(t_{i(N + 1)}) | t_{i(N + 1)} = 
        \\begin{cases}
            1, & \\exists j t_{ij} = 1        \\\\
            0, & \\neg \\exists j t_{ij} = 1
        \\end{cases}
        $$

        where \(T_{ext}\) is classification matrix extension, \(T\) is classification matrix,
        \(N\) is the number of states in the system.
        '''

        classification_mat = self.classification_mat.copy()
        extension = np.any(classification_mat, axis=1).astype(int)
        classification_mat_ext = np.concatenate((classification_mat, extension.reshape(-1, 1)), axis=1)
        return classification_mat_ext

    #endregion

    #region Structure

    def _build_structure(self) -> None:

        '''
        ### Build structure of the chain (into `structure` property).
        That includes:

        * Determining essential and inessential states:
        Essential state is such a state \(i\) that communicates with every (!) state \(j\)
        it has a path to; respectively, a state is inessential if that is not true.

        * Building matrix of equivalency classes:
        In each Markov system, the set of essential states can be uniquely split into 
        classes of equivalency. In each equivalency class, all essential states
        communicate with each other and each class is closed, which means
        once the system got into an essential state in class \(EC_i\), it will never leave that class
        and all the transitions will be only between states within this class.

        * Finding cyclic subclasses of equivalence classes:
        Each equivalence classes has a period \(d\): it equals the GCD of the lengths
        of all the contours going through a state \(i\) belonging to the equivalence class.
        Contour is a cyclic path from a state to itself (e.g. \(i → h → k → ... → i\)), 
        its length is the number of the transitions within it.
        It is proven that all the states of an equivalence class have the same period \(d\).
        If \(d\) equals 1, the equivalence class is acyclic.
        If it is \(\geq\) 2, it can be split into \(d\) cyclic subclasses. Cyclic subclass \(C_r\) of equivalence class \(EC_n\)
        is a subset of states of \(EC_n\) such that 
        
        * the states are not adjacent (i.e. there is no one-step path between them);
        * from any of the states there is a one-step path only to a state of the next cyclic subclass \(C_{r + 1}\);
        * the states belong to an only cyclic subclass;
        * the equivalence class is fully distributed between cyclic subclasses (no states left outside of the subclasses).

        That means the cyclic subclasses are closed and form a cycle themselves:

        \(C_0 → C_1 → C_2 → ... → C_{d - 1} → C_0\)

        * Canonical numbering of the states in the system structure (transformed chain components are built separately):
        Canonical numbering is a numbering that goes in strict order through each state in each cyclic subclass
        of each equivalency class, followed only then by inessential states. That allows to transform the transition matrix
        into a block matrix that represents probabilities of transition between cyclic subclasses.

        Returns
        -------
        equivalency_cls_mat : `numpy.ndarray` of shape (`n_classes`, `n_essential`)
            matrix showing belonging of each essential (!) state to its equivalence class;
            `equivalency_cls_mat[i, j]` indicates that the state `j` is in `i`th equivalence class

        Notes
        -----
        Essentiality of states if defined as follows:
        $$
        i \\in S_e \\iff 
        \\forall j: \\quad i → j \\Longrightarrow j → i
        $$
        $$
        i \\in S_{ie} \\iff 
        \\neg \\forall j: \\quad i → j \\Longrightarrow j → i
        $$

        Where \(S_e\) is the set of essential states, \(S_{ie}\) is the set of inessential states.


        Matrix of equivalency classes is defined as follows:

        $$
        K = (k_{ij}) | k_{it} \\Longleftrightarrow i \\in EC_t
        $$

        where \(K\) is equivalency matrix

        Formally, the period of an equivalency class is found as:

        $$
        d = \\text{GCD}(|(i → i)_1|, ..., |(i → i)_r|)
        $$
            
        where \((i → i)\) is a contour, \(|(i → i)_1|\) is a length of a contour

        If \(d\) equials 1, the equivalence class is acyclic, which means it has no cyclic subclasses.

        
        Canonical numbering goes as follows:

        \(EC_1\):

        \(C_0: \\quad 1, \\quad 2, \\quad 3, \\quad ..., \\quad r_0\)

        \(C_1: \\quad r_0 + 1, \\quad r_0 + 2, \\quad ..., \\quad r_1\)

        \(...\)

        \(C_{d - 1}: r_0 + r_1 + ... + r_{d - 1} + 1, \\quad r_0 + r_1 + ... + r_{d - 1} + 2, 
        \\quad ..., \\quad r_0 + r_1 + ... + r_{d - 1} + r_{d} = N_{EC_1}\)

        \(EC_2\): (same way)

        \(...\)

        \(EC_L\): (same way)

        \(S_{ie}: \\quad N_e + 1, \\quad N_e + 2, \\quad ..., \\quad N_e + N_{ie} = N\)

        where \(EC_k\) is equivalence class \(k\), \(C_s\) is a cyclic subclass \(s\) of the equivalence class,
        \(N_{EC_k}\) is number of states in equivalence class \(k\), \(N_e\) is number of essential states in the system,
        \(N\) is number of states in the system.
        '''

        # NB! We keep all the data pieces as lists as the dataclasses for them are immutable after initialization
        can_id = 0 # for canonical numbering
        # collections for the whole chain
        states_, es_states_, ines_states_ = [], [], []
        cyclic_subclasses_ = []
        equivalency_classes_ = []

        # determine essential and inessential states
        extention = self.classification_mat_ext[:, -1]
        essential_states, = np.where(extention == 0)
        inessential_states, = np.where(extention)

        # equivalency classification matrix
        communication_mat = self.communication_mat.copy()
        # delete rows and columns of all inessential states
        communication_mat = np.delete(communication_mat, inessential_states, axis=0)
        communication_mat = np.delete(communication_mat, inessential_states, axis=1)
        # the resulting matrix will fall apart into groups of repeating rows
        # after removing dublicates we'll get our equivalency classes matrix
        equivalency_cls_mat, indices = np.unique(communication_mat, return_index=True, axis=0)
        equivalency_cls_mat = equivalency_cls_mat[indices.argsort()]

        # in the equivalency classification matrix, the indices of the states
        # might not correspond to the actual state indexing (since we delete columns),
        # which means we need a mapping between the index of a state in the matrix
        # and the actual essential state id 
        mapping = {
            number: state for state, number 
            # `i`th index in the matrix is the `i`th essential (!) state 
            in zip(essential_states, range(len(essential_states)))
        }
        for cl_id, row in enumerate(equivalency_cls_mat):
            class_states, = np.where(row)
            equivalency_class = [mapping[class_state] for class_state in class_states]

            # collections for the equivalence class
            cls_states_ = []
            subclasses_ = []

            # determine cyclic subclasses
            candidates = [[equivalency_class[0]]] # start with any state in the equivalence class
            while True:

                image = self._vertex_image(candidates[-1]) # find its image
                if image == candidates[0]: break # cycle is closed
                candidates.append(image) # proceed otherwise

                to_update, to_pop = [], []
                for ind, prev_cand in enumerate(candidates[:-1]): # check if any of previous candidates overlap with the current one
                    if set(prev_cand) & set(candidates[-1]):
                        to_update.extend(prev_cand)
                        to_pop.append(ind)
                candidates[-1].extend(to_update) # merge if overlap
                candidates[-1] = list(set(candidates[-1]))
                candidates[-1].sort() # set sorts it weirdly
                candidates = [
                    candidate for j, candidate in enumerate(candidates)
                    if not j in to_pop
                ] # remove old candidates, now they are merged into a new one
            cyclic_subclasses = candidates

            # fill in the structure: since we will add 0th to `r`th state of every
            # 0th to `m`th cyclic subclass of each 0th to `s`th equivalency class,
            # the order of appending will follow the canonical numbering!
            for subcl_id, state_ids in enumerate(cyclic_subclasses):
                
                # collection for the cyclic subclass
                subcl_states_ = []

                for state_id in state_ids:
                    state = EssentialState(
                        id=state_id,
                        canonical_number=can_id,
                        equivalency_class=cl_id,
                        cyclic_subclass=subcl_id
                    )

                    # update cyclic subclass
                    subcl_states_.append(state)
                    can_id += 1
                
                subclass = CyclicSubclass(
                    id=subcl_id,
                    class_id=cl_id,
                    states=subcl_states_
                )

                # update equivalency class
                cls_states_.extend(subcl_states_)
                subclasses_.append(subclass)

            d = len(subclasses_)
            eq_class = EquivalenceClass(
                id=cl_id,
                d=d,
                states=cls_states_,
                cyclic_subclasses=subclasses_ if d > 1 else None
            )

            # update chain structure
            cyclic_subclasses_.extend(subclasses_)
            equivalency_classes_.append(eq_class)
            states_.extend(cls_states_); es_states_.extend(cls_states_)

        # finally add inessential states
        for state_id in inessential_states:
            state = InessentialState(
                id=state_id,
                canonical_number=can_id
            )
            states_.append(state); ines_states_.append(state)
            can_id += 1

        structure = ChainStructure(
            states=states_,
            essential_states=es_states_,
            inessential_states=ines_states_,
            equivalency_classes=equivalency_classes_,
            cyclic_subclasses=cyclic_subclasses_
        )

        self.structure = structure
        return equivalency_cls_mat
  
    #endregion

    #region Classification

    def _classify(self) -> None:

        '''
        ### Classifies the chain (into `properties` property).
        Markov Chain classification:

        1. \(S\) = \(Se\) = \(EC\) ⇒ ergodic / irreducible Markov Chain

            1.1. \(EC\) is regular ⇒ regular irreducible Markov Chain

            \(\quad\) 1.1.1. \(EC\) is absorbing ⇒ absorbing irreducible Markov Chain

            1.2. \(EC\) is cyclic ⇒ cyclic irreducible Markov Chain

        2. \(Sie ≠ ∅\) ⇒ reducible Markov Chain

            2.1. a single \(EC\) ⇒ monoergodic Markov Chain

            \(\quad\)2.1.1. the single \(EC\) is regular ⇒ regular monoergodic Markov Chain

            \(\qquad\) 2.1.1.1. the single \(EC\) is absorbing ⇒ absorbing monoergodic Markov Chain

            \(\quad\) 2.1.2. the single \(EC\) is cyclic ⇒ cyclic monoergodic Markov Chain

            2.2. more than one \(EC\)s ⇒ polyergodic Markov Chain

            \(\quad\) 2.2.1. all the \(EC\)s are regular ⇒ regular polyergodic Markov Chain

            \(\qquad\) 2.2.1.1. all the \(EC\)s are absorbing ⇒ absorbing polyergodic Markov Chain

            \(\quad\) 2.2.2. all the \(EC\)s are cyclic ⇒ cyclic polyergodic Markov Chain

            \(\quad\) 2.2.3. there are both regular and cyclic \(EC\)s ⇒ mixed polyergodic Markov Chain,

        where \(S\) is set of states, \(Se\) is set of essential states, \(Sie\) is set of inessential states,
        \(EC\) stands for equivalence class.

        Returns
        -------
        structure : `ChainProperties`
            class containing attributes for chain properties

        Notes
        -----
        Note that in our classification we won't just return a single label of the class, but rather
        use a more "linguistic" notation: we will define a set of properties, each of which can be 
        either `True`, or `False`, or `None` (if not applicable). Thus, we will be able to 
        split the space of classes uniquely, yet be able to access chain features in a more convenient way.

        Reminder: a \(EC\) is regular if it has a period \(d\) = 1, i.e. has no cyclic subclasses, otherwise cyclic.
        Reminder: a \(EC\) is absorbing if it contains a single state.
        
        The properties are:

        * `reducible` (if there are inessential states)
        * `polyergodic` (if there are more than one `EC`)
        * `regular` (cyclic if `False`, mixed if `None`)
        * `absorbing` (may be `True` only if regular)
        * `strong_convergence` (if the matrix strongly converges to a final matrix in a great number of steps)

        The matrix converges strongly if it's regular, weakly otherwise.

        Thus, we'll have a "linguistic-like" notation, e.g. a cyclic polyergodic Markov Chain
        could be notated as ```[+reducible][+polyergodic][-regular][-absorbing][-strong_convergence]```.
        '''

        reducible = None
        polyergodic = None
        regular = None
        absorbing = None
        strong_convergence = None

        if not self.n_inessential_states: # irreducible
            
            reducible = False
            polyergodic = False # it has a single equivalency class then
            period = self.structure.equivalency_classes[0].d
            len_eq_class = len(self.structure.equivalency_classes[0])
            regular = period == 1
            absorbing = len_eq_class == 1 if regular else False

        else: # reducible
            
            reducible = True

            if self.n_equivalency_classes == 1: # monoergodic

                polyergodic = False
                period = self.structure.equivalency_classes[0].d
                len_eq_class = len(self.structure.equivalency_classes[0])
                regular = period == 1
                absorbing = len_eq_class == 1 if regular else False

            else: # polyergodic

                polyergodic = True
                periods = [eq_class.d for eq_class in self.structure.equivalency_classes]
                lens_eq_class = [len(eq_class) for eq_class in self.structure.equivalency_classes]
                if np.all(np.array(periods) == 1): regular = True
                elif np.all(np.array(periods) > 1): regular = False
                # else: regular = None # mixed
                absorbing = np.mean(lens_eq_class) == 1 if regular else False

        strong_convergence = regular == True # might be `None` is mixed so compare to `True`

        self.properties = ChainProperties(
            reducible=reducible,
            polyergodic=polyergodic,
            regular=regular,
            absorbing=absorbing,
            strong_convergence=strong_convergence
        )

    #endregion

    def to_graph(self, use_canonical=True):

        '''
        ### Shows the graph on the transition matrix.

        Parameters
        ----------
        use_canonical : `bool`
            whether to use canonical numbering

        Returns
        -------
        graph : `graphviz.Digraph`
            graph
        '''

        matrix = self.trans_mat.canonical if use_canonical else self.trans_mat.original

        graph = graphviz.Digraph()
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if value == 0: continue
                graph.edge(
                    tail_name=str(i),
                    head_name=str(j),
                    label=str(value),
                    arrowhead='empty',
                    arrowsize='0.6',
                    fontsize='12',
                    shape='circle'
                )
        graph.attr(
            layout='dot',
            size='12.0',
            rankdir='LR'
        )
        graph.node_attr['shape'] = 'circle'
        return graph

    def __len__(self) -> int:
        return self.n_states

    def __repr__(self) -> str:
        return f'{self.properties.type_label} {repr(self.structure)}\n\n{repr(self.properties)}'
    
    def __str__(self) -> str:
        return self.__repr__()