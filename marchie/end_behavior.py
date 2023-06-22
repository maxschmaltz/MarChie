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
from typing import Tuple

#endregion


def _solve_equation_system(coefficient_mat: np.ndarray, solutions: np.ndarray) -> np.ndarray:

    '''
    ### Solves an equation system with Cramer's Rule.
    Cramer's Rule is a method to solve an equation system where
    the number of variables and the number of equations is the same.
    In the method, the matrix of the coefficients of the variables
    and the solutions of the equations are needed.

    Parameters
    ----------
    coefficient_mat : `numpy.ndarray` of shape (`n_equations`, `n_coefficients`), `n_equations` = `n_coefficients`
        matrix of the coefficients of the variables

    solutions : `numpy.ndarray` of shape (`n_equations`, )
        vector of solutions of the equations

    Returns
    -------
    pi : `numpy.ndarray` of shape (`n_equations`, )
        vector of variable values
    '''

    _, n_coefficients = coefficient_mat.shape
    pi = np.zeros((n_coefficients)) # pi is the vector of variables
    coef_delta = np.linalg.det(coefficient_mat)
    for i in range(n_coefficients):
        sub_mat = coefficient_mat.copy()
        sub_mat[:, i] = solutions.T
        delta = np.linalg.det(sub_mat)
        pi[i] = delta / coef_delta
    return pi


#region End Behavior

class EndBehavior:

    '''
    ### Base class to determine end behavior of a Markov Chain.
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

    Note that only regular chains converge to those probabilities strongly;
    cyclic and mixed chains show different empirical end behavior.

    You may refer to the demo notebook ("./demo.ipynb") to find out how the
    end behavior for different types of chains are calculated.

    Parameters
    ----------
    chain : `MarChie`
        an instantiated `MarChie` object to calculate the end behavior of
    '''

    end_state_distr: np.ndarray
    '''
    ### End state distribution vector.
    Calculates what are the probabilities of
    the system to be in each state in a very large (limits to infinity) number of steps.

    Returns
    ----------
    end_state_distr : `numpy.ndarray` of shape (`chain.n_states`, )
        end state distribution vector
    '''

    end_trans_mat: np.ndarray
    '''
    ### End transition matrix.
    Calculates what are the probabilities of
    the transitions in a very large (limits to infinity) number of steps.

    Returns
    ----------
    end_trans_mat : `numpy.ndarray` of shape (`chain.n_states`, `chain.n_states`)
        end transition matrix
    '''

    time_percentage: np.ndarray
    '''
    # End time percentage.
    Calculates how much time the system will spend in each of the states
    during a very long (limits to infinity) period of time.

    Returns
    -------
    time_percentage : `numpy.ndarray` of shape (`chain.n_states`, )
        vector of percentages (0-100) for each state in canonical numbering
    '''

    def __init__(self, chain) -> None:
        self.chain = chain
        self.end_state_distr = self._end_state_distr()
        self.end_trans_mat = self._end_trans_mat()  
        self.time_percentage = self._time_percentage()

    def equation_system(self, matrix: np.ndarray=None) -> Tuple[np.ndarray]:

        '''
        ### Converts a square matrix into an equation system.

        Parameters
        ----------
        matrix : `numpy.ndarray` of shape (`n`, `n`), optional, defaults to `None`
            the matrix to turn into an equation system; if `None`,
            convert the transition matrix of `chain`

        Returns
        -------
        coefficient_matrix : `numpy.ndarray` of shape (`n`, `n`)
            matrix of coefficients of the equation system
        
        answers : `numpy.ndarray` of shape (`n`, )
            vector of answers of the equation system

        Raises
        ------
        AssertionError
            if `matrix` is not square
        '''

        ## Below are examples for a better understanding why the system is built this way.
        ## Imagine we have 3 states, then the transition matrix looks like:
        ##   /   c00     c01     c02   \
        ##   |   c10     c11     c12   |
        ##   \   c20     c21     c22   /
        ## 
        ## If a chain fragment P (e.g. standing for an equivalent class or the whole matrix) 
        ## limits to some end matrix (fragment) Π, that end fragment Π can be represented
        ## as a matrix with equal rows π = (π0, π1, ...), in out case:
        ##   /   π0      π1      π2   \
        ##   |   π0      π1      π2   |
        ##   \   π0      π1      π2   /
        ## 
        ## That actually means that in the end behavior, the following statement will be fair:
        ## πP = π
        ##
        ## It is fair as the end fragment becomes stationary at its end.
        ## If we rewrite the equation, we get:
        ##                          /   c00     c01     c02   \
        ##   (π0      π1      π2)   |   c10     c11     c12   | = (π0      π1      π2)
        ##                          \   c20     c21     c22   /
        ##
        ## Now we can actually multiply matrices and get the equation system:
        ##   |   c00π0 + c10π1 + c20π2 = π0
        ##   |   c01π0 + c11π1 + c21π2 = π1
        ##   |   c02π0 + c12π1 + c22π2 = π2
        ##
        ## We can subtract the answers:
        ##   |   (c00 - 1)π0 + c10π1 + c20π2 = 0
        ##   |   c01π0 + (c11 - 1)π1 + c21π2 = 0
        ##   |   c02π0 + c12π1 + (c22 - 1)π2 = 0
        ##
        ## coefficient matrix looks like this then:
        ##   / c00 - 1   c10     c20   \
        ##   |   c01   c11 - 1   c21   |
        ##   \   c02     c12   c22 - 1 /
        ##
        ## As one can see, the coefficient matrix is basically just the transposed
        ## transition matrix, where the values at the diagonal are subtracted by 1
        ## (just because we subtracted 1 * πi from each equation).
        ## Now the only problem there might be is that the coefficient matrix is nondegenerate
        ## (the determiner equals 0), hence, we will not be able to solve it with Creamer's rule.
        ## To avoid that, we might substitute an equation in the system with
        ## the π vector (it's stochastic, so we know for sure that sum(π) equals 1):
        ##   |   π0    + π1    +   2         = 1
        ##   |   c01π0 + (c11 - 1)π1 + c21π2 = 0
        ##   |   c02π0 + c12π1 + (c22 - 1)π2 = 0
        ##
        ## In the end we have a coefficient matrix that equals the transposed 
        ## transition matrix where the values at the diagonal are subtracted by 1
        ## and some row is substituted by ones (since it's 1 * πi):
        ##   /    1       1        1   \
        ##   |   c01   c11 - 1   c21   |
        ##   \   c02     c12   c22 - 1 /

        if matrix is None: matrix = self.chain.trans_mat.canonical
        n_states, n = matrix.shape

        assert n_states == n, 'Only square matrices are supported.'

        coefficient_matrix = np.zeros(matrix.shape)
        coefficient_matrix = matrix.T # transpose to get coefficients
        for i in range(n_states): coefficient_matrix[i, i] -= 1 # subtract answers
        coefficient_matrix[0] = np.ones((n_states)) # substitute an equation (say, the first)
        answers = np.zeros((n_states)); answers[0] = 1 # update answers
        return coefficient_matrix, answers

    def _end_state_distr(self) -> np.ndarray:

        '''
        ### Defines end state distribution vector.
        Calculates what are the probabilities of
        the system to be in each state in a very large (limits to infinity) number of steps.

        Raises
        ------
        NotImplementedError
            if trying to instantiate the base class 
        '''
        
        raise NotImplementedError('You should define end state distribution vector in your derived class.')

    def _end_trans_mat(self) -> np.ndarray:

        '''
        ### Defines end transition matrix.
        Calculates what are the probabilities of
        the transitions in a very large (limits to infinity) number of steps.

        Raises
        ------
        NotImplementedError
            if trying to instantiate the base class 
        '''

        raise NotImplementedError('You should define end transition probability matrix in your derived class.')

    def _time_percentage(self) -> np.ndarray:

        '''
        ### Defines end time percentage.
        Calculates how much time the system will spend in each of the states
        during a very long (limits to infinity) period of time.

        Returns
        -------
        time_percentage : `numpy.ndarray` of shape (`chain.n_states`, )
            vector of percentages (0-100) for each state in canonical numbering
        '''
        
        time_percentage = self.end_state_distr * 100
        return time_percentage
    

class ErgodicChainEndBehavior(EndBehavior):

    '''
    ### Class to determine end behavior of an ergodic (irreducible) Markov Chain.
    An ergodic (irreducible) Markov Chain has no inessential states and 
    a single equivalency class (and thus coincide with it).

    Refer to the base class `EndBehavior` for parameters and description 
    of what the end behavior is.
    '''

    def _end_state_distr(self) -> np.ndarray:

        '''
        ### Defines end state distribution vector.
        Calculates what are the probabilities of
        the system to be in each state in a very large (limits to infinity) number of steps.

        Returns
        ----------
        end_state_distr : `numpy.ndarray` of shape (`chain.n_states`, )
            end state distribution vector
        '''

        end_state_distr = _solve_equation_system(
            *self.equation_system()
        )
        return end_state_distr

    def _end_trans_mat(self) -> np.ndarray:

        '''
        ### Defines end transition matrix.
        Calculates what are the probabilities of
        the transitions in a very large (limits to infinity) number of steps.

        Returns
        ----------
        end_trans_mat : `numpy.ndarray` of shape (`chain.n_states`, `chain.n_states`)
            end transition matrix
        '''

        end_trans_mat = self.chain._zero_mask
        end_trans_mat[:] = self.end_state_distr
        return end_trans_mat


class MonoergodicChainEndBehavior(EndBehavior):

    '''
    ### Class to determine end behavior of a monoergodic Markov Chain.
    A monoergodic Markov Chain has \(1\) or more inessential states and 
    a single equivalency class.

    Refer to the base class `EndBehavior` for parameters and description 
    of what the end behavior is.
    '''

    def _end_state_distr(self) -> np.ndarray:

        '''
        ### Defines end state distribution vector.
        Calculates what are the probabilities of
        the system to be in each state in a very large (limits to infinity) number of steps.

        Returns
        ----------
        end_state_distr : `numpy.ndarray` of shape (`chain.n_states`, )
            end state distribution vector
        '''

        essential_submat = self.chain.trans_mat.canonical[
            # left upper corner given the canonical numbering
            0: self.chain.n_essential_states,
            0: self.chain.n_essential_states
        ]
        essential_end_state_distr = _solve_equation_system(
            *self.equation_system(essential_submat)
        )
        end_zeros = np.zeros((self.chain.n_inessential_states))
        end_state_distr = np.concatenate((essential_end_state_distr, end_zeros), axis=0)
        return end_state_distr

    def _end_trans_mat(self) -> np.ndarray:

        '''
        ### Defines end transition matrix.
        Calculates what are the probabilities of
        the transitions in a very large (limits to infinity) number of steps.

        Returns
        ----------
        end_trans_mat : `numpy.ndarray` of shape (`chain.n_states`, `chain.n_states`)
            end transition matrix
        '''

        end_trans_mat = self.chain._zero_mask
        end_trans_mat[:] = self.end_state_distr
        return end_trans_mat


class PolyergodicChainEndBehavior(EndBehavior):

    '''
    ### Class to determine end behavior of a polyergodic Markov Chain.
    A polyergodic Markov Chain has \(1\) or more inessential states and 
    several (\(2\) or more) equivalency classes.

    Refer to the base class `EndBehavior` for parameters and description 
    of what the end behavior is.
    '''
  
    def __init__(self, chain) -> None:
        self.chain = chain
        self.end_trans_mat = self._end_trans_mat()  
        self.end_state_distr = self._end_state_distr()
        self.time_percentage = None

    def _end_state_distr(self) -> np.ndarray:

        '''
        ### Defines end state distribution vector.
        Calculates what are the probabilities of
        the system to be in each state in a very large (limits to infinity) number of steps.

        Returns
        ----------
        end_state_distr : `numpy.ndarray` of shape (`chain.n_states`, )
            end state distribution vector
        '''

        end_state_distr = np.matmul(
            self.chain.init_distr,
            self.end_trans_mat
        )
        return end_state_distr

    def _end_trans_mat(self) -> np.ndarray:

        '''
        ### Defines end transition matrix.
        Calculates what are the probabilities of
        the transitions in a very large (limits to infinity) number of steps.

        Returns
        ----------
        end_trans_mat : `numpy.ndarray` of shape (`chain.n_states`, `chain.n_states`)
            end transition matrix
        '''

        c = 0

        # essential submatrix `Пe`
        essential_submat = np.zeros((self.chain.n_essential_states, self.chain.n_states))
        essential_vectors = []
        for i, eq_class in enumerate(self.chain.structure.equivalency_classes):

            # transitions in each class `t` are described by its submatrix `Tt`
            begin, end = eq_class.canonical_states[0], eq_class.canonical_states[-1]
            class_mat = self.chain.trans_mat.canonical[
                begin: end + 1,
                begin: end + 1
            ]

            # each row is a vector of `π` form `(0, ..., π_, ..., 0)`,
            # `π_` is a subvector such that `π_ x Tt = π_`;
            # here we find `π_`
            class_vec = _solve_equation_system(
                *self.equation_system(class_mat)
            )
            # number of zeros on the left from `π_`
            n_zeros_left = sum([
                len(prev_eq_class) 
                for prev_eq_class in self.chain.structure.equivalency_classes[:i]
            ])
            # number of zeros on the right from `π_`
            n_zeros_right = sum([
                len(next_eq_class) 
                for next_eq_class in self.chain.structure.equivalency_classes[i + 1:]
            ]) + self.chain.n_inessential_states

            # finally `π`
            class_vec = np.concatenate((
                np.zeros((n_zeros_left)),
                class_vec,
                np.zeros((n_zeros_right))
            ), axis=0)

            # put in `Пe`; store `π`
            for _ in range(len(eq_class)): essential_submat[c] = class_vec; c += 1;
            essential_vectors.append(class_vec)


        # inessential to essential submatrix `B`
        absorbing_trans_mat = self.chain.trans_mat.canonical.copy()
        states_to_delete = []
        for eq_class in self.chain.structure.equivalency_classes:
            # leave the first state of each class
            states_to_delete += eq_class.canonical_states[1:]
            # substitute the probabilities with a single sum of them
            absorbing_trans_mat[:, eq_class.canonical_states[0]] = (absorbing_trans_mat[:, eq_class.canonical_states]).sum(axis=1)
        # delete remaining states
        absorbing_trans_mat = np.delete(absorbing_trans_mat, states_to_delete, axis=0)
        absorbing_trans_mat = np.delete(absorbing_trans_mat, states_to_delete, axis=1)

        absorbing_n_states, _ = absorbing_trans_mat.shape
        # each equivalency class we substituted with a single state,
        # which means in `absorbing_trans_mat` the number of essential states
        # equals the number of equivalency classes in the original matrix
        absorbing_identity_mat = np.identity(self.chain.n_inessential_states)
        absorbing_inessential_submat = absorbing_trans_mat[
            # right lower corner given the canonical numbering
            self.chain.n_equivalency_classes: absorbing_n_states,
            self.chain.n_equivalency_classes: absorbing_n_states
        ]
        absorbing_end_inessential_submat = np.linalg.inv(absorbing_identity_mat - absorbing_inessential_submat)
        absorbing_inessential_to_essential_submat = absorbing_trans_mat[
            # left lower corner given the canonical numbering
            self.chain.n_equivalency_classes: absorbing_n_states,
            0: self.chain.n_equivalency_classes
        ]
        end_inessential_to_essential_submat = np.matmul(
            absorbing_end_inessential_submat,
            absorbing_inessential_to_essential_submat
        )


        # inessential submatrix `Пie`
        inessential_submat = np.zeros((self.chain.n_inessential_states, self.chain.n_states))
        for i, iness_row in enumerate(inessential_submat):
            for t, ess_row in enumerate(essential_vectors):
                iness_row += end_inessential_to_essential_submat[i, t] * ess_row


        end_trans_mat = np.concatenate((essential_submat, inessential_submat), axis=0)
        return end_trans_mat

    def _time_percentage(self) -> np.ndarray:

        '''
        ### Defines end time percentage.
        Calculates how much time the system will spend in each of the states
        during a very long (limits to infinity) period of time.
        NB! Not supported for polyergodic chains.

        Raises
        -------
        NotImplementedError
            when trying to calculate end time percentage for a polyergodic Markov Chain
        '''

        raise NotImplementedError(
            'Percentages of time for each state are not supposed to be calculated for polyergodic chains.'
        )
  

class PolyergodicAbsorbingChainEndBehavior(PolyergodicChainEndBehavior):

    '''
    ### Class to determine end behavior of a polyergodic absorbing Markov Chain.
    A polyergodic absorbing Markov Chain has 1 or more inessential states and 
    several (2 or more) equivalency classes, each of which is absorbing.

    Refer to the base class `EndBehavior` for parameters and description 
    of what the end behavior is.
    '''

    def _end_trans_mat(self) -> np.ndarray:

        '''
        ### Defines end transition matrix.
        Calculates what are the probabilities of
        the transitions in a very large (limits to infinity) number of steps.

        Returns
        ----------
        end_trans_mat : `numpy.ndarray` of shape (`chain.n_states`, `chain.n_states`)
            end transition matrix
        '''

        # essential submatrix `T`
        end_essential_submat = np.identity(self.chain.n_essential_states)

        # inessential to essential submatrix `B`
        identity_mat = np.identity(self.chain.n_inessential_states)
        inessential_submat = self.chain.trans_mat.canonical[
            # right lower corner given the canonical numbering
            self.chain.n_essential_states: self.chain.n_states,
            self.chain.n_essential_states: self.chain.n_states
        ]
        end_inessential_submat = np.linalg.inv(identity_mat - inessential_submat)
        inessential_to_essential_submat = self.chain.trans_mat.canonical[
            # left lower corner given the canonical numbering
            self.chain.n_essential_states: self.chain.n_states,
            0: self.chain.n_essential_states
        ]
        end_inessential_to_essential_submat = np.matmul(end_inessential_submat, inessential_to_essential_submat)

        end_trans_mat = self.chain._zero_mask
        end_trans_mat[
            # left upper corner given the canonical numbering
            0: self.chain.n_essential_states,
            0: self.chain.n_essential_states
        ] = end_essential_submat
        end_trans_mat[
            # left lower corner given the canonical numbering
            self.chain.n_essential_states: self.chain.n_states,
            0: self.chain.n_essential_states
        ] = end_inessential_to_essential_submat

        return end_trans_mat
    
#endregion
