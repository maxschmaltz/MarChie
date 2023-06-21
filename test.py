#region Imports

import unittest
import numpy as np

from marchie import *

#endregion


#region Tests

class MarChieTest(unittest.TestCase):

    '''
    ### Tests features of a Markov Chain: reachability, communication etc.
    '''

    #region Check Chain Tests

    def test_check_marchie_0(self):

        trans_mat = np.array([
            [0.5,   0.5],
            [1,     0  ]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        self.assertIsInstance(marchie, MarChie)

    def test_check_marchie_1(self):
        
        trans_mat = np.array([
            [0.5,   0.5],
            [1,     0  ]
        ])
        init_distr = np.array([0.3, 0.7])
        marchie = MarChie(init_distr=init_distr, trans_mat=trans_mat)
        
        self.assertIsInstance(marchie, MarChie)

    def test_check_marchie_2(self):
        
        trans_mat = np.array([
            [0.5,   0.2], # < 1
            [1,     0  ]
        ])

        with self.assertRaises(AssertionError): MarChie(trans_mat=trans_mat)

    def test_check_marchie_3(self):
        
        trans_mat = np.array([
            [0.5,   0.5],
            [1,     0.1] # > 1
        ])

        with self.assertRaises(AssertionError): MarChie(trans_mat=trans_mat)

    def test_check_marchie_4(self):
        
        trans_mat = np.array([
            [0.5,   0.5],
            [1,     0  ]
        ])
        init_distr = np.array([0.3, 0.3]) # < 1
        
        with self.assertRaises(AssertionError): MarChie(init_distr=init_distr, trans_mat=trans_mat)

    def test_check_marchie_5(self):
        
        trans_mat = np.array([
            [0.5,   0.5],
            [1,     0  ]
        ])
        init_distr = np.array([0, 2]) # > 1
        
        with self.assertRaises(AssertionError): MarChie(init_distr=init_distr, trans_mat=trans_mat)

    def test_check_marchie_6(self):
        
        trans_mat = np.array([
            [2,   -1], # invalid probs
            [1,    0]
        ])
        init_distr = np.array([0, 1])
        
        with self.assertRaises(AssertionError): MarChie(init_distr=init_distr, trans_mat=trans_mat)

    def test_check_marchie_7(self):
        
        trans_mat = np.array([
            [0.5,  0.5],
            [1,    0  ]
        ])
        init_distr = np.array([0, -1]) # invalid probs
        
        with self.assertRaises(AssertionError): MarChie(init_distr=init_distr, trans_mat=trans_mat)

    def test_check_marchie_8(self):
        
        trans_mat = np.array([
            [[0.5,   0.5], # 3D matrix
             [1,     0 ]]
        ])

        with self.assertRaises(AssertionError): MarChie(trans_mat=trans_mat)

    def test_check_marchie_9(self):
        
        trans_mat = np.array([
            [0.5,   0.5],
            [1,     0  ]
        ])
        init_distr = np.array([[0, 1]]) # 2D matrix

        with self.assertRaises(AssertionError): MarChie(init_distr=init_distr, trans_mat=trans_mat)

    def test_check_marchie_10(self):
        
        trans_mat = np.array([
            [0.5,   0.5], # not a square matrix
            [1,     0  ],
            [0.3,   0.7]
        ])

        with self.assertRaises(AssertionError): MarChie(trans_mat=trans_mat)

    def test_check_marchie_11(self):
        
        trans_mat = np.array([
            [0.5,   0.2,   0.3], # not a square matrix
            [0.9,   0,     0.1]
        ])

        with self.assertRaises(AssertionError): MarChie(trans_mat=trans_mat)

    def test_check_marchie_12(self):
        
        trans_mat = np.array([
            [0.5,   0.2], # different number of states
            [0.9,   0  ]
        ])
        init_distr = np.array([0, 0.4, 0.6])

        with self.assertRaises(AssertionError): MarChie(init_distr=init_distr, trans_mat=trans_mat)

    #endregion

    #region Communication Tests

    def test_adjacency_mat_0(self):
        
        trans_mat = np.array([
            [1,     0,     0  ],
            [0.8,   0.2,   0  ],
            [0.3,   0.5,   0.2]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_adjacency_mat = np.array([
            [1,     0,     0  ],
            [1,     1,     0  ],
            [1,     1,     1  ]
        ])
        actual_adjacency_mat = marchie.adjacency_mat.original

        self.assertTrue(
            np.array_equal(expected_adjacency_mat, actual_adjacency_mat)
        )
        self.assertIs(actual_adjacency_mat.dtype, np.dtype(int))

    def test_adjacency_mat_1(self):
        
        trans_mat = np.array([
            [1e-10,    1 - 1e-10,    0       ],
            [0,        1e-5,         1 - 1e-5],
            [0,        1,            0       ]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_adjacency_mat = np.array([
            [1,     1,     0],
            [0,     1,     1],
            [0,     1,     0]
        ])
        actual_adjacency_mat = marchie.adjacency_mat.original

        self.assertTrue(
            np.array_equal(expected_adjacency_mat, actual_adjacency_mat)
        )
        self.assertIs(actual_adjacency_mat.dtype, np.dtype(int))

    def test_reachability_mat_0(self):
        
        trans_mat = np.array([
            [0.01,   0.09,   0.9,    0   ],
            [0.03,   0.07,   0,      0.9 ],
            [0.7,    0,      0.21,   0.09],
            [0,      0.7,    0.03,   0.27]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_reachability_mat = np.array([
            [1,     1,     1,     1],
            [1,     1,     1,     1],
            [1,     1,     1,     1],
            [1,     1,     1,     1]
        ])
        actual_reachability_mat = marchie.reachability_mat.original

        self.assertTrue(
            np.array_equal(expected_reachability_mat, actual_reachability_mat)
        )
        self.assertIs(actual_reachability_mat.dtype, np.dtype(int))

    def test_reachability_mat_1(self):
        
        trans_mat = np.array([
            [1,     0,     0  ],
            [0.8,   0.2,   0  ],
            [0.3,   0.5,   0.2]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_reachability_mat = np.array([
            [1,     0,     0  ],
            [1,     1,     0  ],
            [1,     1,     1  ]
        ])
        actual_reachability_mat = marchie.reachability_mat.original

        self.assertTrue(
            np.array_equal(expected_reachability_mat, actual_reachability_mat)
        )
        self.assertIs(actual_reachability_mat.dtype, np.dtype(int))

    def test_reachability_mat_tr_0(self):
        
        trans_mat = np.array([
            [0.01,   0.09,   0.9,    0   ],
            [0.03,   0.07,   0,      0.9 ],
            [0.7,    0,      0.21,   0.09],
            [0,      0.7,    0.03,   0.27]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_reachability_mat_tr = np.array([
            [1,     1,     1,     1],
            [1,     1,     1,     1],
            [1,     1,     1,     1],
            [1,     1,     1,     1]
        ])
        actual_reachability_mat_tr = marchie.reachability_mat_tr.original

        self.assertTrue(
            np.array_equal(expected_reachability_mat_tr, actual_reachability_mat_tr)
        )
        self.assertIs(actual_reachability_mat_tr.dtype, np.dtype(int))

    def test_reachability_mat_tr_1(self):
        
        trans_mat = np.array([
            [1,     0,     0  ],
            [0.8,   0.2,   0  ],
            [0.3,   0.5,   0.2]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_reachability_mat_tr = np.array([
            [1,     1,     1],
            [0,     1,     1],
            [0,     0,     1]
        ])
        actual_reachability_mat_tr = marchie.reachability_mat_tr.original

        self.assertTrue(
            np.array_equal(expected_reachability_mat_tr, actual_reachability_mat_tr)
        )
        self.assertIs(actual_reachability_mat_tr.dtype, np.dtype(int))

    def test_communication_mat_0(self):
        
        trans_mat = np.array([
            [0.55,  0.45,  0],
            [0,     0,     1],
            [1,     0,     0]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_communication_mat = np.array([
            [1,     1,     1],
            [1,     1,     1],
            [1,     1,     1]
        ])
        actual_communication_mat = marchie.communication_mat.original

        self.assertTrue(
            np.array_equal(expected_communication_mat, actual_communication_mat)
        )
        self.assertIs(actual_communication_mat.dtype, np.dtype(int))

    def test_communication_mat_1(self):
        
        trans_mat = np.array([
            [1,     0,     0  ],
            [0.8,   0.2,   0  ],
            [0.3,   0.5,   0.2]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_communication_mat = np.array([
            [1,     0,     0],
            [0,     1,     0],
            [0,     0,     1]
        ])
        actual_communication_mat = marchie.communication_mat.original

        self.assertTrue(
            np.array_equal(expected_communication_mat, actual_communication_mat)
        )
        self.assertIs(actual_communication_mat.dtype, np.dtype(int))

    def test_communication_mat_comp_0(self):
        
        trans_mat = np.array([
            [0.55,  0.45,  0],
            [0,     0,     1],
            [1,     0,     0]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_communication_mat_comp = np.array([
            [0,     0,     0],
            [0,     0,     0],
            [0,     0,     0]
        ])
        actual_communication_mat_comp = marchie.communication_mat_comp.original

        self.assertTrue(
            np.array_equal(expected_communication_mat_comp, actual_communication_mat_comp)
        )
        self.assertIs(actual_communication_mat_comp.dtype, np.dtype(int))

    def test_communication_mat_comp_1(self):
        
        trans_mat = np.array([
            [1,     0,     0  ],
            [0.8,   0.2,   0  ],
            [0.3,   0.5,   0.2]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_communication_mat_comp = np.array([
            [0,     1,     1],
            [1,     0,     1],
            [1,     1,     0]
        ])
        actual_communication_mat_comp = marchie.communication_mat_comp.original

        self.assertTrue(
            np.array_equal(expected_communication_mat_comp, actual_communication_mat_comp)
        )
        self.assertIs(actual_communication_mat_comp.dtype, np.dtype(int))

    #endregion

    #region Classification Tests

    def test_classification_mat_0(self):
        
        trans_mat = np.array([
            [0.55,  0.45,  0],
            [0,     0,     1],
            [1,     0,     0]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_classification_mat = np.array([
            [0,     0,     0],
            [0,     0,     0],
            [0,     0,     0]
        ])
        actual_classification_mat = marchie.classification_mat.original

        self.assertTrue(
            np.array_equal(expected_classification_mat, actual_classification_mat)
        )
        self.assertIs(actual_classification_mat.dtype, np.dtype(int))

    def test_classification_mat_1(self):
        
        trans_mat = np.array([
            [1,     0,     0  ],
            [0.8,   0.2,   0  ],
            [0.3,   0.5,   0.2]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_classification_mat = np.array([
            [0,     0,     0],
            [1,     0,     0],
            [1,     1,     0]
        ])
        actual_classification_mat = marchie.classification_mat.original

        self.assertTrue(
            np.array_equal(expected_classification_mat, actual_classification_mat)
        )
        self.assertIs(actual_classification_mat.dtype, np.dtype(int))

    def test_classification_mat_ext_0(self):
        
        trans_mat = np.array([
            [0.55,  0.45,  0],
            [0,     0,     1],
            [1,     0,     0]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_classification_mat_ext = np.array([
            [0,     0,     0,     0],
            [0,     0,     0,     0],
            [0,     0,     0,     0]
        ])
        actual_classification_mat_ext = marchie.classification_mat_ext.original

        self.assertTrue(
            np.array_equal(expected_classification_mat_ext, actual_classification_mat_ext)
        )
        self.assertIs(actual_classification_mat_ext.dtype, np.dtype(int))

    def test_classification_mat_ext_1(self):
        
        trans_mat = np.array([
            [1,     0,     0  ],
            [0.8,   0.2,   0  ],
            [0.3,   0.5,   0.2]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_classification_mat_ext = np.array([
            [0,     0,     0,     0],
            [1,     0,     0,     1],
            [1,     1,     0,     1]
        ])
        actual_classification_mat_ext = marchie.classification_mat_ext.original

        self.assertTrue(
            np.array_equal(expected_classification_mat_ext, actual_classification_mat_ext)
        )
        self.assertIs(actual_classification_mat_ext.dtype, np.dtype(int))

    def test_equivalency_cls_mat_0(self):
        
        trans_mat = np.array([
            [0.55,  0.45,  0],
            [0,     0,     1],
            [1,     0,     0]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_equivalency_cls_mat = np.array([
            [1,     1,     1]
        ])
        actual_equivalency_cls_mat = marchie.equivalency_cls_mat.original

        self.assertTrue(
            np.array_equal(expected_equivalency_cls_mat, actual_equivalency_cls_mat)
        )
        self.assertIs(actual_equivalency_cls_mat.dtype, np.dtype(int))

    def test_equivalency_cls_mat_1(self):
        
        trans_mat = np.array([
            [1,     0,     0  ],
            [0.8,   0.2,   0  ],
            [0.3,   0.5,   0.2]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_equivalency_cls_mat = np.array([
            [1]
        ])
        actual_equivalency_cls_mat = marchie.equivalency_cls_mat.original

        self.assertTrue(
            np.array_equal(expected_equivalency_cls_mat, actual_equivalency_cls_mat)
        )
        self.assertIs(actual_equivalency_cls_mat.dtype, np.dtype(int))

    def test_equivalency_cls_mat_2(self):
        
        trans_mat = np.array([
            [1,     0,     0,     0,     0,     0,     0  ],
            [0,     0,     1,     0,     0,     0,     0  ],
            [0,     1,     0,     0,     0,     0,     0  ],
            [0,     0,     0,     0,     0,     0,     1  ],
            [0,     0,     0,     0,     0,     0.5,   0.5],
            [0,     0.5,   0,     0,     0.5,   0,     0  ],
            [0.5,   0.5,   0,     0,     0,     0,     0  ]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        expected_equivalency_cls_mat = np.array([
            [1,     0,     0],
            [0,     1,     1]
        ])
        actual_equivalency_cls_mat = marchie.equivalency_cls_mat.original

        self.assertTrue(
            np.array_equal(expected_equivalency_cls_mat, actual_equivalency_cls_mat)
        )
        self.assertIs(actual_equivalency_cls_mat.dtype, np.dtype(int))

    #endregion


class EndBehaviourTest(unittest.TestCase):

    '''
    ### Tests properties, structure, classification and end behavior of a Markov Chain.
    '''

    #region Ergodic Chains

    # ergodic regular chain 
    def test_example_ergodic_marchie_0(self):

        # create the test chain
        trans_mat = np.array([
            [0.5,    0.25,   0.25],
            [0.5,    0,      0.5 ],
            [0.25,   0.25,   0.5 ]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        # test canonical numbering: should be the same as we have 
        # a single cyclic subclass and no inessential states here
        expected_canonical_mapping = {0: 0, 1: 1, 2: 2}
        expected_canonical_trans_mat = np.array([
            [0.5,    0.25,   0.25],
            [0.5,    0,      0.5 ],
            [0.25,   0.25,   0.5 ]
        ])
        actual_canonical_mapping = marchie.canonical_mapping
        actual_canonical_trans_mat = marchie.trans_mat.canonical
        self.assertDictEqual(expected_canonical_mapping, actual_canonical_mapping)
        self.assertTrue(
            np.array_equal(expected_canonical_trans_mat, actual_canonical_trans_mat)
        )

        # test structure
        self.assertListEqual(marchie.structure.original_inessential_states, [])
        self.assertListEqual(marchie.structure.canonical_inessential_states, [])
        self.assertEqual(marchie.n_equivalency_classes, 1)
        self.assertEqual(marchie.n_cyclic_subclasses, 1)
        self.assertEqual(marchie.structure.equivalency_classes[0].d, 1)
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].original_states, [0, 1, 2])
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].canonical_states, [0, 1, 2])

        # test classification
        self.assertFalse(marchie.properties.reducible)
        self.assertFalse(marchie.properties.polyergodic)
        self.assertTrue(marchie.properties.regular)
        self.assertFalse(marchie.properties.absorbing)
        self.assertTrue(marchie.properties.strong_convergence)
        self.assertIs(marchie.end_behavior.__class__, ErgodicChainEndBehavior)
        
        # test end behavior
        expected_end_trans_mat = np.array([
            [0.4,    0.2,    0.4],
            [0.4,    0.2,    0.4],
            [0.4,    0.2,    0.4]
        ])
        expected_end_state_distr = np.array(
            [0.4,    0.2,    0.4]
        )
        expected_time_percentage = np.array(
            [40,     20,     40]
        )
        actual_end_trans_mat = marchie.end_behavior.end_trans_mat
        actual_end_state_distr = marchie.end_behavior.end_state_distr
        actual_time_percentage = marchie.end_behavior.time_percentage.round(2)
        self.assertTrue(
            np.allclose(expected_end_trans_mat, actual_end_trans_mat)
        )
        self.assertTrue(
            np.allclose(expected_end_state_distr, actual_end_state_distr)
        )
        self.assertTrue(
            np.allclose(expected_time_percentage, actual_time_percentage)
        )
        # here the matrix in a large number of steps is the best test
        # since it's actual empiric end values
        self.assertTrue(
            np.allclose(actual_end_trans_mat, marchie.trans_insteps(n_steps=100000))
        )
        self.assertTrue(
            np.allclose(actual_end_state_distr, marchie.states_insteps(n_steps=100000))
        )

    # ergodic cyclic chain
    def test_example_ergodic_marchie_1(self):

        # create the test chain
        trans_mat = np.array([
            [0,   1,     0,     0,     0  ],
            [0.3, 0,     0.7,   0,     0  ],
            [0,   0.3,   0,     0.7,   0  ],
            [0,   0,     0.3,   0,     0.7],
            [0,   0,     0,     1,     0  ],
        ])
        marchie = MarChie(trans_mat=trans_mat)

        # test canonical numbering
        expected_canonical_mapping = {0: 2, 1: 0, 2: 3, 3: 1, 4: 4}
        expected_canonical_trans_mat = np.array([
            [0,   0,     0.3,   0.7,   0  ],
            [0,   0,     0,     0.3,   0.7],
            [1,   0,     0,     0,     0  ],
            [0.3, 0.7,   0,     0,     0  ],
            [0,   1,     0,     0,     0  ]
        ])
        actual_canonical_mapping = marchie.canonical_mapping
        actual_canonical_trans_mat = marchie.trans_mat.canonical
        self.assertDictEqual(expected_canonical_mapping, actual_canonical_mapping)
        self.assertTrue(
            np.array_equal(expected_canonical_trans_mat, actual_canonical_trans_mat)
        )

        # test structure
        self.assertListEqual(marchie.structure.original_inessential_states, [])
        self.assertListEqual(marchie.structure.canonical_inessential_states, [])
        self.assertEqual(marchie.n_equivalency_classes, 1)
        self.assertEqual(marchie.n_cyclic_subclasses, 2)
        self.assertEqual(marchie.structure.equivalency_classes[0].d, 2)
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].original_states, [1, 3])
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].canonical_states, [0, 1])
        self.assertListEqual(marchie.structure.cyclic_subclasses[1].original_states, [0, 2, 4])
        self.assertListEqual(marchie.structure.cyclic_subclasses[1].canonical_states, [2, 3, 4])

        # test classification
        self.assertFalse(marchie.properties.reducible)
        self.assertFalse(marchie.properties.polyergodic)
        self.assertFalse(marchie.properties.regular)
        self.assertFalse(marchie.properties.absorbing)
        self.assertFalse(marchie.properties.strong_convergence)
        self.assertIs(marchie.end_behavior.__class__, ErgodicChainEndBehavior)
        
        # test end behavior
        expected_end_trans_mat = np.array([
            [0.07758621,   0.42241379,   0.02327586,   0.18103448,   0.29568966],
            [0.07758621,   0.42241379,   0.02327586,   0.18103448,   0.29568966],
            [0.07758621,   0.42241379,   0.02327586,   0.18103448,   0.29568966],
            [0.07758621,   0.42241379,   0.02327586,   0.18103448,   0.29568966],
            [0.07758621,   0.42241379,   0.02327586,   0.18103448,   0.29568966]
        ])
        expected_end_state_distr = np.array(
            [0.07758621,   0.42241379,   0.02327586,   0.18103448,   0.29568966]
        )
        expected_time_percentage = np.array(
            [7.76,   42.24,   2.33,   18.1,   29.57],
        )
        actual_end_trans_mat = marchie.end_behavior.end_trans_mat
        actual_end_state_distr = marchie.end_behavior.end_state_distr
        actual_time_percentage = marchie.end_behavior.time_percentage.round(2)
        self.assertTrue(
            np.allclose(expected_end_trans_mat, actual_end_trans_mat)
        )
        self.assertTrue(
            np.allclose(expected_end_state_distr, actual_end_state_distr)
        )
        self.assertTrue(
            np.allclose(expected_time_percentage, actual_time_percentage)
        )
        # convergence is weak
        self.assertFalse(
            np.allclose(actual_end_trans_mat, marchie.trans_insteps(n_steps=100000))
        )
        self.assertFalse(
            np.allclose(actual_end_state_distr, marchie.states_insteps(n_steps=100000))
        )

    #endregion

    #region Monoergodic Chains

    # monoergodic regular chain 
    def test_example_monoergodic_marchie_0(self):

        # create the test chain
        trans_mat = np.array([
            [0,      0.5,    0,      0.5,    0],
            [0.5,    0,      0,      0,    0.5],
            [0,      0,      0.5,    0,    0.5],
            [0,      0,      1,      0,      0],
            [0,      0,      0,      1,      0]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        # test canonical numbering
        expected_canonical_mapping = {0: 3, 1: 4, 2: 0, 3: 1, 4: 2}
        expected_canonical_trans_mat = np.array([
            [0.5,    0,      0.5,    0,      0],
            [1,      0,      0,      0,      0],
            [0,      1,      0,      0,      0],
            [0,      0.5,    0,      0,    0.5],
            [0,      0,      0.5,    0.5,    0]
        ])
        actual_canonical_mapping = marchie.canonical_mapping
        actual_canonical_trans_mat = marchie.trans_mat.canonical
        self.assertDictEqual(expected_canonical_mapping, actual_canonical_mapping)
        self.assertTrue(
            np.array_equal(expected_canonical_trans_mat, actual_canonical_trans_mat)
        )

        # test structure
        self.assertListEqual(marchie.structure.original_inessential_states, [0, 1])
        self.assertListEqual(marchie.structure.canonical_inessential_states, [3, 4])
        self.assertEqual(marchie.n_equivalency_classes, 1)
        self.assertEqual(marchie.n_cyclic_subclasses, 1)
        self.assertEqual(marchie.structure.equivalency_classes[0].d, 1)
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].original_states, [2, 3, 4])
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].canonical_states, [0, 1, 2])

        # test classification
        self.assertTrue(marchie.properties.reducible)
        self.assertFalse(marchie.properties.polyergodic)
        self.assertTrue(marchie.properties.regular)
        self.assertFalse(marchie.properties.absorbing)
        self.assertTrue(marchie.properties.strong_convergence)
        self.assertIs(marchie.end_behavior.__class__, MonoergodicChainEndBehavior)
        
        # test end behavior
        expected_end_trans_mat = np.array([
            [0.5,    0.25,    0.25,   0,     0],
            [0.5,    0.25,    0.25,   0,     0],
            [0.5,    0.25,    0.25,   0,     0],
            [0.5,    0.25,    0.25,   0,     0],
            [0.5,    0.25,    0.25,   0,     0]
        ])
        expected_end_state_distr = np.array(
            [0.5,    0.25,    0.25,   0,     0]
        )
        expected_time_percentage = np.array(
            [50,     25,      25,     0,     0]
        )
        actual_end_trans_mat = marchie.end_behavior.end_trans_mat
        actual_end_state_distr = marchie.end_behavior.end_state_distr
        actual_time_percentage = marchie.end_behavior.time_percentage.round(2)
        self.assertTrue(
            np.allclose(expected_end_trans_mat, actual_end_trans_mat)
        )
        self.assertTrue(
            np.allclose(expected_end_state_distr, actual_end_state_distr)
        )
        self.assertTrue(
            np.allclose(expected_time_percentage, actual_time_percentage)
        )
        # here the matrix in a large number of steps is the best test
        # since it's actual empiric end values
        self.assertTrue(
            np.allclose(actual_end_trans_mat, marchie.trans_insteps(n_steps=100000))
        )
        self.assertTrue(
            np.allclose(actual_end_state_distr, marchie.states_insteps(n_steps=100000))
        )

    # monoergodic cyclic chain 
    def test_example_monoergodic_marchie_0(self):

        # create the test chain
        trans_mat = np.array([
            [0.5,    0.5,    0,      0,      0,      0  ],
            [0.25,   0.25,   0.25,   0,      0.25,   0  ],
            [0,      0,      0,      0,      1,      0  ],
            [0,      0,      0.5,    0.5,    0,      0  ],
            [0,      0,      1,      0,      0,      0  ],
            [0,      0.25,   0,      0.25,   0.5,    0  ]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        # test canonical numbering
        expected_canonical_mapping = { 0: 2, 1: 3, 2: 0, 3: 4, 4: 1, 5: 5}
        expected_canonical_trans_mat = np.array([
            [0,      1,      0,      0,      0,      0  ],
            [1,      0,      0,      0,      0,      0  ],
            [0,      0,      0.5,    0.5,    0,      0  ],
            [0.25,   0.25,   0.25,   0.25,   0,      0  ],
            [0.5,    0,      0,      0,      0.5,    0  ],
            [0,      0.5,    0,      0.25,   0.25,   0  ]
        ])
        actual_canonical_mapping = marchie.canonical_mapping
        actual_canonical_trans_mat = marchie.trans_mat.canonical
        self.assertDictEqual(expected_canonical_mapping, actual_canonical_mapping)
        self.assertTrue(
            np.array_equal(expected_canonical_trans_mat, actual_canonical_trans_mat)
        )

        # test structure
        self.assertListEqual(marchie.structure.original_inessential_states, [0, 1, 3, 5])
        self.assertListEqual(marchie.structure.canonical_inessential_states, [2, 3, 4, 5])
        self.assertEqual(marchie.n_equivalency_classes, 1)
        self.assertEqual(marchie.n_cyclic_subclasses, 2)
        self.assertEqual(marchie.structure.equivalency_classes[0].d, 2)
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].original_states, [2])
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].canonical_states, [0])
        self.assertListEqual(marchie.structure.cyclic_subclasses[1].original_states, [4])
        self.assertListEqual(marchie.structure.cyclic_subclasses[1].canonical_states, [1])

        # test classification
        self.assertTrue(marchie.properties.reducible)
        self.assertFalse(marchie.properties.polyergodic)
        self.assertFalse(marchie.properties.regular)
        self.assertFalse(marchie.properties.absorbing)
        self.assertFalse(marchie.properties.strong_convergence)
        self.assertIs(marchie.end_behavior.__class__, MonoergodicChainEndBehavior)
        
        # test end behavior
        expected_end_trans_mat = np.array([
            [0.5,    0.5,     0,      0,     0,     0],
            [0.5,    0.5,     0,      0,     0,     0],
            [0.5,    0.5,     0,      0,     0,     0],
            [0.5,    0.5,     0,      0,     0,     0],
            [0.5,    0.5,     0,      0,     0,     0],
            [0.5,    0.5,     0,      0,     0,     0]
        ])
        expected_end_state_distr = np.array(
            [0.5,    0.5,     0,      0,     0,     0]
        )
        expected_time_percentage = np.array(
            [50,     50,      0,      0,     0,     0]
        )
        actual_end_trans_mat = marchie.end_behavior.end_trans_mat
        actual_end_state_distr = marchie.end_behavior.end_state_distr
        actual_time_percentage = marchie.end_behavior.time_percentage.round(2)
        self.assertTrue(
            np.allclose(expected_end_trans_mat, actual_end_trans_mat)
        )
        self.assertTrue(
            np.allclose(expected_end_state_distr, actual_end_state_distr)
        )
        self.assertTrue(
            np.allclose(expected_time_percentage, actual_time_percentage)
        )
        # convergence is weak
        self.assertFalse(
            np.allclose(actual_end_trans_mat, marchie.trans_insteps(n_steps=100000))
        )
        self.assertFalse(
            np.allclose(actual_end_state_distr, marchie.states_insteps(n_steps=100000))
        )

    # monoergodic absorbing chain 
    def test_example_monoergodic_marchie_1(self):

        # create the test chain
        trans_mat = np.array([
            [1,      0,      0,      0,      0   ],
            [0.77,   0,      0.23,   0,      0   ],
            [0,      0.77,   0,      0.23,   0   ],
            [0,      0,      0.77,   0,      0.23],
            [0,      0,      0,      1,      0   ]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        # test canonical numbering
        expected_canonical_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        expected_canonical_trans_mat = np.array([
            [1,      0,      0,      0,      0   ],
            [0.77,   0,      0.23,   0,      0   ],
            [0,      0.77,   0,      0.23,   0   ],
            [0,      0,      0.77,   0,      0.23],
            [0,      0,      0,      1,      0   ]
        ])
        actual_canonical_mapping = marchie.canonical_mapping
        actual_canonical_trans_mat = marchie.trans_mat.canonical
        self.assertDictEqual(expected_canonical_mapping, actual_canonical_mapping)
        self.assertTrue(
            np.array_equal(expected_canonical_trans_mat, actual_canonical_trans_mat)
        )

        # test structure
        self.assertListEqual(marchie.structure.original_inessential_states, [1, 2, 3, 4])
        self.assertListEqual(marchie.structure.canonical_inessential_states, [1, 2, 3, 4])
        self.assertEqual(marchie.n_equivalency_classes, 1)
        self.assertEqual(marchie.n_cyclic_subclasses, 1)
        self.assertEqual(marchie.structure.equivalency_classes[0].d, 1)
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].original_states, [0])
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].canonical_states, [0])

        # test classification
        self.assertTrue(marchie.properties.reducible)
        self.assertFalse(marchie.properties.polyergodic)
        self.assertTrue(marchie.properties.regular)
        self.assertTrue(marchie.properties.absorbing)
        self.assertTrue(marchie.properties.strong_convergence)
        self.assertIs(marchie.end_behavior.__class__, MonoergodicChainEndBehavior)
        
        # test end behavior
        expected_end_trans_mat = np.array([
            [1,      0,      0,      0,      0],
            [1,      0,      0,      0,      0],
            [1,      0,      0,      0,      0],
            [1,      0,      0,      0,      0],
            [1,      0,      0,      0,      0]
        ])
        expected_end_state_distr = np.array(
            [1,      0,      0,      0,      0]
        )
        expected_time_percentage = np.array(
            [100,    0,      0,      0,      0]
        )
        actual_end_trans_mat = marchie.end_behavior.end_trans_mat
        actual_end_state_distr = marchie.end_behavior.end_state_distr
        actual_time_percentage = marchie.end_behavior.time_percentage.round(2)
        self.assertTrue(
            np.allclose(expected_end_trans_mat, actual_end_trans_mat)
        )
        self.assertTrue(
            np.allclose(expected_end_state_distr, actual_end_state_distr)
        )
        self.assertTrue(
            np.allclose(expected_time_percentage, actual_time_percentage)
        )
        # here the matrix in a large number of steps is the best test
        # since it's actual empiric end values
        self.assertTrue(
            np.allclose(actual_end_trans_mat, marchie.trans_insteps(n_steps=100000))
        )
        self.assertTrue(
            np.allclose(actual_end_state_distr, marchie.states_insteps(n_steps=100000))
        )

    #endregion

    #region Polyergodic Chains

    # polyergodic regular chain 
    def test_example_polyergodic_marchie_0(self):

        # create the test chain
        trans_mat = np.array([            
            [1/3,   0,     0,     0,     0,     2/3,   0,     0,     0  ],   
            [0,     0,     0.5,   0,     0,     0,     0,     0.5,   0  ],   
            [0,     0.5,   0,     0,     0,     0,     0,     0.5,   0  ],
            [0,     0,     0,     1,     0,     0,     0,     0,     0  ],   
            [0,     0,     1/3,   1/3,   0,     0,     0,     1/3,   0  ],   
            [1,     0,     0,     0,     0,     0,     0,     0,     0  ],
            [1/3,   0,     0,     0,     1/3,   0,     0,     0,     1/3],   
            [0,     0.5,   0.5,   0,     0,     0,     0,     0,     0  ],   
            [1/3,   0,     0,     0,     0,     1/3,   1/3,   0,     0  ]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        # test canonical numbering
        expected_canonical_mapping = {0: 0, 1: 2, 2: 3, 3: 5, 4: 6, 5: 1, 6: 7, 7: 4, 8: 8}
        expected_canonical_trans_mat = np.array([
            [1/3,   2/3,   0,     0,     0,     0,     0,     0,     0  ],   
            [1,     0,     0,     0,     0,     0,     0,     0,     0  ],   
            [0,     0,     0,     0.5,   0.5,   0,     0,     0,     0  ],
            [0,     0,     0.5,   0,     0.5,   0,     0,     0,     0  ],   
            [0,     0,     0.5,   0.5,   0,     0,     0,     0,     0  ],   
            [0,     0,     0,     0,     0,     1,     0,     0,     0  ],
            [0,     0,     0,     1/3,   1/3,   1/3,   0,     0,     0  ],   
            [1/3,   0,     0,     0,     0,     0,     1/3,   0,     1/3],   
            [1/3,   1/3,   0,     0,     0,     0,     0,     1/3,   0  ]
        ])
        actual_canonical_mapping = marchie.canonical_mapping
        actual_canonical_trans_mat = marchie.trans_mat.canonical
        self.assertDictEqual(expected_canonical_mapping, actual_canonical_mapping)
        self.assertTrue(
            np.array_equal(expected_canonical_trans_mat, actual_canonical_trans_mat)
        )

        # test structure
        self.assertListEqual(marchie.structure.original_inessential_states, [4, 6, 8])
        self.assertListEqual(marchie.structure.canonical_inessential_states, [6, 7, 8])
        self.assertEqual(marchie.n_equivalency_classes, 3)
        self.assertEqual(marchie.n_cyclic_subclasses, 3)
        self.assertEqual(marchie.structure.equivalency_classes[0].d, 1)
        self.assertEqual(marchie.structure.equivalency_classes[1].d, 1)
        self.assertEqual(marchie.structure.equivalency_classes[2].d, 1)
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].original_states, [0, 5])
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].canonical_states, [0, 1])
        self.assertListEqual(marchie.structure.cyclic_subclasses[1].original_states, [1, 2, 7])
        self.assertListEqual(marchie.structure.cyclic_subclasses[1].canonical_states, [2, 3, 4])
        self.assertListEqual(marchie.structure.cyclic_subclasses[2].original_states, [3])
        self.assertListEqual(marchie.structure.cyclic_subclasses[2].canonical_states, [5])

        # test classification
        self.assertTrue(marchie.properties.reducible)
        self.assertTrue(marchie.properties.polyergodic)
        self.assertTrue(marchie.properties.regular)
        self.assertFalse(marchie.properties.absorbing)
        self.assertTrue(marchie.properties.strong_convergence)
        self.assertIs(marchie.end_behavior.__class__, PolyergodicChainEndBehavior)
        
        # test end behavior
        expected_end_trans_mat = np.array([
            [0.6,   0.4,   0,     0,     0,     0,     0,     0,     0  ],   
            [0.6,   0.4,   0,     0,     0,     0,     0,     0,     0  ],   
            [0,     0,     1/3,   1/3,   1/3,   0,     0,     0,     0  ],
            [0,     0,     1/3,   1/3,   1/3,   0,     0,     0,     0  ],   
            [0,     0,     1/3,   1/3,   1/3,   0,     0,     0,     0  ],   
            [0,     0,     0,     0,     0,     1,     0,     0,     0  ],
            [0,     0,     2/9,   2/9,   2/9,   1/3,   0,     0,     0  ],   
            [3/8,   0.25,  1/12,  1/12,  1/12,  0.125, 0,     0,     0  ],   
            [21/40, 14/40, 1/36,  1/36,  1/36,  1/24,  0,     0,     0  ]
        ])
        actual_end_trans_mat = marchie.end_behavior.end_trans_mat
        actual_end_state_distr = marchie.end_behavior.end_state_distr
        self.assertTrue(
            np.allclose(expected_end_trans_mat, actual_end_trans_mat)
        )
        self.assertIsNone(marchie.end_behavior.time_percentage) # not supposed for polyergodic chains
        # here the matrix in a large number of steps is the best test
        # since it's actual empiric end values
        self.assertTrue(
            np.allclose(actual_end_trans_mat, marchie.trans_insteps(n_steps=100000))
        )
        self.assertTrue(
            np.allclose(actual_end_state_distr, marchie.states_insteps(n_steps=100000))
        )

    # polyergodic cyclic chain 
    def test_example_polyergodic_marchie_1(self):

        # create the test chain
        trans_mat = np.array([            
            [0,     0.3,   0.7,   0,     0,     0,     0,     0,     0,     0,     0],
            [0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0],
            [1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
            [0,     0,     0,     0,     0.2,   0,     0,     0.8,   0,     0,     0],
            [0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0],
            [0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0],
            [0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0],
            [0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0],
            [0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0],
            [0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1],
            [0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        # test canonical numbering
        expected_canonical_mapping = {0: 7, 1: 8, 2: 9, 3: 10, 4: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6}
        expected_canonical_trans_mat = np.array([
            [0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0],
            [0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0],
            [1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
            [0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0],
            [0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0],
            [0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0],
            [0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0],
            [0,     0,     0,     0,     0,     0,     0,     0,     0.3,   0.7,   0],
            [0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1],
            [0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0],
            [0.2,   0,     0,     0.8,   0,     0,     0,     0,     0,     0,     0]
        ])
        actual_canonical_mapping = marchie.canonical_mapping
        actual_canonical_trans_mat = marchie.trans_mat.canonical
        self.assertDictEqual(expected_canonical_mapping, actual_canonical_mapping)
        self.assertTrue(
            np.array_equal(expected_canonical_trans_mat, actual_canonical_trans_mat)
        )

        # test structure
        self.assertListEqual(marchie.structure.original_inessential_states, [0, 1, 2, 3])
        self.assertListEqual(marchie.structure.canonical_inessential_states, [7, 8, 9, 10])
        self.assertEqual(marchie.n_equivalency_classes, 2)
        self.assertEqual(marchie.n_cyclic_subclasses, 7)
        self.assertEqual(marchie.structure.equivalency_classes[0].d, 3)
        self.assertEqual(marchie.structure.equivalency_classes[1].d, 4)
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].original_states, [4])
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].canonical_states, [0])
        self.assertListEqual(marchie.structure.cyclic_subclasses[1].original_states, [5])
        self.assertListEqual(marchie.structure.cyclic_subclasses[1].canonical_states, [1])
        self.assertListEqual(marchie.structure.cyclic_subclasses[2].original_states, [6])
        self.assertListEqual(marchie.structure.cyclic_subclasses[2].canonical_states, [2])
        self.assertListEqual(marchie.structure.cyclic_subclasses[3].original_states, [7])
        self.assertListEqual(marchie.structure.cyclic_subclasses[3].canonical_states, [3])
        self.assertListEqual(marchie.structure.cyclic_subclasses[4].original_states, [8])
        self.assertListEqual(marchie.structure.cyclic_subclasses[4].canonical_states, [4])
        self.assertListEqual(marchie.structure.cyclic_subclasses[5].original_states, [9])
        self.assertListEqual(marchie.structure.cyclic_subclasses[5].canonical_states, [5])
        self.assertListEqual(marchie.structure.cyclic_subclasses[6].original_states, [10])
        self.assertListEqual(marchie.structure.cyclic_subclasses[6].canonical_states, [6])

        # test classification
        self.assertTrue(marchie.properties.reducible)
        self.assertTrue(marchie.properties.polyergodic)
        self.assertFalse(marchie.properties.regular)
        self.assertFalse(marchie.properties.absorbing)
        self.assertFalse(marchie.properties.strong_convergence)
        self.assertIs(marchie.end_behavior.__class__, PolyergodicChainEndBehavior)
        
        # test end behavior
        expected_end_trans_mat = np.array([
            [1/3,   1/3,   1/3,   0,     0,     0,     0,     0,     0,     0,     0],
            [1/3,   1/3,   1/3,   0,     0,     0,     0,     0,     0,     0,     0],
            [1/3,   1/3,   1/3,   0,     0,     0,     0,     0,     0,     0,     0],
            [0,     0,     0,     0.25,  0.25,  0.25,  0.25,  0,     0,     0,     0],
            [0,     0,     0,     0.25,  0.25,  0.25,  0.25,  0,     0,     0,     0],
            [0,     0,     0,     0.25,  0.25,  0.25,  0.25,  0,     0,     0,     0],
            [0,     0,     0,     0.25,  0.25,  0.25,  0.25,  0,     0,     0,     0],
            [1/15,  1/15,  1/15,  0.2,   0.2,   0.2,   0.2,   0,     0,     0,     0],
            [1/15,  1/15,  1/15,  0.2,   0.2,   0.2,   0.2,   0,     0,     0,     0],
            [1/15,  1/15,  1/15,  0.2,   0.2,   0.2,   0.2,   0,     0,     0,     0],
            [1/15,  1/15,  1/15,  0.2,   0.2,   0.2,   0.2,   0,     0,     0,     0]
        ])
        actual_end_trans_mat = marchie.end_behavior.end_trans_mat
        actual_end_state_distr = marchie.end_behavior.end_state_distr
        self.assertTrue(
            np.allclose(expected_end_trans_mat, actual_end_trans_mat)
        )
        self.assertIsNone(marchie.end_behavior.time_percentage) # not supposed for polyergodic chains
        # convergence is weak
        self.assertFalse(
            np.allclose(actual_end_trans_mat, marchie.trans_insteps(n_steps=100000))
        )
        self.assertFalse(
            np.allclose(actual_end_state_distr, marchie.states_insteps(n_steps=100000))
        )

    # polyergodic absorbing chain 
    def test_example_polyergodic_absorbing_marchie_0(self):

         # create the test chain
        trans_mat = np.array([            
            [1,    0,     0,     0,     0  ],
            [0.3,  0,     0.7,   0,     0  ],
            [0,    0.3,   0,     0.7,   0  ],
            [0,    0,     0.3,   0,     0.7],
            [0,    0,     0,     0,     1  ]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        # test canonical numbering
        expected_canonical_mapping = {0: 0, 1: 2, 2: 3, 3: 4, 4: 1}
        expected_canonical_trans_mat = np.array([
            [1,    0,     0,     0,     0  ],
            [0,    1,     0,     0,     0  ],
            [0.3,  0,     0,     0.7,   0  ],
            [0,    0,     0.3,   0,     0.7],
            [0,    0.7,   0,     0.3,   0  ]
        ])
        actual_canonical_mapping = marchie.canonical_mapping
        actual_canonical_trans_mat = marchie.trans_mat.canonical
        self.assertDictEqual(expected_canonical_mapping, actual_canonical_mapping)
        self.assertTrue(
            np.array_equal(expected_canonical_trans_mat, actual_canonical_trans_mat)
        )

        # test structure
        self.assertListEqual(marchie.structure.original_inessential_states, [1, 2, 3])
        self.assertListEqual(marchie.structure.canonical_inessential_states, [2, 3, 4])
        self.assertEqual(marchie.n_equivalency_classes, 2)
        self.assertEqual(marchie.n_cyclic_subclasses, 2)
        self.assertEqual(marchie.structure.equivalency_classes[0].d, 1)
        self.assertEqual(marchie.structure.equivalency_classes[1].d, 1)
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].original_states, [0])
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].canonical_states, [0])
        self.assertListEqual(marchie.structure.cyclic_subclasses[1].original_states, [4])
        self.assertListEqual(marchie.structure.cyclic_subclasses[1].canonical_states, [1])

        # test classification
        self.assertTrue(marchie.properties.reducible)
        self.assertTrue(marchie.properties.polyergodic)
        self.assertTrue(marchie.properties.regular)
        self.assertTrue(marchie.properties.absorbing)
        self.assertTrue(marchie.properties.strong_convergence)
        self.assertIs(marchie.end_behavior.__class__, PolyergodicAbsorbingChainEndBehavior)
        
        # test end behavior
        expected_end_trans_mat = np.array([
            [1,           0,            0,     0,     0],
            [0,           1,            0,     0,     0],
            [0.40862069,  0.59137931,   0,     0,     0],
            [0.15517241,  0.84482759,   0,     0,     0],
            [0.04655172,  0.95344828,   0,     0,     0]
        ])
        actual_end_trans_mat = marchie.end_behavior.end_trans_mat
        actual_end_state_distr = marchie.end_behavior.end_state_distr
        self.assertTrue(
            np.allclose(expected_end_trans_mat, actual_end_trans_mat)
        )
        self.assertIsNone(marchie.end_behavior.time_percentage) # not supposed for polyergodic chains
        # here the matrix in a large number of steps is the best test
        # since it's actual empiric end values
        self.assertTrue(
            np.allclose(actual_end_trans_mat, marchie.trans_insteps(n_steps=100000))
        )
        self.assertTrue(
            np.allclose(actual_end_state_distr, marchie.states_insteps(n_steps=100000))
        )

    # polyergodic absorbing chain 
    def test_example_polyergodic_absorbing_marchie_1(self):

        # create the test chain
        trans_mat = np.array([            
            [1/3,  0,     0,     0,     1/3,   1/3],
            [1/3,  1/3,   0,     0,     1/3,   0  ],
            [0,    1/3,   1/3,   0,     1/3,   0  ], 
            [0,    0,     1/3,   1/3,   1/3,   0  ],
            [0,    0,     0,     0,     1,     0  ],
            [0,    0,     0,     0,     0,     1  ]
        ])
        marchie = MarChie(trans_mat=trans_mat)

        # test canonical numbering
        expected_canonical_mapping = {4: 0, 5: 1, 0: 2, 1: 3, 2: 4, 3: 5}
        expected_canonical_trans_mat = np.array([
            [1,      0,     0,     0,     0,     0  ],
            [0,      1,     0,     0,     0,     0  ],
            [1/3,    1/3,   1/3,   0,     0,     0  ], 
            [1/3,    0,     1/3,   1/3,   0,     0  ],
            [1/3,    0,     0,     1/3,   1/3,   0  ],
            [1/3,    0,     0,     0,     1/3,   1/3]
        ])
        actual_canonical_mapping = marchie.canonical_mapping
        actual_canonical_trans_mat = marchie.trans_mat.canonical
        self.assertDictEqual(expected_canonical_mapping, actual_canonical_mapping)
        self.assertTrue(
            np.array_equal(expected_canonical_trans_mat, actual_canonical_trans_mat)
        )

        # test structure
        self.assertListEqual(marchie.structure.original_inessential_states, [0, 1, 2, 3])
        self.assertListEqual(marchie.structure.canonical_inessential_states, [2, 3, 4, 5])
        self.assertEqual(marchie.n_equivalency_classes, 2)
        self.assertEqual(marchie.n_cyclic_subclasses, 2)
        self.assertEqual(marchie.structure.equivalency_classes[0].d, 1)
        self.assertEqual(marchie.structure.equivalency_classes[1].d, 1)
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].original_states, [4])
        self.assertListEqual(marchie.structure.cyclic_subclasses[0].canonical_states, [0])
        self.assertListEqual(marchie.structure.cyclic_subclasses[1].original_states, [5])
        self.assertListEqual(marchie.structure.cyclic_subclasses[1].canonical_states, [1])

        # test classification
        self.assertTrue(marchie.properties.reducible)
        self.assertTrue(marchie.properties.polyergodic)
        self.assertTrue(marchie.properties.regular)
        self.assertTrue(marchie.properties.absorbing)
        self.assertTrue(marchie.properties.strong_convergence)
        self.assertIs(marchie.end_behavior.__class__, PolyergodicAbsorbingChainEndBehavior)
        
        # test end behavior
        expected_end_trans_mat = np.array([
            [1,      0,     0,     0,     0,     0  ],
            [0,      1,     0,     0,     0,     0  ],
            [0.5,    0.5,   0,     0,     0,     0  ], 
            [0.75,   0.25,  0,     0,     0,     0  ],
            [0.875,  0.125, 0,     0,     0,     0  ],
            [0.9375, 0.0625,0,     0,     0,     0  ]
        ])
        actual_end_trans_mat = marchie.end_behavior.end_trans_mat
        actual_end_state_distr = marchie.end_behavior.end_state_distr
        self.assertTrue(
            np.allclose(expected_end_trans_mat, actual_end_trans_mat)
        )
        self.assertIsNone(marchie.end_behavior.time_percentage) # not supposed for polyergodic chains
        # here the matrix in a large number of steps is the best test
        # since it's actual empiric end values
        self.assertTrue(
            np.allclose(actual_end_trans_mat, marchie.trans_insteps(n_steps=100000))
        )
        self.assertTrue(
            np.allclose(actual_end_state_distr, marchie.states_insteps(n_steps=100000))
        )

    #endregion

#endregion


#region Executive

if __name__ == '__main__':
    unittest.main()

#endregion