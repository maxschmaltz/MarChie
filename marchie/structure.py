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

from dataclasses import dataclass, field
import re
from typing import List

#endregion


#region States

@dataclass(frozen=True, kw_only=True)
class State:

    '''
    ### Basic class for a Markov Chain state.

    Parameters
    ----------
    id : `int`
        number of the state in original numbering

    canonical_number : `int`
        number of the state in canonical numbering

    essential : `bool`
        whether the state is essential

    equivalency_class : `int`
        equivalency class the state belongs to

    cyclic_subclass : `int`
        cyclic subclass the state belongs to
    '''

    id:                 int
    canonical_number:   int
    essential:          bool
    equivalency_class:  int
    cyclic_subclass:    int

    def __repr__(self) -> str:
        return f'State {self.id}, canonical number: {self.canonical_number}'
    
    def __str__(self) -> str:
        return self.__repr__()


@dataclass(frozen=True, kw_only=True, repr=False)
class EssentialState(State):

    '''
    ### Class for a Markov Chain essential state.

    Parameters
    ----------
    id : `int`
        number of the state in original numbering

    canonical_number : `int`
        number of the state in canonical numbering

    equivalency_class : `int`
        equivalency class the state belongs to

    cyclic_subclass : `int`
        cyclic subclass the state belongs to
    '''

    essential: bool = field(init=False, default=True)


@dataclass(frozen=True, kw_only=True, repr=False)
class InessentialState(State):

    '''
    ### Class for a Markov Chain inessential state.

    Parameters
    ----------
    id : `int`
        number of the state in original numbering

    canonical_number : `int`
        number of the state in canonical numbering
    '''

    essential:          bool = field(init=False, default=False)
    equivalency_class:  int = field(init=False, default=None)
    cyclic_subclass:    int = field(init=False, default=None)

#endregion


#region Equivalency Classes

@dataclass(frozen=True, kw_only=True)
class CyclicSubclass:

    '''
    ### Class for a Markov Chain cyclic subclass.

    Parameters
    ----------
    id : `int`
        number of the subclass

    class_id : `int`
        equivalency class the state belongs to

    states : `list[EssentialState]`
        states that belong to the subclass
    '''

    id:         int
    class_id:   int
    states:     List[EssentialState]

    @property
    def original_states(self) -> List[int]:

        '''
        ### Returns a list of the numbers of its states in original numbering.

        Returns
        -------
        ids : `list[int]`
            list of the numbers of the states that belong to the subclass
        '''

        ids = [state.id for state in self.states]
        return ids

    @property
    def canonical_states(self) -> List[int]:

        '''
        ### Returns a list of the numbers of its states in canonical numbering.

        Returns
        -------
        ids : `list[int]`
            list of the numbers of the states that belong to the subclass
        '''

        return [state.canonical_number for state in self.states]
    
    def __len__(self) -> int:
        return len(self.states)

    def __repr__(self) -> str:
        return f'Cyclic Subclass {self.id}: states {", ".join([str(state.id) for state in self.states])}'
    
    def __str__(self) -> str:
        return self.__repr__()


@dataclass(frozen=True, kw_only=True)
class EquivalenceClass:

    '''
    ### Class for a Markov Chain equivalence class.

    Parameters
    ----------
    id : `int`
        number of the subclass

    d : `int`
        period of the class (i.e. the number of cyclic subclasses in it)

    states : `list[EssentialState]`
        states that belong to the subclass

    cyclic_subclasses : `list[CyclicSubclass]` or `None`
        list of cyclic subclasses that are in this class if `d` > 1,
        `None` otherwise
    '''

    id:                 int
    d:                  int
    states:             List[EssentialState]
    cyclic_subclasses:  List[CyclicSubclass] | None

    @property
    def original_states(self) -> List[int]:

        '''
        ### Returns a list of the numbers of its states in original numbering.

        Returns
        -------
        ids : `list[int]`
            list of the numbers of the states that belong to the class
        '''

        ids = [state.id for state in self.states]
        return ids

    @property
    def canonical_states(self) -> List[int]:

        '''
        ### Returns a list of the numbers of its states in canonical numbering.

        Returns
        -------
        ids : `list[int]`
            list of the numbers of the states that belong to the class
        '''

        return [state.canonical_number for state in self.states]

    def __len__(self) -> int:
        return len(self.states)

    def __repr__(self) -> str:

        if self.cyclic_subclasses is not None:
            prefix = 'Cyclic'
            tree = '\n' + ''.join(f'|\n|___{repr(cyclic_subclass)}\n' for cyclic_subclass in self.cyclic_subclasses)
        else: 
            prefix = 'Acyclic'
            tree = f' : states {", ".join([str(state.id) for state in self.states])}'

        return f'{prefix} Equivalency Class {self.id}{tree.rstrip()}'
    
    def __str__(self) -> str:
        return self.__repr__()
    
#endregion


@dataclass(frozen=True, kw_only=True)
class ChainStructure:

    '''
    ### Class for a Markov Chain structure.

    Parameters
    ----------
    states : `list[State]`
        states that there are in the system

    essential_states : `list[EssentialState]`
        essential states that there are in the system

    inessential_states : `list[InessentialState]`
        inessential states that there are in the system

    equivalency_classes : `list[EquivalenceClass]`
        equivalency classes of the system

    cyclic_subclasses : `list[CyclicSubclass]`
        cyclic subclasses of the system
    '''

    states:                 List[State]
    essential_states:       List[EssentialState]
    inessential_states:     List[InessentialState]
    equivalency_classes:    List[EquivalenceClass]
    cyclic_subclasses:      List[CyclicSubclass]

    @property
    def original_states(self) -> List[int]:

        '''
        ### Returns a list of the numbers of its states in original numbering.

        Returns
        -------
        ids : `list[int]`
            list of the numbers of the states that belong to the class
        '''

        ids = [state.id for state in self.states]
        return ids

    @property
    def canonical_states(self) -> List[int]:

        '''
        ### Returns a list of the numbers of its states in canonical numbering.

        Returns
        -------
        ids : `list[int]`
            list of the numbers of the states that belong to the class
        '''

        ids = [state.canonical_number for state in self.states]
        return ids
    
    @property
    def original_essential_states(self) -> List[int]:

        '''
        ### Returns a list of the numbers of its essential states in original numbering.

        Returns
        -------
        ids : `list[int]`
            list of the numbers of the essential states that belong to the class
        '''

        ids = [state.id for state in self.essential_states]
        return ids

    @property
    def canonical_essential_states(self) -> List[int]:

        '''
        ### Returns a list of the numbers of its essential states in canonical numbering.

        Returns
        -------
        ids : `list[int]`
            list of the numbers of the essential states that belong to the class
        '''

        ids = [state.canonical_number for state in self.essential_states]
        return ids
    
    @property
    def original_inessential_states(self) -> List[int]:

        '''
        ### Returns a list of the numbers of its inessential states in original numbering.

        Returns
        -------
        ids : `list[int]`
            list of the numbers of the inessential states that belong to the class
        '''

        ids = [state.id for state in self.inessential_states]
        return ids

    @property
    def canonical_inessential_states(self) -> List[int]:

        '''
        ### Returns a list of the numbers of its inessential states in canonical numbering.

        Returns
        -------
        ids : `list[int]`
            list of the numbers of the inessential states that belong to the class
        '''

        ids = [state.canonical_number for state in self.inessential_states]
        return ids

    def __len__(self) -> int:
        return len(self.states)

    def __repr__(self) -> str:
        ws = ' ' * 4
        col = re.compile('\|')
        if len(self.essential_states):
            ess_tree = f'Essential States\n{ws}' + ''.join(f'|\n{ws}|___{col.sub(f"{ws}{ws}|", repr(equivalency_class))}\n' \
                       for equivalency_class in self.equivalency_classes)
        else: ess_tree = 'Essential States: None'
        if len(self.inessential_states):
            iness_tree =  f'Inessential States: states {", ".join([str(state.id) for state in self.inessential_states])}'
        else: iness_tree = 'Inessential States: None'
        tree = f'\n|\n|___{ess_tree}|\n|___{iness_tree}'
        return f'Markov Chain{tree}'
    
    def __str__(self) -> str:
        return self.__repr__()
    

@dataclass(frozen=True, kw_only=True)
class ChainProperties:

    '''
    ### Class containing properties of a Markov Chain.

    Parameters
    ----------
    reducible : `bool`
        if the chain is reducible (i.e. has inessential states)

    polyergodic : `bool`
        if the chain is polyergodic (i.e. has several equivalency classes)

    regular : `bool` or `None`
        `True` if all the equivalency classes of the chain are acyclic;
        `False` if all the equivalency classes of the chain are cyclic;
        `None` if there are both cyclic and acyclic equivalency classes

    absorbing : `bool`
        whether all the equivalency classes of the chain are absorbing

    strong_convergence : `bool`
        whether the transition matrix converges to the end transition matrix strongly
    '''

    reducible:          bool
    polyergodic:        bool
    regular:            bool | None
    absorbing:          bool
    strong_convergence: bool

    @property
    def type_label(self) -> str:

        '''
        ### Returns label for the type of the chain based on its properties.

        Returns
        -------
        type_label : `str`
            2-place label, where "Ergodic", "Monoergodic" or "Polyergodic" stands at the first place,
            "Absorbing", "Regular", "Mixed" or "Cyclic" at the second one
        '''

        type_label = ''
        type_label += 'Ergodic' if not self.reducible else 'Monoergodic' if not self.polyergodic else 'Polyergodic'
        type_label += ' Absorbing' if self.absorbing else ' Regular' if self.regular \
                      else ' Mixed' if self.regular is None else ' Cyclic'
        return type_label

    def __repr__(self) -> str:
        props = []
        for attr, _ in self.__annotations__.items():
            value = getattr(self, attr)
            if value is None: sym = '±'
            else: sym = '+' if value else '-'
            prop = f'[{sym}{attr}]'
            props.append(prop)
        return ''.join(props)

    def __str__(self) -> str:
        return self.__repr__()
