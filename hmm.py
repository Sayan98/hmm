"""Hidden Markov Model"""

from collections import Counter
import copy
import numpy as np
import pprint
from typing import List, Tuple


class MM:
    def __init__(self, states: List, t_mat: List[Tuple]):
        self.states = states
        self.t_mat = t_mat


    def next(self, state):
        """State transistion for the HMM"""
        next_states, transition_prob = zip(*self.t_mat[state].items())
        
        return np.random.choice(next_states, p=transition_prob)


    def _test():
        states = ['a', 'b', 'c']
        t_mat = {
                    'a': {'a': 0.3, 'b': 0.5, 'c': 0.2},
                    'b': {'a': 0.1, 'b': 0.1, 'c': 0.8},
                    'c': {'a': 0.2, 'b': 0.5, 'c': 0.3}
                }

        markov_model = MM(states, t_mat)

        present_state = 'a'
        
        for _ in range(10):
            print(" → " + present_state, end='')
            present_state = markov_model.next(state=present_state)


class HMM():
    def __init__(self, corpus: str):
        self.corpus = self.preprocess_corpus(corpus)
        
        self.characters = set(self.corpus)
        self.t_mat = self.get_transistion_probability()

        self.mm = MM(states=self.characters,
                     t_mat=self.t_mat)

    def get_transistion_probability(self):
        transition_counts = dict([(ps,dict([(ns, 0) for ns in self.characters])) for ps in self.characters])
        character_counts = Counter(self.corpus)
        
        for idx, cur_char in enumerate(self.corpus):
            next_char = self.corpus[idx+1] if idx != len(self.corpus) - 1 else self.corpus[0]
            transition_counts[cur_char][next_char] += 1 

        transistion_mat = copy.deepcopy(transition_counts)

        for key, val in transistion_mat.items():
            total_count = character_counts[key]
            val = dict(map(lambda tup: (tup[0], tup[1]/total_count), transistion_mat[key].items()))

            transistion_mat[key] = val

        return transistion_mat

    def next(self, state):
        return self.mm.next(state=state)

    def preprocess_corpus(self, corpus):
        corpus = corpus.lower()
        return corpus.replace("\s+", "")


if __name__ == "__main__":
    corpus = open("tinyshakespeare.txt", "rt").read()
    
    hidden_markov_model = HMM(corpus=corpus)


    present_state = input("Enter a character to start with: ")

    for _ in range(140):
        print(present_state, end='')
        present_state = hidden_markov_model.next(state=present_state)
