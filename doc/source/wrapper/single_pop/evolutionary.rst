..
   This file is part of

   Culebra is free software: you can redistribute it and/or modify it under the
   terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   Culebra is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
   details.

   You should have received a copy of the GNU General Public License along with
    If not, see <http://www.gnu.org/licenses/>.

   This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
   de Ciencia, Innovación y Universidades"), and by the European Regional
   Development Fund (ERDF).

:py:class:`wrapper.single_pop.Evolutionary` class
=================================================

.. autoclass:: wrapper.single_pop.Evolutionary

Class attributes
----------------
.. autoattribute:: wrapper.single_pop.Evolutionary.stats_names
.. autoattribute:: wrapper.single_pop.Evolutionary.objective_stats

Properties
----------
.. autoproperty:: wrapper.single_pop.Evolutionary.individual_cls
.. autoproperty:: wrapper.single_pop.Evolutionary.species
.. autoproperty:: wrapper.single_pop.Evolutionary.fitness_function
.. autoproperty:: wrapper.single_pop.Evolutionary.num_gens
.. autoproperty:: wrapper.single_pop.Evolutionary.pop_size
.. autoproperty:: wrapper.single_pop.Evolutionary.crossover_func
.. autoproperty:: wrapper.single_pop.Evolutionary.mutation_func
.. autoproperty:: wrapper.single_pop.Evolutionary.selection_func
.. autoproperty:: wrapper.single_pop.Evolutionary.crossover_prob
.. autoproperty:: wrapper.single_pop.Evolutionary.mutation_prob
.. autoproperty:: wrapper.single_pop.Evolutionary.gene_ind_mutation_prob
.. autoproperty:: wrapper.single_pop.Evolutionary.selection_func_params
.. autoproperty:: wrapper.single_pop.Evolutionary.checkpoint_enable
.. autoproperty:: wrapper.single_pop.Evolutionary.checkpoint_freq
.. autoproperty:: wrapper.single_pop.Evolutionary.checkpoint_filename
.. autoproperty:: wrapper.single_pop.Evolutionary.verbose
.. autoproperty:: wrapper.single_pop.Evolutionary.random_seed
.. autoproperty:: wrapper.single_pop.Evolutionary.pop
.. autoproperty:: wrapper.single_pop.Evolutionary.logbook
.. autoproperty:: wrapper.single_pop.Evolutionary.num_evals
.. autoproperty:: wrapper.single_pop.Evolutionary.runtime
.. autoproperty:: wrapper.single_pop.Evolutionary.index
.. autoproperty:: wrapper.single_pop.Evolutionary.container
.. autoproperty:: wrapper.single_pop.Evolutionary.representatives

Private Properties
------------------
.. autoproperty:: wrapper.single_pop.Evolutionary._state

Methods
-------
.. automethod:: wrapper.single_pop.Evolutionary.reset
.. automethod:: wrapper.single_pop.Evolutionary.evaluate
.. automethod:: wrapper.single_pop.Evolutionary.best_solutions
.. automethod:: wrapper.single_pop.Evolutionary.best_representatives
.. automethod:: wrapper.single_pop.Evolutionary.train
.. automethod:: wrapper.single_pop.Evolutionary.test

Private methods
---------------
.. automethod:: wrapper.single_pop.Evolutionary._save_state
.. automethod:: wrapper.single_pop.Evolutionary._load_state
.. automethod:: wrapper.single_pop.Evolutionary._generate_initial_pop
.. automethod:: wrapper.single_pop.Evolutionary._evaluate_pop
.. automethod:: wrapper.single_pop.Evolutionary._do_generation_stats
.. automethod:: wrapper.single_pop.Evolutionary._new_state
.. automethod:: wrapper.single_pop.Evolutionary._reset_state
.. automethod:: wrapper.single_pop.Evolutionary._init_state
.. automethod:: wrapper.single_pop.Evolutionary._init_internals
.. automethod:: wrapper.single_pop.Evolutionary._reset_internals
.. automethod:: wrapper.single_pop.Evolutionary._init_search
.. automethod:: wrapper.single_pop.Evolutionary._start_generation
.. automethod:: wrapper.single_pop.Evolutionary._preprocess_generation
.. automethod:: wrapper.single_pop.Evolutionary._do_generation
.. automethod:: wrapper.single_pop.Evolutionary._postprocess_generation
.. automethod:: wrapper.single_pop.Evolutionary._finish_generation
.. automethod:: wrapper.single_pop.Evolutionary._search
.. automethod:: wrapper.single_pop.Evolutionary._finish_search
.. automethod:: wrapper.single_pop.Evolutionary._init_representatives
