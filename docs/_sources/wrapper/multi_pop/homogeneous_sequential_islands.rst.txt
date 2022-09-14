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

:py:class:`wrapper.multi_pop.HomogeneousSequentialIslands` class
================================================================

.. autoclass:: wrapper.multi_pop.HomogeneousSequentialIslands

Class attributes
----------------
.. autoattribute:: wrapper.multi_pop.HomogeneousSequentialIslands.stats_names
.. autoattribute:: wrapper.multi_pop.HomogeneousSequentialIslands.objective_stats

Properties
----------
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.individual_cls
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.species
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.fitness_function
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.subpop_wrapper_cls
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.subpop_wrapper_params
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.num_gens
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.pop_size
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.crossover_func
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.mutation_func
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.selection_func
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.crossover_prob
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.mutation_prob
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.gene_ind_mutation_prob
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.selection_func_params
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.num_subpops
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.representation_size
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.representation_freq
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.representation_topology_func
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.representation_topology_func_params
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.representation_selection_func
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.representation_selection_func_params
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.checkpoint_enable
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.checkpoint_freq
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.checkpoint_filename
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.verbose
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.random_seed
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.subpop_wrappers
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.subpop_wrapper_checkpoint_filenames
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.logbook
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.num_evals
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.runtime
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.index
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.container
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands.representatives

Private Properties
------------------
.. autoproperty:: wrapper.multi_pop.HomogeneousSequentialIslands._state

Static methods
--------------
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands.receive_representatives
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands.send_representatives


Methods
-------
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands.reset
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands.evaluate
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands.best_solutions
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands.best_representatives
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands.train
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands.test


Private methods
---------------
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._generate_subpop_wrappers
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._save_state
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._load_state
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._new_state
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._reset_state
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._init_state
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._init_internals
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._reset_internals
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._init_search
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._start_generation
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._preprocess_generation
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._do_generation
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._postprocess_generation
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._finish_generation
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._search
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._finish_search
.. automethod:: wrapper.multi_pop.HomogeneousSequentialIslands._init_representatives
