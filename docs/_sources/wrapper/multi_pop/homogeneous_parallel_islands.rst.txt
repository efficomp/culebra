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

:py:class:`wrapper.multi_pop.HomogeneousParallelIslands` class
================================================================

.. autoclass:: wrapper.multi_pop.HomogeneousParallelIslands

Class attributes
----------------
.. autoattribute:: wrapper.multi_pop.HomogeneousParallelIslands.stats_names
.. autoattribute:: wrapper.multi_pop.HomogeneousParallelIslands.objective_stats

Properties
----------
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.individual_cls
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.species
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.fitness_function
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.subpop_wrapper_cls
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.subpop_wrapper_params
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.num_gens
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.pop_size
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.crossover_func
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.mutation_func
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.selection_func
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.crossover_prob
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.mutation_prob
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.gene_ind_mutation_prob
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.selection_func_params
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.num_subpops
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.representation_size
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.representation_freq
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.representation_topology_func
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.representation_topology_func_params
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.representation_selection_func
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.representation_selection_func_params
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.checkpoint_enable
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.checkpoint_freq
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.checkpoint_filename
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.verbose
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.random_seed
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.subpop_wrappers
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.subpop_wrapper_checkpoint_filenames
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.logbook
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.num_evals
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.runtime
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.index
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.container
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands.representatives

Private Properties
------------------
.. autoproperty:: wrapper.multi_pop.HomogeneousParallelIslands._state

Static methods
--------------
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands.receive_representatives
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands.send_representatives


Methods
-------
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands.reset
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands.evaluate
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands.best_solutions
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands.best_representatives
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands.train
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands.test


Private methods
---------------
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._generate_subpop_wrappers
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._save_state
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._load_state
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._new_state
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._reset_state
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._init_state
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._init_internals
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._reset_internals
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._init_search
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._start_generation
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._preprocess_generation
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._do_generation
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._postprocess_generation
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._finish_generation
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._search
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._finish_search
.. automethod:: wrapper.multi_pop.HomogeneousParallelIslands._init_representatives
