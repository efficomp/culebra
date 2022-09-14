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

:py:class:`wrapper.multi_pop.HomogeneousIslands` class
======================================================

.. autoclass:: wrapper.multi_pop.HomogeneousIslands

Class attributes
----------------
.. autoattribute:: wrapper.multi_pop.HomogeneousIslands.stats_names
.. autoattribute:: wrapper.multi_pop.HomogeneousIslands.objective_stats

Properties
----------
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.individual_cls
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.species
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.fitness_function
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.subpop_wrapper_cls
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.subpop_wrapper_params
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.num_gens
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.pop_size
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.crossover_func
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.mutation_func
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.selection_func
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.crossover_prob
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.mutation_prob
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.gene_ind_mutation_prob
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.selection_func_params
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.num_subpops
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.representation_size
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.representation_freq
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.representation_topology_func
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.representation_topology_func_params
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.representation_selection_func
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.representation_selection_func_params
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.checkpoint_enable
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.checkpoint_freq
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.checkpoint_filename
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.verbose
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.random_seed
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.subpop_wrappers
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.subpop_wrapper_checkpoint_filenames
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.logbook
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.num_evals
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.runtime
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.index
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.container
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands.representatives

Private Properties
------------------
.. autoproperty:: wrapper.multi_pop.HomogeneousIslands._state

Static methods
--------------
.. automethod:: wrapper.multi_pop.HomogeneousIslands.receive_representatives
.. automethod:: wrapper.multi_pop.HomogeneousIslands.send_representatives


Methods
-------
.. automethod:: wrapper.multi_pop.HomogeneousIslands.reset
.. automethod:: wrapper.multi_pop.HomogeneousIslands.evaluate
.. automethod:: wrapper.multi_pop.HomogeneousIslands.best_solutions
.. automethod:: wrapper.multi_pop.HomogeneousIslands.best_representatives
.. automethod:: wrapper.multi_pop.HomogeneousIslands.train
.. automethod:: wrapper.multi_pop.HomogeneousIslands.test


Private methods
---------------
.. automethod:: wrapper.multi_pop.HomogeneousIslands._generate_subpop_wrappers
.. automethod:: wrapper.multi_pop.HomogeneousIslands._save_state
.. automethod:: wrapper.multi_pop.HomogeneousIslands._load_state
.. automethod:: wrapper.multi_pop.HomogeneousIslands._new_state
.. automethod:: wrapper.multi_pop.HomogeneousIslands._reset_state
.. automethod:: wrapper.multi_pop.HomogeneousIslands._init_state
.. automethod:: wrapper.multi_pop.HomogeneousIslands._init_internals
.. automethod:: wrapper.multi_pop.HomogeneousIslands._reset_internals
.. automethod:: wrapper.multi_pop.HomogeneousIslands._init_search
.. automethod:: wrapper.multi_pop.HomogeneousIslands._start_generation
.. automethod:: wrapper.multi_pop.HomogeneousIslands._preprocess_generation
.. automethod:: wrapper.multi_pop.HomogeneousIslands._do_generation
.. automethod:: wrapper.multi_pop.HomogeneousIslands._postprocess_generation
.. automethod:: wrapper.multi_pop.HomogeneousIslands._finish_generation
.. automethod:: wrapper.multi_pop.HomogeneousIslands._search
.. automethod:: wrapper.multi_pop.HomogeneousIslands._finish_search
.. automethod:: wrapper.multi_pop.HomogeneousIslands._init_representatives
