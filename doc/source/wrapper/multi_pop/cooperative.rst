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

:py:class:`wrapper.multi_pop.Cooperative` class
===============================================

.. autoclass:: wrapper.multi_pop.Cooperative

Class attributes
----------------
.. autoattribute:: wrapper.multi_pop.Cooperative.stats_names
.. autoattribute:: wrapper.multi_pop.Cooperative.objective_stats

Properties
----------
.. autoproperty:: wrapper.multi_pop.Cooperative.individual_classes
.. autoproperty:: wrapper.multi_pop.Cooperative.species
.. autoproperty:: wrapper.multi_pop.Cooperative.fitness_function
.. autoproperty:: wrapper.multi_pop.Cooperative.subpop_wrapper_cls
.. autoproperty:: wrapper.multi_pop.Cooperative.subpop_wrapper_params
.. autoproperty:: wrapper.multi_pop.Cooperative.num_gens
.. autoproperty:: wrapper.multi_pop.Cooperative.pop_sizes
.. autoproperty:: wrapper.multi_pop.Cooperative.crossover_funcs
.. autoproperty:: wrapper.multi_pop.Cooperative.mutation_funcs
.. autoproperty:: wrapper.multi_pop.Cooperative.selection_funcs
.. autoproperty:: wrapper.multi_pop.Cooperative.crossover_probs
.. autoproperty:: wrapper.multi_pop.Cooperative.mutation_probs
.. autoproperty:: wrapper.multi_pop.Cooperative.gene_ind_mutation_probs
.. autoproperty:: wrapper.multi_pop.Cooperative.selection_funcs_params
.. autoproperty:: wrapper.multi_pop.Cooperative.num_subpops
.. autoproperty:: wrapper.multi_pop.Cooperative.representation_size
.. autoproperty:: wrapper.multi_pop.Cooperative.representation_freq
.. autoproperty:: wrapper.multi_pop.Cooperative.representation_topology_func
.. autoproperty:: wrapper.multi_pop.Cooperative.representation_topology_func_params
.. autoproperty:: wrapper.multi_pop.Cooperative.representation_selection_func
.. autoproperty:: wrapper.multi_pop.Cooperative.representation_selection_func_params
.. autoproperty:: wrapper.multi_pop.Cooperative.checkpoint_enable
.. autoproperty:: wrapper.multi_pop.Cooperative.checkpoint_freq
.. autoproperty:: wrapper.multi_pop.Cooperative.checkpoint_filename
.. autoproperty:: wrapper.multi_pop.Cooperative.verbose
.. autoproperty:: wrapper.multi_pop.Cooperative.random_seed
.. autoproperty:: wrapper.multi_pop.Cooperative.subpop_wrappers
.. autoproperty:: wrapper.multi_pop.Cooperative.subpop_wrapper_checkpoint_filenames
.. autoproperty:: wrapper.multi_pop.Cooperative.logbook
.. autoproperty:: wrapper.multi_pop.Cooperative.num_evals
.. autoproperty:: wrapper.multi_pop.Cooperative.runtime
.. autoproperty:: wrapper.multi_pop.Cooperative.index
.. autoproperty:: wrapper.multi_pop.Cooperative.container
.. autoproperty:: wrapper.multi_pop.Cooperative.representatives

Private Properties
------------------
.. autoproperty:: wrapper.multi_pop.Cooperative._state

Static methods
--------------
.. automethod:: wrapper.multi_pop.Cooperative.receive_representatives
.. automethod:: wrapper.multi_pop.Cooperative.send_representatives

Private static methods
----------------------
.. automethod:: wrapper.multi_pop.Cooperative._init_subpop_wrapper_representatives

Methods
-------
.. automethod:: wrapper.multi_pop.Cooperative.reset
.. automethod:: wrapper.multi_pop.Cooperative.evaluate
.. automethod:: wrapper.multi_pop.Cooperative.best_solutions
.. automethod:: wrapper.multi_pop.Cooperative.best_representatives
.. automethod:: wrapper.multi_pop.Cooperative.train
.. automethod:: wrapper.multi_pop.Cooperative.test


Private methods
---------------
.. automethod:: wrapper.multi_pop.Cooperative._generate_subpop_wrappers
.. automethod:: wrapper.multi_pop.Cooperative._save_state
.. automethod:: wrapper.multi_pop.Cooperative._load_state
.. automethod:: wrapper.multi_pop.Cooperative._new_state
.. automethod:: wrapper.multi_pop.Cooperative._reset_state
.. automethod:: wrapper.multi_pop.Cooperative._init_state
.. automethod:: wrapper.multi_pop.Cooperative._init_internals
.. automethod:: wrapper.multi_pop.Cooperative._reset_internals
.. automethod:: wrapper.multi_pop.Cooperative._init_search
.. automethod:: wrapper.multi_pop.Cooperative._start_generation
.. automethod:: wrapper.multi_pop.Cooperative._preprocess_generation
.. automethod:: wrapper.multi_pop.Cooperative._do_generation
.. automethod:: wrapper.multi_pop.Cooperative._postprocess_generation
.. automethod:: wrapper.multi_pop.Cooperative._finish_generation
.. automethod:: wrapper.multi_pop.Cooperative._search
.. automethod:: wrapper.multi_pop.Cooperative._finish_search
.. automethod:: wrapper.multi_pop.Cooperative._init_representatives
