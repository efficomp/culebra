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
   Culebra. If not, see <http://www.gnu.org/licenses/>.

   This work is supported by projects PGC2018-098813-B-C31 and
   PID2022-137461NB-C31, both funded by the Spanish "Ministerio de Ciencia,
   Innovación y Universidades" and by the European Regional Development Fund
   (ERDF).

:class:`culebra.trainer.ea.abc.CooperativeEA` class
===================================================

.. autoclass:: culebra.trainer.ea.abc.CooperativeEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.abc.CooperativeEA.objective_stats
.. autoattribute:: culebra.trainer.ea.abc.CooperativeEA.stats_names

Class methods
-------------
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.load

Properties
----------
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.checkpoint_activation
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.container
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.crossover_funcs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.crossover_probs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.current_iter
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.fitness_function
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.gene_ind_mutation_probs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.index
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.logbook
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.mutation_funcs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.mutation_probs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.num_evals
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.num_subtrainers
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.pop_sizes
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.random_seed
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.representation_freq
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.representation_selection_func
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.representation_selection_func_params
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.representation_size
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.representation_topology_func
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.representation_topology_func_params
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.representatives
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.runtime
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.selection_funcs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.selection_funcs_params
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.solution_classes
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.species
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.subtrainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.subtrainer_cls
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.subtrainer_params
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.subtrainers
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.verbosity

Private properties
------------------
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_checkpoint_activation
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_checkpoint_filename
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_checkpoint_freq
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_crossover_funcs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_crossover_probs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_gene_ind_mutation_probs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_index
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_max_num_iters
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_mutation_funcs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_mutation_probs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_num_subtrainers
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_pop_sizes
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_representation_freq
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_representation_selection_func
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_representation_selection_func_params
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_representation_size
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_representation_topology_func
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_representation_topology_func_params
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_selection_funcs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_selection_funcs_params
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._default_verbosity
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._subtrainer_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.receive_representatives
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.send_representatives

Private static methods
----------------------
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._init_subtrainer_representatives

Methods
-------
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.best_representatives
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.best_solutions
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.dump
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.evaluate
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.reset
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.test
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.train

Private methods
---------------
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._default_termination_func
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._do_iteration
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._finish_iteration
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._finish_search
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._generate_subtrainers
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._get_state
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._init_internals
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._init_representatives
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._init_search
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._init_state
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._load_state
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._new_state
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._reset_internals
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._reset_state
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._save_state
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._search
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._set_cooperative_fitness
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._set_state
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._start_iteration
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._termination_criterion
