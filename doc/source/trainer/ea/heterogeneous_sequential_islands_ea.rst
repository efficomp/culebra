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

:py:class:`culebra.trainer.ea.HeterogeneousSequentialIslandsEA` class
=====================================================================

.. autoclass:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.stats_names
.. autoattribute:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.load

Properties
----------
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.solution_cls
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.species
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.fitness_function
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.subtrainer_cls
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.current_iter
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.pop_sizes
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.crossover_funcs
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.mutation_funcs
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.selection_funcs
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.crossover_probs
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.mutation_probs
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.gene_ind_mutation_probs
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.selection_funcs_params
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.num_subtrainers
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.representation_size
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.representation_freq
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.representation_topology_func
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.representation_topology_func_params
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.representation_selection_func
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.representation_selection_func_params
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.checkpoint_enable
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.verbose
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.random_seed
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.logbook
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.num_evals
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.runtime
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.index
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.container
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.representatives
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.subtrainer_params
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.subtrainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.subtrainers

Private properties
------------------
.. autoproperty:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._subtrainer_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.receive_representatives
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.send_representatives

Methods
-------
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.dump
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.reset
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.evaluate
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.best_solutions
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.best_representatives
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.train
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA.test

Private methods
---------------
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._generate_subtrainers
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._get_state
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._set_state
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._save_state
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._load_state
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._new_state
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._init_state
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._reset_state
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._init_internals
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._reset_internals
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._init_search
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._search
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._finish_search
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._start_iteration
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._do_iteration
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._finish_iteration
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._default_termination_func
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._termination_criterion
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._init_representatives
.. automethod:: culebra.trainer.ea.HeterogeneousSequentialIslandsEA._set_cooperative_fitness
