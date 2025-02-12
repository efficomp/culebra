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

:py:class:`culebra.trainer.ea.HomogeneousParallelIslandsEA` class
=================================================================

.. autoclass:: culebra.trainer.ea.HomogeneousParallelIslandsEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.HomogeneousParallelIslandsEA.stats_names
.. autoattribute:: culebra.trainer.ea.HomogeneousParallelIslandsEA.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA.load_pickle

Properties
----------
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.solution_cls
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.species
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.fitness_function
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.subtrainer_cls
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.current_iter
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.pop_size
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.crossover_func
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.mutation_func
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.selection_func
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.crossover_prob
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.mutation_prob
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.gene_ind_mutation_prob
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.selection_func_params
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.num_subtrainers
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.representation_size
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.representation_freq
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.representation_topology_func
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.representation_topology_func_params
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.representation_selection_func
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.representation_selection_func_params
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.checkpoint_enable
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.verbose
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.random_seed
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.logbook
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.num_evals
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.runtime
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.index
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.container
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.representatives
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.subtrainer_params
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.subtrainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA.subtrainers

Private properties
------------------
.. autoproperty:: culebra.trainer.ea.HomogeneousParallelIslandsEA._subtrainer_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA.receive_representatives
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA.send_representatives

Methods
-------
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA.save_pickle
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA.reset
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA.evaluate
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA.best_solutions
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA.best_representatives
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA.train
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA.test

Private methods
---------------
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._generate_subtrainers
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._get_state
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._set_state
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._save_state
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._load_state
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._new_state
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._init_state
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._reset_state
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._init_internals
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._reset_internals
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._init_search
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._search
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._finish_search
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._start_iteration
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._do_iteration
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._finish_iteration
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._default_termination_func
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._termination_criterion
.. automethod:: culebra.trainer.ea.HomogeneousParallelIslandsEA._init_representatives
