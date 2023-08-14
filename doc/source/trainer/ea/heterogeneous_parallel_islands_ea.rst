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

:py:class:`culebra.trainer.ea.HeterogeneousParallelIslandsEA` class
===================================================================

.. autoclass:: culebra.trainer.ea.HeterogeneousParallelIslandsEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.stats_names
.. autoattribute:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.solution_cls
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.species
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.fitness_function
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.subpop_trainer_cls
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.current_iter
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.pop_sizes
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.crossover_funcs
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.mutation_funcs
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.selection_funcs
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.crossover_probs
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.mutation_probs
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.gene_ind_mutation_probs
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.selection_funcs_params
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.num_subpops
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.representation_size
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.representation_freq
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.representation_topology_func
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.representation_topology_func_params
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.representation_selection_func
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.representation_selection_func_params
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.checkpoint_enable
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.verbose
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.random_seed
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.logbook
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.num_evals
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.runtime
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.index
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.container
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.representatives
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.subpop_trainer_params
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.subpop_trainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.subpop_trainers


Private properties
------------------
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._state
.. autoproperty:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._subpop_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.receive_representatives
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.send_representatives

Methods
-------
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.reset
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.evaluate
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.best_solutions
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.best_representatives
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.train
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA.test

Private methods
---------------
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._generate_subpop_trainers
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._save_state
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._load_state
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._new_state
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._init_state
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._reset_state
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._init_internals
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._reset_internals
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._init_search
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._search
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._finish_search
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._start_iteration
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._do_iteration
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._finish_iteration
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._default_termination_func
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._termination_criterion
.. automethod:: culebra.trainer.ea.HeterogeneousParallelIslandsEA._init_representatives
