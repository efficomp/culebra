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

:py:class:`culebra.trainer.ea.abc.HeterogeneousIslandsEA` class
===============================================================

.. autoclass:: culebra.trainer.ea.abc.HeterogeneousIslandsEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.stats_names
.. autoattribute:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.load

Properties
----------
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.solution_cls
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.species
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.fitness_function
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.subtrainer_cls
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.current_iter
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.pop_sizes
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.crossover_funcs
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.mutation_funcs
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.selection_funcs
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.crossover_probs
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.mutation_probs
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.gene_ind_mutation_probs
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.selection_funcs_params
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.num_subtrainers
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.representation_size
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.representation_freq
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.representation_topology_func
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.representation_topology_func_params
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.representation_selection_func
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.representation_selection_func_params
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.checkpoint_enable
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.verbose
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.random_seed
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.logbook
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.num_evals
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.runtime
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.index
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.container
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.representatives
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.subtrainer_params
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.subtrainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.subtrainers

Private properties
------------------
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._subtrainer_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.receive_representatives
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.send_representatives

Methods
-------
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.dump
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.reset
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.evaluate
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.best_solutions
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.best_representatives
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.train
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA.test

Private methods
---------------
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._generate_subtrainers
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._get_state
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._set_state
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._save_state
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._load_state
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._new_state
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._init_state
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._reset_state
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._init_internals
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._reset_internals
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._init_search
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._search
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._finish_search
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._start_iteration
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._do_iteration
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._finish_iteration
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_termination_func
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._termination_criterion
.. automethod:: culebra.trainer.ea.abc.HeterogeneousIslandsEA._init_representatives
