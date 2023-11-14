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

:py:class:`culebra.trainer.ea.abc.HeterogeneousEA` class
========================================================

.. autoclass:: culebra.trainer.ea.abc.HeterogeneousEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.abc.HeterogeneousEA.stats_names
.. autoattribute:: culebra.trainer.ea.abc.HeterogeneousEA.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.fitness_function
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.subtrainer_cls
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.current_iter
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.pop_sizes
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.crossover_funcs
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.mutation_funcs
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.selection_funcs
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.crossover_probs
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.mutation_probs
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.gene_ind_mutation_probs
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.selection_funcs_params
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.num_subtrainers
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.representation_size
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.representation_freq
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.representation_topology_func
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.representation_topology_func_params
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.representation_selection_func
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.representation_selection_func_params
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.checkpoint_enable
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.verbose
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.random_seed
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.logbook
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.num_evals
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.runtime
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.index
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.container
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.representatives
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.subtrainer_params
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.subtrainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA.subtrainers


Private properties
------------------
.. autoproperty:: culebra.trainer.ea.abc.HeterogeneousEA._subtrainer_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA.receive_representatives
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA.send_representatives

Methods
-------
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA.reset
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA.evaluate
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA.best_solutions
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA.best_representatives
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA.train
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA.test

Private methods
---------------
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._generate_subtrainers
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._get_state
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._set_state
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._save_state
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._load_state
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._new_state
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._init_state
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._reset_state
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._init_internals
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._reset_internals
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._init_search
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._search
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._finish_search
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._start_iteration
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._do_iteration
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._finish_iteration
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._default_termination_func
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._termination_criterion
.. automethod:: culebra.trainer.ea.abc.HeterogeneousEA._init_representatives
