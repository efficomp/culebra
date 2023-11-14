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

:py:class:`culebra.trainer.ea.SequentialCooperativeEA` class
============================================================

.. autoclass:: culebra.trainer.ea.SequentialCooperativeEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.SequentialCooperativeEA.stats_names
.. autoattribute:: culebra.trainer.ea.SequentialCooperativeEA.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.solution_classes
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.species
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.fitness_function
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.subtrainer_cls
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.current_iter
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.pop_sizes
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.crossover_funcs
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.mutation_funcs
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.selection_funcs
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.crossover_probs
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.mutation_probs
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.gene_ind_mutation_probs
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.selection_funcs_params
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.num_subtrainers
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.representation_size
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.representation_freq
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.representation_topology_func
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.representation_topology_func_params
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.representation_selection_func
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.representation_selection_func_params
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.checkpoint_enable
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.verbose
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.random_seed
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.logbook
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.num_evals
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.runtime
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.index
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.container
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.representatives
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.subtrainer_params
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.subtrainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA.subtrainers


Private properties
------------------
.. autoproperty:: culebra.trainer.ea.SequentialCooperativeEA._subtrainer_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA.receive_representatives
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA.send_representatives

Private static methods
----------------------
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._init_subtrainer_representatives

Methods
-------
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA.reset
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA.evaluate
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA.best_solutions
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA.best_representatives
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA.train
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA.test

Private methods
---------------
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._generate_subtrainers
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._get_state
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._set_state
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._save_state
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._load_state
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._new_state
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._init_state
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._reset_state
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._init_internals
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._reset_internals
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._init_search
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._search
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._finish_search
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._start_iteration
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._do_iteration
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._finish_iteration
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._default_termination_func
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._termination_criterion
.. automethod:: culebra.trainer.ea.SequentialCooperativeEA._init_representatives
