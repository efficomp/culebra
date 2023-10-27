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

:py:class:`culebra.trainer.ea.ParallelCooperativeEA` class
==========================================================

.. autoclass:: culebra.trainer.ea.ParallelCooperativeEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.ParallelCooperativeEA.stats_names
.. autoattribute:: culebra.trainer.ea.ParallelCooperativeEA.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.solution_classes
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.species
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.fitness_function
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.subtrainer_cls
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.current_iter
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.pop_sizes
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.crossover_funcs
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.mutation_funcs
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.selection_funcs
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.crossover_probs
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.mutation_probs
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.gene_ind_mutation_probs
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.selection_funcs_params
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.num_subtrainers
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.representation_size
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.representation_freq
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.representation_topology_func
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.representation_topology_func_params
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.representation_selection_func
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.representation_selection_func_params
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.checkpoint_enable
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.verbose
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.random_seed
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.logbook
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.num_evals
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.runtime
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.index
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.container
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.representatives
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.subtrainer_params
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.subtrainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA.subtrainers


Private properties
------------------
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA._state
.. autoproperty:: culebra.trainer.ea.ParallelCooperativeEA._subtrainer_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA.receive_representatives
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA.send_representatives

Private static methods
----------------------
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._init_subtrainer_representatives

Methods
-------
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA.reset
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA.evaluate
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA.best_solutions
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA.best_representatives
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA.train
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA.test

Private methods
---------------
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._generate_subtrainers
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._save_state
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._load_state
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._new_state
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._init_state
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._reset_state
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._init_internals
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._reset_internals
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._init_search
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._search
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._finish_search
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._start_iteration
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._do_iteration
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._finish_iteration
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._default_termination_func
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._termination_criterion
.. automethod:: culebra.trainer.ea.ParallelCooperativeEA._init_representatives
