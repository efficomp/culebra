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

:class:`culebra.trainer.ea.ElitistEA` class
===========================================

.. autoclass:: culebra.trainer.ea.ElitistEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.ElitistEA.objective_stats
.. autoattribute:: culebra.trainer.ea.ElitistEA.stats_names

Class methods
-------------
.. automethod:: culebra.trainer.ea.ElitistEA.load

Properties
----------
.. autoproperty:: culebra.trainer.ea.ElitistEA.checkpoint_activation
.. autoproperty:: culebra.trainer.ea.ElitistEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.ElitistEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.ElitistEA.container
.. autoproperty:: culebra.trainer.ea.ElitistEA.crossover_func
.. autoproperty:: culebra.trainer.ea.ElitistEA.crossover_prob
.. autoproperty:: culebra.trainer.ea.ElitistEA.current_iter
.. autoproperty:: culebra.trainer.ea.ElitistEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.ElitistEA.elite_size
.. autoproperty:: culebra.trainer.ea.ElitistEA.fitness_function
.. autoproperty:: culebra.trainer.ea.ElitistEA.gene_ind_mutation_prob
.. autoproperty:: culebra.trainer.ea.ElitistEA.index
.. autoproperty:: culebra.trainer.ea.ElitistEA.logbook
.. autoproperty:: culebra.trainer.ea.ElitistEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.ElitistEA.mutation_func
.. autoproperty:: culebra.trainer.ea.ElitistEA.mutation_prob
.. autoproperty:: culebra.trainer.ea.ElitistEA.num_evals
.. autoproperty:: culebra.trainer.ea.ElitistEA.pop
.. autoproperty:: culebra.trainer.ea.ElitistEA.pop_size
.. autoproperty:: culebra.trainer.ea.ElitistEA.random_seed
.. autoproperty:: culebra.trainer.ea.ElitistEA.representatives
.. autoproperty:: culebra.trainer.ea.ElitistEA.runtime
.. autoproperty:: culebra.trainer.ea.ElitistEA.selection_func
.. autoproperty:: culebra.trainer.ea.ElitistEA.selection_func_params
.. autoproperty:: culebra.trainer.ea.ElitistEA.solution_cls
.. autoproperty:: culebra.trainer.ea.ElitistEA.species
.. autoproperty:: culebra.trainer.ea.ElitistEA.verbosity

Private properties
------------------
.. autoproperty:: culebra.trainer.ea.ElitistEA._default_checkpoint_activation
.. autoproperty:: culebra.trainer.ea.ElitistEA._default_checkpoint_filename
.. autoproperty:: culebra.trainer.ea.ElitistEA._default_checkpoint_freq
.. autoproperty:: culebra.trainer.ea.ElitistEA._default_crossover_func
.. autoproperty:: culebra.trainer.ea.ElitistEA._default_crossover_prob
.. autoproperty:: culebra.trainer.ea.ElitistEA._default_elite_size
.. autoproperty:: culebra.trainer.ea.ElitistEA._default_gene_ind_mutation_prob
.. autoproperty:: culebra.trainer.ea.ElitistEA._default_index
.. autoproperty:: culebra.trainer.ea.ElitistEA._default_max_num_iters
.. autoproperty:: culebra.trainer.ea.ElitistEA._default_mutation_func
.. autoproperty:: culebra.trainer.ea.ElitistEA._default_mutation_prob
.. autoproperty:: culebra.trainer.ea.ElitistEA._default_pop_size
.. autoproperty:: culebra.trainer.ea.ElitistEA._default_selection_func
.. autoproperty:: culebra.trainer.ea.ElitistEA._default_selection_func_params
.. autoproperty:: culebra.trainer.ea.ElitistEA._default_verbosity

Methods
-------
.. automethod:: culebra.trainer.ea.ElitistEA.best_representatives
.. automethod:: culebra.trainer.ea.ElitistEA.best_solutions
.. automethod:: culebra.trainer.ea.ElitistEA.dump
.. automethod:: culebra.trainer.ea.ElitistEA.evaluate
.. automethod:: culebra.trainer.ea.ElitistEA.reset
.. automethod:: culebra.trainer.ea.ElitistEA.test
.. automethod:: culebra.trainer.ea.ElitistEA.train

Private methods
---------------
.. automethod:: culebra.trainer.ea.ElitistEA._default_termination_func
.. automethod:: culebra.trainer.ea.ElitistEA._do_iteration
.. automethod:: culebra.trainer.ea.ElitistEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.ElitistEA._evaluate_pop
.. automethod:: culebra.trainer.ea.ElitistEA._finish_iteration
.. automethod:: culebra.trainer.ea.ElitistEA._finish_search
.. automethod:: culebra.trainer.ea.ElitistEA._generate_initial_pop
.. automethod:: culebra.trainer.ea.ElitistEA._get_state
.. automethod:: culebra.trainer.ea.ElitistEA._init_internals
.. automethod:: culebra.trainer.ea.ElitistEA._init_representatives
.. automethod:: culebra.trainer.ea.ElitistEA._init_search
.. automethod:: culebra.trainer.ea.ElitistEA._init_state
.. automethod:: culebra.trainer.ea.ElitistEA._load_state
.. automethod:: culebra.trainer.ea.ElitistEA._new_state
.. automethod:: culebra.trainer.ea.ElitistEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.ElitistEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.ElitistEA._reset_internals
.. automethod:: culebra.trainer.ea.ElitistEA._reset_state
.. automethod:: culebra.trainer.ea.ElitistEA._save_state
.. automethod:: culebra.trainer.ea.ElitistEA._search
.. automethod:: culebra.trainer.ea.ElitistEA._set_cooperative_fitness
.. automethod:: culebra.trainer.ea.ElitistEA._set_state
.. automethod:: culebra.trainer.ea.ElitistEA._start_iteration
.. automethod:: culebra.trainer.ea.ElitistEA._termination_criterion
