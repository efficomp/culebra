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

:py:class:`culebra.trainer.ea.abc.SinglePopEA` class
====================================================

.. autoclass:: culebra.trainer.ea.abc.SinglePopEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.abc.SinglePopEA.stats_names
.. autoattribute:: culebra.trainer.ea.abc.SinglePopEA.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.ea.abc.SinglePopEA.load

Properties
----------
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.solution_cls
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.species
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.fitness_function
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.current_iter
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.pop_size
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.crossover_func
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.mutation_func
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.selection_func
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.crossover_prob
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.mutation_prob
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.gene_ind_mutation_prob
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.selection_func_params
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.checkpoint_enable
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.verbose
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.random_seed
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.logbook
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.num_evals
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.runtime
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.index
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.container
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.representatives
.. autoproperty:: culebra.trainer.ea.abc.SinglePopEA.pop

Methods
-------
.. automethod:: culebra.trainer.ea.abc.SinglePopEA.dump
.. automethod:: culebra.trainer.ea.abc.SinglePopEA.reset
.. automethod:: culebra.trainer.ea.abc.SinglePopEA.evaluate
.. automethod:: culebra.trainer.ea.abc.SinglePopEA.best_solutions
.. automethod:: culebra.trainer.ea.abc.SinglePopEA.best_representatives
.. automethod:: culebra.trainer.ea.abc.SinglePopEA.train
.. automethod:: culebra.trainer.ea.abc.SinglePopEA.test

Private methods
---------------
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._get_state
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._set_state
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._save_state
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._load_state
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._new_state
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._init_state
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._reset_state
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._init_internals
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._reset_internals
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._init_search
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._search
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._finish_search
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._start_iteration
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._do_iteration
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._finish_iteration
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._default_termination_func
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._termination_criterion
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._init_representatives
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._generate_initial_pop
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._evaluate_pop
.. automethod:: culebra.trainer.ea.abc.SinglePopEA._set_cooperative_fitness
