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

:py:class:`culebra.trainer.ea.NSGA` class
=========================================

.. autoclass:: culebra.trainer.ea.NSGA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.NSGA.stats_names
.. autoattribute:: culebra.trainer.ea.NSGA.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.ea.NSGA.load_pickle

Properties
----------
.. autoproperty:: culebra.trainer.ea.NSGA.solution_cls
.. autoproperty:: culebra.trainer.ea.NSGA.species
.. autoproperty:: culebra.trainer.ea.NSGA.fitness_function
.. autoproperty:: culebra.trainer.ea.NSGA.max_num_iters
.. autoproperty:: culebra.trainer.ea.NSGA.current_iter
.. autoproperty:: culebra.trainer.ea.NSGA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.NSGA.pop_size
.. autoproperty:: culebra.trainer.ea.NSGA.crossover_func
.. autoproperty:: culebra.trainer.ea.NSGA.mutation_func
.. autoproperty:: culebra.trainer.ea.NSGA.selection_func
.. autoproperty:: culebra.trainer.ea.NSGA.crossover_prob
.. autoproperty:: culebra.trainer.ea.NSGA.mutation_prob
.. autoproperty:: culebra.trainer.ea.NSGA.gene_ind_mutation_prob
.. autoproperty:: culebra.trainer.ea.NSGA.selection_func_params
.. autoproperty:: culebra.trainer.ea.NSGA.nsga3_reference_points
.. autoproperty:: culebra.trainer.ea.NSGA.nsga3_reference_points_p
.. autoproperty:: culebra.trainer.ea.NSGA.nsga3_reference_points_scaling
.. autoproperty:: culebra.trainer.ea.NSGA.checkpoint_enable
.. autoproperty:: culebra.trainer.ea.NSGA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.NSGA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.NSGA.verbose
.. autoproperty:: culebra.trainer.ea.NSGA.random_seed
.. autoproperty:: culebra.trainer.ea.NSGA.logbook
.. autoproperty:: culebra.trainer.ea.NSGA.num_evals
.. autoproperty:: culebra.trainer.ea.NSGA.runtime
.. autoproperty:: culebra.trainer.ea.NSGA.index
.. autoproperty:: culebra.trainer.ea.NSGA.container
.. autoproperty:: culebra.trainer.ea.NSGA.representatives
.. autoproperty:: culebra.trainer.ea.NSGA.pop

Methods
-------
.. automethod:: culebra.trainer.ea.NSGA.save_pickle
.. automethod:: culebra.trainer.ea.NSGA.reset
.. automethod:: culebra.trainer.ea.NSGA.evaluate
.. automethod:: culebra.trainer.ea.NSGA.best_solutions
.. automethod:: culebra.trainer.ea.NSGA.best_representatives
.. automethod:: culebra.trainer.ea.NSGA.train
.. automethod:: culebra.trainer.ea.NSGA.test

Private methods
---------------
.. automethod:: culebra.trainer.ea.NSGA._get_state
.. automethod:: culebra.trainer.ea.NSGA._set_state
.. automethod:: culebra.trainer.ea.NSGA._save_state
.. automethod:: culebra.trainer.ea.NSGA._load_state
.. automethod:: culebra.trainer.ea.NSGA._new_state
.. automethod:: culebra.trainer.ea.NSGA._init_state
.. automethod:: culebra.trainer.ea.NSGA._reset_state
.. automethod:: culebra.trainer.ea.NSGA._init_internals
.. automethod:: culebra.trainer.ea.NSGA._reset_internals
.. automethod:: culebra.trainer.ea.NSGA._init_search
.. automethod:: culebra.trainer.ea.NSGA._search
.. automethod:: culebra.trainer.ea.NSGA._finish_search
.. automethod:: culebra.trainer.ea.NSGA._start_iteration
.. automethod:: culebra.trainer.ea.NSGA._preprocess_iteration
.. automethod:: culebra.trainer.ea.NSGA._do_iteration
.. automethod:: culebra.trainer.ea.NSGA._postprocess_iteration
.. automethod:: culebra.trainer.ea.NSGA._finish_iteration
.. automethod:: culebra.trainer.ea.NSGA._do_iteration_stats
.. automethod:: culebra.trainer.ea.NSGA._default_termination_func
.. automethod:: culebra.trainer.ea.NSGA._termination_criterion
.. automethod:: culebra.trainer.ea.NSGA._init_representatives
.. automethod:: culebra.trainer.ea.NSGA._generate_initial_pop
.. automethod:: culebra.trainer.ea.NSGA._evaluate_pop
