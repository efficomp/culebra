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

:py:class:`wrapper.multi_pop.MultiPop` class
============================================

.. autoclass:: wrapper.multi_pop.MultiPop

Class attributes
----------------
.. autoattribute:: wrapper.multi_pop.MultiPop.stats_names
.. autoattribute:: wrapper.multi_pop.MultiPop.objective_stats

Properties
----------
.. autoproperty:: wrapper.multi_pop.MultiPop.fitness_function
.. autoproperty:: wrapper.multi_pop.MultiPop.subpop_wrapper_cls
.. autoproperty:: wrapper.multi_pop.MultiPop.subpop_wrapper_params
.. autoproperty:: wrapper.multi_pop.MultiPop.num_gens
.. autoproperty:: wrapper.multi_pop.MultiPop.num_subpops
.. autoproperty:: wrapper.multi_pop.MultiPop.representation_size
.. autoproperty:: wrapper.multi_pop.MultiPop.representation_freq
.. autoproperty:: wrapper.multi_pop.MultiPop.representation_topology_func
.. autoproperty:: wrapper.multi_pop.MultiPop.representation_topology_func_params
.. autoproperty:: wrapper.multi_pop.MultiPop.representation_selection_func
.. autoproperty:: wrapper.multi_pop.MultiPop.representation_selection_func_params
.. autoproperty:: wrapper.multi_pop.MultiPop.checkpoint_enable
.. autoproperty:: wrapper.multi_pop.MultiPop.checkpoint_freq
.. autoproperty:: wrapper.multi_pop.MultiPop.checkpoint_filename
.. autoproperty:: wrapper.multi_pop.MultiPop.verbose
.. autoproperty:: wrapper.multi_pop.MultiPop.random_seed
.. autoproperty:: wrapper.multi_pop.MultiPop.subpop_wrappers
.. autoproperty:: wrapper.multi_pop.MultiPop.subpop_wrapper_checkpoint_filenames
.. autoproperty:: wrapper.multi_pop.MultiPop.logbook
.. autoproperty:: wrapper.multi_pop.MultiPop.num_evals
.. autoproperty:: wrapper.multi_pop.MultiPop.runtime
.. autoproperty:: wrapper.multi_pop.MultiPop.index
.. autoproperty:: wrapper.multi_pop.MultiPop.container
.. autoproperty:: wrapper.multi_pop.MultiPop.representatives

Private Properties
------------------
.. autoproperty:: wrapper.multi_pop.MultiPop._state

Static methods
--------------
.. automethod:: wrapper.multi_pop.MultiPop.receive_representatives
.. automethod:: wrapper.multi_pop.MultiPop.send_representatives


Methods
-------
.. automethod:: wrapper.multi_pop.MultiPop.reset
.. automethod:: wrapper.multi_pop.MultiPop.evaluate
.. automethod:: wrapper.multi_pop.MultiPop.best_solutions
.. automethod:: wrapper.multi_pop.MultiPop.best_representatives
.. automethod:: wrapper.multi_pop.MultiPop.train
.. automethod:: wrapper.multi_pop.MultiPop.test


Private methods
---------------
.. automethod:: wrapper.multi_pop.MultiPop._generate_subpop_wrappers
.. automethod:: wrapper.multi_pop.MultiPop._save_state
.. automethod:: wrapper.multi_pop.MultiPop._load_state
.. automethod:: wrapper.multi_pop.MultiPop._new_state
.. automethod:: wrapper.multi_pop.MultiPop._reset_state
.. automethod:: wrapper.multi_pop.MultiPop._init_state
.. automethod:: wrapper.multi_pop.MultiPop._init_internals
.. automethod:: wrapper.multi_pop.MultiPop._reset_internals
.. automethod:: wrapper.multi_pop.MultiPop._init_search
.. automethod:: wrapper.multi_pop.MultiPop._start_generation
.. automethod:: wrapper.multi_pop.MultiPop._preprocess_generation
.. automethod:: wrapper.multi_pop.MultiPop._do_generation
.. automethod:: wrapper.multi_pop.MultiPop._postprocess_generation
.. automethod:: wrapper.multi_pop.MultiPop._finish_generation
.. automethod:: wrapper.multi_pop.MultiPop._search
.. automethod:: wrapper.multi_pop.MultiPop._finish_search
.. automethod:: wrapper.multi_pop.MultiPop._init_representatives
