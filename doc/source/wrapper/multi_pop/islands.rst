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

:py:class:`wrapper.multi_pop.Islands` class
===========================================

.. autoclass:: wrapper.multi_pop.Islands

Class attributes
----------------
.. autoattribute:: wrapper.multi_pop.Islands.stats_names
.. autoattribute:: wrapper.multi_pop.Islands.objective_stats

Properties
----------
.. autoproperty:: wrapper.multi_pop.Islands.individual_cls
.. autoproperty:: wrapper.multi_pop.Islands.species
.. autoproperty:: wrapper.multi_pop.Islands.fitness_function
.. autoproperty:: wrapper.multi_pop.Islands.subpop_wrapper_cls
.. autoproperty:: wrapper.multi_pop.Islands.subpop_wrapper_params
.. autoproperty:: wrapper.multi_pop.Islands.num_gens
.. autoproperty:: wrapper.multi_pop.Islands.num_subpops
.. autoproperty:: wrapper.multi_pop.Islands.representation_size
.. autoproperty:: wrapper.multi_pop.Islands.representation_freq
.. autoproperty:: wrapper.multi_pop.Islands.representation_topology_func
.. autoproperty:: wrapper.multi_pop.Islands.representation_topology_func_params
.. autoproperty:: wrapper.multi_pop.Islands.representation_selection_func
.. autoproperty:: wrapper.multi_pop.Islands.representation_selection_func_params
.. autoproperty:: wrapper.multi_pop.Islands.checkpoint_enable
.. autoproperty:: wrapper.multi_pop.Islands.checkpoint_freq
.. autoproperty:: wrapper.multi_pop.Islands.checkpoint_filename
.. autoproperty:: wrapper.multi_pop.Islands.verbose
.. autoproperty:: wrapper.multi_pop.Islands.random_seed
.. autoproperty:: wrapper.multi_pop.Islands.subpop_wrappers
.. autoproperty:: wrapper.multi_pop.Islands.subpop_wrapper_checkpoint_filenames
.. autoproperty:: wrapper.multi_pop.Islands.logbook
.. autoproperty:: wrapper.multi_pop.Islands.num_evals
.. autoproperty:: wrapper.multi_pop.Islands.runtime
.. autoproperty:: wrapper.multi_pop.Islands.index
.. autoproperty:: wrapper.multi_pop.Islands.container
.. autoproperty:: wrapper.multi_pop.Islands.representatives

Private Properties
------------------
.. autoproperty:: wrapper.multi_pop.Islands._state

Static methods
--------------
.. automethod:: wrapper.multi_pop.Islands.receive_representatives
.. automethod:: wrapper.multi_pop.Islands.send_representatives


Methods
-------
.. automethod:: wrapper.multi_pop.Islands.reset
.. automethod:: wrapper.multi_pop.Islands.evaluate
.. automethod:: wrapper.multi_pop.Islands.best_solutions
.. automethod:: wrapper.multi_pop.Islands.best_representatives
.. automethod:: wrapper.multi_pop.Islands.train
.. automethod:: wrapper.multi_pop.Islands.test


Private methods
---------------
.. automethod:: wrapper.multi_pop.Islands._generate_subpop_wrappers
.. automethod:: wrapper.multi_pop.Islands._save_state
.. automethod:: wrapper.multi_pop.Islands._load_state
.. automethod:: wrapper.multi_pop.Islands._new_state
.. automethod:: wrapper.multi_pop.Islands._reset_state
.. automethod:: wrapper.multi_pop.Islands._init_state
.. automethod:: wrapper.multi_pop.Islands._init_internals
.. automethod:: wrapper.multi_pop.Islands._reset_internals
.. automethod:: wrapper.multi_pop.Islands._init_search
.. automethod:: wrapper.multi_pop.Islands._start_generation
.. automethod:: wrapper.multi_pop.Islands._preprocess_generation
.. automethod:: wrapper.multi_pop.Islands._do_generation
.. automethod:: wrapper.multi_pop.Islands._postprocess_generation
.. automethod:: wrapper.multi_pop.Islands._finish_generation
.. automethod:: wrapper.multi_pop.Islands._search
.. automethod:: wrapper.multi_pop.Islands._finish_search
.. automethod:: wrapper.multi_pop.Islands._init_representatives
