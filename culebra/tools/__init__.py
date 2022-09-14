# This file is part of culebra.
#
# Culebra is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Culebra is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Culebra. If not, see <http://www.gnu.org/licenses/>.
#
# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

"""Tools to automate the execution of experiments.

Automated experimentation is also a quite valuable characteristic when a
:py:class:`~base.Wrapper` method has to be run many times. Culebra provides
this features by means of the following classes:

  * The :py:class:`~tools.Results` class, to manage the results provided by
    the evaluation of any :py:class:`~base.Wrapper`
  * The :py:class:`~tools.Evaluation` class, a base class for the evaluation
    of wrappers
  * The :py:class:`~tools.Experiment` class, designed to run a single
    experiment with a :py:class:`~base.Wrapper`
  * The :py:class:`~tools.Batch` class, which allows to run a batch of
    experiments with the same configuration
"""

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2022, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


from .results import *
from .evaluation import *
