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

"""Implementation of some wrapper algorithms.

By the moment, culebra is centered on the use of generationsl evolutionary
algorithms as the search engine for wrapper procedures. Thus, the wrapper
module is composed by:

  * The :py:class:`~wrapper.Generational` class, which inherits from
    :py:class:`~base.Wrapper`, and provides the interface for all the
    evolutionary wrappers within this module.
  * The :py:mod:`~wrapper.single_pop` sub-module, which provides
    single-population generational wrappers.
  * The :py:mod:`~wrapper.multi_pop` sub-module, which provides
    multiple-population generational wrappers.
"""

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

from .generational import *
from . import single_pop
from . import multi_pop

__all__ = [
    'single_pop',
    'multi_pop'
]
