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

# A bash shell will be used for the recipes
SHELL = /bin/bash

# Example
EXAMPLE = example.py

# Data
DATA = people.sc.fsu.edu

# Tools
PYTHON = /usr/bin/env python3
RM = rm -rf


.PHONY: all
all : $(EXAMPLE)
	@echo "Running ..."
	@$(PYTHON) $(EXAMPLE)
	@echo

.PHONY: clean
clean:
	@echo "Cleaning ..."
	@$(RM) __pycache__
	@echo

.PHONY: distclean
distclean: clean
	@$(RM) checkpoint* $(DATA)
