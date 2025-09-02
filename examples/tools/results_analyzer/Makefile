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
# This work is supported by projects PGC2018-098813-B-C31 and
# PID2022-137461NB-C31, both funded by the Spanish "Ministerio de Ciencia,
# Innovación y Universidades" and by the European Regional Development Fund
# (ERDF).

# A bash shell will be used for the recipes
SHELL = /bin/bash

# Script
SCRIPT = script.py

# Results
SERIALIZED_FILE_EXTENSION = "dill.gz"
RESULTS_BASENAME = "results"
RESULTS_SERIALIZED = $(RESULTS_BASENAME).$(SERIALIZED_FILE_EXTENSION)
RESULTS = $(RESULTS_SERIALIZED)

# Wrapper to be compared
WRAPPER = nsga2_ea_wrapper

# Temporary files
TEMP = __pycache__

# Tools
PYTHON = /usr/bin/env python3
RM = rm -rf

.PHONY: compare rank effect-size
compare rank effect-size: $(SCRIPT) $(WRAPPER)/$(RESULTS_SERIALIZED)
	@$(PYTHON) $(SCRIPT) $@
	@echo

$(WRAPPER)/$(RESULTS_SERIALIZED):
	@echo "Generating the $(WRAPPER) results ..."
	@echo
	@make -s -C $(WRAPPER)

.PHONY: clean
clean:
	@$(RM) $(TEMP)
	@make -s -C $(WRAPPER) clean
	@echo


.PHONY: deepclean
deepclean:
	@$(RM) $(TEMP)
	@make -s -C $(WRAPPER) deepclean



