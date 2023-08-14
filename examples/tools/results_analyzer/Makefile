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

# Compare script
COMPARE = compare.py

# Rank script
RANK = rank.py

# Results filename
RESULTS = results.gz

# Wrapper to be compared
WRAPPER = nsga2_ea_wrapper

# Temporary files
TEMP = __pycache__

# Tools
PYTHON = /usr/bin/env python3
RM = rm -rf


.PHONY: compare
compare : $(WRAPPER)/$(RESULTS)
	@echo "Comparing $(WRAPPER) ..."
	@$(PYTHON) $(COMPARE)
	@echo

.PHONY: rank
rank : $(WRAPPER)/$(RESULTS)
	@echo "Ranking $(WRAPPER) ..."
	@echo
	@$(PYTHON) $(RANK)
	@echo

$(WRAPPER)/$(RESULTS):
	@echo "Generating the $(WRAPPER) results ..."
	@echo
	@make -s -C $(WRAPPER)

.PHONY: clean
clean:
	@$(RM) $(TEMP)
	@make -s -C $(WRAPPER) clean
	@echo


.PHONY: distclean
distclean:
	@$(RM) $(TEMP)
	@make -s -C $(WRAPPER) distclean



