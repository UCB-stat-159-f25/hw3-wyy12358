# ===============================
# Makefile for STAT159/259 HW3
# Author: Yuyang Wu
# ===============================

# Environment name
ENV_NAME = hw3

# ---------- Targets ----------

# Create or update the conda environment
env:
    @echo ">>> Creating or updating the '$(ENV_NAME)' environment..."
	@conda env update -n $(ENV_NAME) -f environment.yml --prune || conda env create -n $(ENV_NAME) -f environment.yml
	@echo ">>> Environment setup complete."

# Build the MyST site into HTML (local view only)
html:
	@echo ">>> Building MyST site into _build/html ..."
	myst build --html
	@echo ">>> Build complete. You can view the site locally via:"
	@echo "    python -m http.server -d _build/html 8000"

# Clean generated files
clean:
	@echo ">>> Cleaning up generated directories..."
	rm -rf figures audio _build
	@echo ">>> Cleanup complete."