# financebench_runner

CONFIG := .config

.PHONY: help menuconfig

help:
	@echo "financebench_runner - Available targets:"
	@echo ""
	@echo "  menuconfig  - Interactive configuration GUI (edits $(CONFIG))"
	@echo ""

menuconfig:
	@python3 menuconfig_gui.py $(CONFIG)
