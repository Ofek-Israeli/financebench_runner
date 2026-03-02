# financebench_runner — same menuconfig style as compressor_2 (ncurses Kconfig)

KCONFIG := Kconfig
CONFIG := .config
DEFCONFIG := defconfig

.PHONY: help menuconfig defconfig savedefconfig

help:
	@echo "financebench_runner - Available targets:"
	@echo ""
	@echo "  Configuration (same as compressor_2):"
	@echo "    menuconfig     - Interactive configuration menu (ncurses)"
	@echo "    defconfig      - Load default configuration"
	@echo "    savedefconfig  - Save current config as defconfig"
	@echo ""

menuconfig:
	@if [ -f $(CONFIG) ] && grep -q '^model_id:' $(CONFIG) 2>/dev/null; then \
		cp $(CONFIG) .config.yaml.bak; \
		echo "Backed up YAML .config to .config.yaml.bak"; \
		cp $(DEFCONFIG) $(CONFIG); \
	fi; \
	if [ ! -f $(CONFIG) ]; then cp $(DEFCONFIG) $(CONFIG); fi; \
	python3 -c "from kconfiglib import Kconfig; import menuconfig; \
		import os; os.environ['KCONFIG_CONFIG'] = '$(CONFIG)'; \
		menuconfig.menuconfig(Kconfig('$(KCONFIG)'))"

defconfig:
	@if [ -f $(DEFCONFIG) ]; then \
		cp $(DEFCONFIG) $(CONFIG); \
		echo "Loaded default configuration from $(DEFCONFIG)"; \
	else \
		echo "Error: $(DEFCONFIG) not found"; \
		exit 1; \
	fi

savedefconfig:
	@if [ -f $(CONFIG) ]; then \
		cp $(CONFIG) $(DEFCONFIG); \
		echo "Saved current configuration to $(DEFCONFIG)"; \
	else \
		echo "Error: $(CONFIG) not found"; \
		exit 1; \
	fi
