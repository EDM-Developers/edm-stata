EDM_BUILD_DIR ?= build

all: $(EDM_BUILD_DIR)/CMakeCache.txt release install

$(EDM_BUILD_DIR)/CMakeCache.txt:
	cmake -B $(EDM_BUILD_DIR) -S .

.PHONY: release
release: $(EDM_BUILD_DIR)/CMakeCache.txt
	cmake --build $(EDM_BUILD_DIR) --config release

.PHONY: install
install:
	cmake --build $(EDM_BUILD_DIR) --config release --target install

.PHONY: clean
clean:
	rm -rf $(EDM_BUILD_DIR)
	rm -f compile_commands.json

.PHONY: format
format:
	cmake --build $(EDM_BUILD_DIR) --target format