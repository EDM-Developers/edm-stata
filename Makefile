all: build/CMakeCache.txt release install

build/CMakeCache.txt:
	cmake -B build -S .

.PHONY: release
release: build/CMakeCache.txt
	cmake --build build --config release

.PHONY: install
install:
	cmake --build build --config release --target install

.PHONY: clean
clean:
	rm -rf build
	rm -f compile_commands.json

.PHONY: format
format:
	cmake --build build --target format