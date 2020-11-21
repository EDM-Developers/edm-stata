all: build release
	cmake --build build --config release --target install

build:
	cmake -B build -S .

.PHONY: release
release: build
	cmake --build build --config release

.PHONY: clean
clean:
	rm -rf build
	rm -f compile_commands.json

.PHONY: format
format:
	cmake --build build --target format