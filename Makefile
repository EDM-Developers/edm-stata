all: build
	cmake --build build -- $(MFLAGS) install

build:
	cmake -B build -S . ${CMAKE_OPTS} ${CMAKE_BUILD_TYPE}
	ln -sf build/compile_commands.json

release: CMAKE_BUILD_TYPE = -DCMAKE_BUILD_TYPE=Release
release: build all

.PHONY: clean
clean:
	rm -rf build
	rm -f compile_commands.json
