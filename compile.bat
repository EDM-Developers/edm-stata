cmake -B build -S .
cmake --build build --config release
cmake --build build --config release --target format
cmake --build build --config release --target install
