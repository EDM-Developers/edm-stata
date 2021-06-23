cmake -B %EDM_BUILD_DIR% -S .
cmake --build %EDM_BUILD_DIR% --config release
cmake --build %EDM_BUILD_DIR% --config release --target format
cmake --build %EDM_BUILD_DIR% --config release --target install
