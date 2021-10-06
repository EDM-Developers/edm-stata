cmake -B %EDM_BUILD_DIR% -S .
cmake --build %EDM_BUILD_DIR% --config release --target edm_plugin
cmake --build %EDM_BUILD_DIR% --config release --target install
cmake --build %EDM_BUILD_DIR% --target format