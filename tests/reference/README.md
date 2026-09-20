# Reference Sources

These files preserve the upstream examples and experiments for comparison.

The root CMake project builds `gem_upstream_reference` from the original GEM example. Its includes point to the active kernel in `src/graph/` and numerical code in `src/distance/`. `GEM_BUILD_LEGACY_EXAMPLE=ON` also builds its original executable for the unchanged root README.

The remaining hnswlib examples, Python packaging files, glass experiments, and preprocessing scripts are source references. Their nested build files are retained as historical input and are not supported build entry points. Production targets do not include this directory.
