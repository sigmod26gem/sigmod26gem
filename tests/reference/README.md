# Reference Sources

`gem_example.cpp` preserves the upstream GEM example for equivalence tests.

The root CMake project builds `gem_upstream_reference` against the active kernel in `src/graph/` and numerical code in `src/distance/`. The example's main function is not executed.

`preprocess/` contains the original coarse clustering and TF-IDF assignment scripts for comparison. Production targets do not include this directory.
