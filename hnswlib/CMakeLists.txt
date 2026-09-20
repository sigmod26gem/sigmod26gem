cmake_minimum_required(VERSION 3.18)
project(gem_legacy_entry LANGUAGES CXX)

# Keep the original README command working; all implementation lives in src/.
set(GEM_BUILD_LEGACY_EXAMPLE ON CACHE BOOL "Build the original GEM example" FORCE)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." gem)
