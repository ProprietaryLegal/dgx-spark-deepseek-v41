Build-time source fix applied by scripts/build/build_vllm_in_container.sh (idempotent):
CMakeLists.txt: append 12.1 to every CUDA_SUPPORTED_ARCHS list that ends at 12.0 (the CUDA >= 13.0 and >= 13.4 lists),
so TORCH_CUDA_ARCH_LIST=12.1a is not narrowed to 12.0 (sm_120a cubins do not load on GB10). Upstream PR candidate.
