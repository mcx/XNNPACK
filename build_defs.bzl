"""Build definitions and rules for XNNPACK."""

load("//:emscripten.bzl", "xnnpack_emscripten_benchmark_linkopts", "xnnpack_emscripten_deps", "xnnpack_emscripten_minimal_linkopts", "xnnpack_emscripten_test_linkopts")

def xnnpack_select_if(cond = None, val_true = [], val_false = []):
    if cond != None:
        return select({
            cond: val_true,
            "//conditions:default": val_false,
        })
    else:
        return val_true

def xnnpack_configurable_defines():
    return xnnpack_select_if(
        "//:cpuinfo_enabled",
        ["XNN_ENABLE_CPUINFO=1"],
        ["XNN_ENABLE_CPUINFO=0"],
    ) + xnnpack_select_if(
        "//:sparse_enabled",
        ["XNN_ENABLE_SPARSE=1"],
        ["XNN_ENABLE_SPARSE=0"],
    ) + xnnpack_select_if(
        "//:assembly_enabled",
        ["XNN_ENABLE_ASSEMBLY=1"],
        ["XNN_ENABLE_ASSEMBLY=0"],
    ) + xnnpack_select_if(
        "//:arm_fp16_scalar_enabled",
        ["XNN_ENABLE_ARM_FP16_SCALAR=1"],
        ["XNN_ENABLE_ARM_FP16_SCALAR=0"],
    ) + xnnpack_select_if(
        "//:arm_fp16_vector_enabled",
        ["XNN_ENABLE_ARM_FP16_VECTOR=1"],
        ["XNN_ENABLE_ARM_FP16_VECTOR=0"],
    ) + xnnpack_select_if(
        "//:arm_bf16_enabled",
        ["XNN_ENABLE_ARM_BF16=1"],
        ["XNN_ENABLE_ARM_BF16=0"],
    ) + xnnpack_select_if(
        "//:arm_dotprod_enabled",
        ["XNN_ENABLE_ARM_DOTPROD=1"],
        ["XNN_ENABLE_ARM_DOTPROD=0"],
    ) + xnnpack_select_if(
        "//:arm_i8mm_enabled",
        ["XNN_ENABLE_ARM_I8MM=1"],
        ["XNN_ENABLE_ARM_I8MM=0"],
    ) + xnnpack_select_if(
        "//:riscv_fp16_vector_enabled",
        ["XNN_ENABLE_RISCV_FP16_VECTOR=1"],
        ["XNN_ENABLE_RISCV_FP16_VECTOR=0"],
    ) + xnnpack_select_if(
        "//:avx512amx_enabled",
        ["XNN_ENABLE_AVX512AMX=1"],
        ["XNN_ENABLE_AVX512AMX=0"],
    ) + xnnpack_select_if(
        "//:avx512fp16_enabled",
        ["XNN_ENABLE_AVX512FP16=1"],
        ["XNN_ENABLE_AVX512FP16=0"],
    ) + xnnpack_select_if(
        "//:avx512bf16_enabled",
        ["XNN_ENABLE_AVX512BF16=1"],
        ["XNN_ENABLE_AVX512BF16=0"],
    ) + xnnpack_select_if(
        "//:avxvnni_enabled",
        ["XNN_ENABLE_AVXVNNI=1"],
        ["XNN_ENABLE_AVXVNNI=0"],
    ) + xnnpack_select_if(
        "//:avxvnniint8_enabled",
        ["XNN_ENABLE_AVXVNNIINT8=1"],
        ["XNN_ENABLE_AVXVNNIINT8=0"],
    ) + xnnpack_select_if(
        "//:avx512f_enabled",
        ["XNN_ENABLE_AVX512F=1"],
        ["XNN_ENABLE_AVX512F=0"],
    ) + xnnpack_select_if(
        "//:avx256skx_enabled",
        ["XNN_ENABLE_AVX256SKX=1"],
        ["XNN_ENABLE_AVX256SKX=0"],
    ) + xnnpack_select_if(
        "//:avx256vnni_enabled",
        ["XNN_ENABLE_AVX256VNNI=1"],
        ["XNN_ENABLE_AVX256VNNI=0"],
    ) + xnnpack_select_if(
        "//:avx256vnnigfni_enabled",
        ["XNN_ENABLE_AVX256VNNIGFNI=1"],
        ["XNN_ENABLE_AVX256VNNIGFNI=0"],
    ) + xnnpack_select_if(
        "//:avx512skx_enabled",
        ["XNN_ENABLE_AVX512SKX=1"],
        ["XNN_ENABLE_AVX512SKX=0"],
    ) + xnnpack_select_if(
        "//:avx512vbmi_enabled",
        ["XNN_ENABLE_AVX512VBMI=1"],
        ["XNN_ENABLE_AVX512VBMI=0"],
    ) + xnnpack_select_if(
        "//:avx512vnni_enabled",
        ["XNN_ENABLE_AVX512VNNI=1"],
        ["XNN_ENABLE_AVX512VNNI=0"],
    ) + xnnpack_select_if(
        "//:avx512vnnigfni_enabled",
        ["XNN_ENABLE_AVX512VNNIGFNI=1"],
        ["XNN_ENABLE_AVX512VNNIGFNI=0"],
    ) + xnnpack_select_if(
        "//:hvx_enabled",
        ["XNN_ENABLE_HVX=1"],
        ["XNN_ENABLE_HVX=0"],
    ) + xnnpack_select_if(
        "//:kleidiai_enabled",
        ["XNN_ENABLE_KLEIDIAI=1"],
        ["XNN_ENABLE_KLEIDIAI=0"],
    ) + xnnpack_select_if(
        "//:arm_sme_enabled",
        ["XNN_ENABLE_ARM_SME=1"],
        ["XNN_ENABLE_SRM_SME=0"],
    ) + xnnpack_select_if(
        "//:arm_sme2_enabled",
        ["XNN_ENABLE_ARM_SME2=1"],
        ["XNN_ENABLE_ARM_SME2=0"],
    ) + xnnpack_select_if(
        "//:wasm_revectorize_enabled",
        ["XNN_ENABLE_WASM_REVECTORIZE=1"],
        ["XNN_ENABLE_WASM_REVECTORIZE=0"],
    ) + xnnpack_slinky_defines()

def xnnpack_visibility():
    """Visibility of :XNNPACK target.

    All other targets have private visibility, and can not have external
    dependencies.
    """
    return ["//visibility:public"]

def xnnpack_min_size_copts():
    """Compiler flags for size-optimized builds."""
    return ["-Os"]

def xnnpack_gcc_std_copts():
    """GCC-like compiler flags to specify language standard for C sources."""
    return ["-std=c99"]

def xnnpack_msvc_std_copts():
    """MSVC compiler flags to specify language standard for C sources."""
    return ["/Drestrict="]

def xnnpack_std_copts():
    """Compiler flags to specify language standard for C sources."""
    return ["-std=c99"]

def xnnpack_std_cxxopts():
    """Compiler flags to specify language standard for C++ sources."""
    return ["-std=gnu++14"]

def xnnpack_std_c_defines():
    """Default defines used throughout the C sources.

    We need this for things like `struct timespec` that are not in c99, but
    are part of POSIX 1003.1b-1993."""

    return ["_DARWIN_C_SOURCE=1", "_POSIX_C_SOURCE=199309L"]

def xnnpack_test_deps_for_library():
    """Depencies needed for a library to use gunit."""
    return ["@com_google_googletest//:gtest_main"]

def xnnpack_optional_ruy_copts():
    """Compiler flags to optionally enable Ruy benchmarks."""
    return []

def xnnpack_optional_gemmlowp_copts():
    """Compiler flags to optionally enable Gemmlowp benchmarks."""
    return []

def xnnpack_optional_tflite_copts():
    """Compiler flags to optionally enable TensorFlow Lite benchmarks."""
    return []

def xnnpack_optional_dnnl_copts():
    """Compiler flags to optionally enable Intel DNNL benchmarks."""
    return []

def xnnpack_optional_ruy_deps():
    """Optional Ruy dependencies."""
    return []

def xnnpack_optional_gemmlowp_deps():
    """Optional Gemmlowp dependencies."""
    return []

def xnnpack_optional_tflite_deps():
    """Optional TensorFlow Lite dependencies."""
    return []

def xnnpack_optional_dnnl_deps():
    """Optional Intel DNNL dependencies."""
    return []

def xnnpack_slinky_deps():
    return []

def xnnpack_slinky_defines():
    return []

def xnnpack_if_kleidiai_enabled(enabled = [], not_enabled = []):
    return select({
        "//:kleidiai_enabled": enabled,
        "//conditions:default": not_enabled,
    })

def xnnpack_kleidiai_defines():
    return xnnpack_if_kleidiai_enabled(
        enabled = ["XNN_ENABLE_KLEIDIAI=1"],
        not_enabled = ["XNN_ENABLE_KLEIDIAI=0"],
    )

def xnnpack_slow_benchmark_tags():
    return ["manual"]

def xnnpack_cc_library(
        name,
        srcs = [],
        x86_srcs = [],
        aarch32_srcs = [],
        aarch64_srcs = [],
        hexagon_srcs = [],
        riscv_srcs = [],
        wasm_srcs = [],
        wasmsimd_srcs = [],
        wasmrelaxedsimd_srcs = [],
        linkopts = [],
        copts = xnnpack_std_copts(),
        gcc_copts = xnnpack_gcc_std_copts(),
        msvc_copts = xnnpack_msvc_std_copts(),
        mingw_copts = [],
        msys_copts = [],
        gcc_x86_copts = [],
        msvc_x86_32_copts = [],
        msvc_x86_64_copts = [],
        aarch32_copts = [],
        aarch64_copts = [],
        hexagon_copts = [],
        riscv_copts = [],
        wasm_copts = [],
        wasmsimd_copts = [],
        wasmrelaxedsimd_copts = [],
        optimized_copts = ["-O2"],
        hdrs = [],
        defines = [],
        includes = [],
        deps = [],
        visibility = None,
        testonly = False,
        **kwargs):
    """C/C++/assembly library with architecture-specific configuration.

    Define a static library with architecture- and instruction-specific
    source files and/or compiler flags.

    Args:
      name: The name of the library target to define.
      srcs: The list of architecture-independent source files.
      x86_srcs: The list of x86-specific source files.
      aarch32_srcs: The list of AArch32-specific source files.
      aarch64_srcs: The list of AArch64-specific source files.
      hexagon_srcs: The list of Hexagon-specific source files.
      riscv_srcs: The list of RISC-V-specific source files.
      wasm_srcs: The list of WebAssembly 1.0-specific source files.
      wasmsimd_srcs: The list of WebAssembly SIMD-specific source files.
      wasmrelaxedsimd_srcs: The list of WebAssembly Relaxed SIMD-specific
                            source files.
      copts: The list of compiler flags to use in all builds.
      gcc_copts: The list of compiler flags to use with GCC-like compilers.
      msvc_copts: The list of compiler flags to use with MSVC compiler.
      mingw_copts: The list of compiler flags to use with MinGW GCC compilers.
      msys_copts: The list of compiler flags to use with MSYS (Cygwin) GCC
                  compilers.
      gcc_x86_copts: The list of GCC-like compiler flags to use in x86 (32-bit
                     and 64-bit) builds.
      msvc_x86_32_copts: The list of MSVC compiler flags to use in x86 (32-bit)
                         builds.
      msvc_x86_64_copts: The list of MSVC compiler flags to use in x86 (64-bit)
                         builds.
      aarch32_copts: The list of compiler flags to use in AArch32 builds.
      aarch64_copts: The list of compiler flags to use in AArch64 builds.
      hexagon_copts: The list of compiler flags to use in hexagon builds.
      riscv_copts: The list of compiler flags to use in RISC-V builds.
      wasm_copts: The list of compiler flags to use in WebAssembly 1.0 builds.
      wasmsimd_copts: The list of compiler flags to use in WebAssembly SIMD
                      builds.
      wasmrelaxedsimd_copts: The list of compiler flags to use in WebAssembly
                             Relaxed SIMD builds.
      optimized_copts: The list of compiler flags to use in optimized builds.
                       Defaults to -O2.
      hdrs: The list of header files published by this library to be textually
            included by sources in dependent rules.
      defines: List of predefines macros to be added to the compile line.
      includes: List of include dirs to be added to the compile line.
      deps: The list of other libraries to be linked.
      visibility: The list of packages that can depend on this target.
      testonly: If True only testonly targets (such as tests) can depend on this.
      **kwargs: Other arguments to pass to the cc_library rule.
    """
    # Set the default defines.
    defines = defines or xnnpack_configurable_defines()

    native.cc_library(
        name = name,
        srcs = srcs + select({
            "//build_config:aarch32": aarch32_srcs,
            "//build_config:aarch64": aarch64_srcs,
            "//build_config:riscv": riscv_srcs,
            "//build_config:x86": x86_srcs,
            "//build_config:emscripten_wasm": wasm_srcs,
            "//build_config:emscripten_wasmsimd": wasmsimd_srcs,
            "//build_config:emscripten_wasmrelaxedsimd": wasmrelaxedsimd_srcs,
            "//conditions:default": [],
        }),
        copts = copts + select({
            "//build_config:linux_k8": gcc_x86_copts,
            "//build_config:linux_arm": aarch32_copts,
            "//build_config:linux_armeabi": aarch32_copts,
            "//build_config:linux_armhf": aarch32_copts,
            "//build_config:linux_armv7a": aarch32_copts,
            "//build_config:linux_arm64": aarch64_copts,
            "//build_config:macos_x86_64": gcc_x86_copts,
            "//build_config:macos_x86_64_legacy": gcc_x86_copts,
            "//build_config:macos_arm64": aarch64_copts,
            "//build_config:windows_x86_64_clang": ["/clang:" + opt for opt in gcc_x86_copts],
            "//build_config:windows_x86_64_mingw": mingw_copts + gcc_x86_copts,
            "//build_config:windows_x86_64_msys": msys_copts + gcc_x86_copts,
            "//build_config:windows_x86_64": msvc_x86_64_copts,
            "//build_config:android_armv7": aarch32_copts,
            "//build_config:android_arm64": aarch64_copts,
            "//build_config:android_x86": gcc_x86_copts,
            "//build_config:android_x86_64": gcc_x86_copts,
            "//build_config:ios_arm64": aarch64_copts,
            "//build_config:ios_arm64e": aarch64_copts,
            "//build_config:ios_sim_arm64": aarch64_copts,
            "//build_config:ios_x86_64": gcc_x86_copts,
            "//build_config:watchos_arm64_32": aarch64_copts,
            "//build_config:watchos_x86_64": gcc_x86_copts,
            "//build_config:tvos_arm64": aarch64_copts,
            "//build_config:tvos_x86_64": gcc_x86_copts,
            "//build_config:emscripten_wasm": wasm_copts,
            "//build_config:emscripten_wasmsimd": wasmsimd_copts,
            "//build_config:emscripten_wasmrelaxedsimd": wasmrelaxedsimd_copts,
            "//conditions:default": [],
        }) + select({
            "//build_config:windows_x86_64_clang": ["/clang:" + opt for opt in gcc_copts],
            "//build_config:windows_x86_64_mingw": gcc_copts,
            "//build_config:windows_x86_64_msys": gcc_copts,
            "//build_config:windows_x86_64": msvc_copts,
            "//conditions:default": gcc_copts,
        }) + select({
            "//:optimized_build": optimized_copts,
            "//conditions:default": [],
        }),
        defines = xnnpack_std_c_defines() + defines,
        deps = deps,
        includes = ["include", "src"] + includes,
        linkstatic = True,
        linkopts = select({
            "//build_config:linux_k8": ["-lpthread"],
            "//build_config:linux_arm": ["-lpthread"],
            "//build_config:linux_armeabi": ["-lpthread"],
            "//build_config:linux_armhf": ["-lpthread"],
            "//build_config:linux_armv7a": ["-lpthread"],
            "//build_config:linux_arm64": ["-lpthread"],
            "//build_config:android": ["-lm"],
            "//conditions:default": [],
        }),
        hdrs = hdrs,
        visibility = visibility,
        testonly = testonly,
        **kwargs,
    )

def xnnpack_cxx_library(name, copts = xnnpack_std_cxxopts(), gcc_copts = [], msvc_copts = [], **kwargs):
    xnnpack_cc_library(
        name,
        copts = copts,
        gcc_copts = gcc_copts,
        msvc_copts = msvc_copts,
        **kwargs
    )

def xnnpack_unit_test(name, srcs, copts = [], mingw_copts = [], msys_copts = [], deps = [], tags = [], linkopts = [], defines = [], timeout = "short", shard_count = 1, **kwargs):
    """Unit test binary based on Google Test.

    Args:
      name: The name of the test target to define.
      srcs: The list of source and header files.
      copts: The list of additional compiler flags for the target.
      mingw_copts: The list of compiler flags to use with MinGW GCC compilers.
      msys_copts: The list of compiler flags to use with MSYS (Cygwin) GCC compilers.
      deps: The list of additional libraries to be linked. Google Test library
            (with main() function) is always added as a dependency and does not
            need to be explicitly specified.
      linkopts: The list of linking options
      defines: List of predefines macros to be added to the compile line.
      tags: List of arbitrary text tags.
      timeout: How long the test is expected to run before returning.
      shard_count: Specifies the number of parallel shards to use to run the test.
      **kwargs: Other arguments to pass to the cc_test rule.
    """

    # Set the default defines.
    defines = defines or xnnpack_configurable_defines()

    native.cc_test(
        name = name,
        srcs = srcs,
        copts = xnnpack_std_cxxopts() + select({
            "//build_config:windows_x86_64_mingw": mingw_copts,
            "//build_config:windows_x86_64_msys": msys_copts,
            "//conditions:default": [],
        }) + select({
            "//build_config:windows_x86_64_clang": ["/clang:-Wno-unused-function"],
            "//build_config:windows_x86_64_mingw": ["-Wno-unused-function"],
            "//build_config:windows_x86_64_msys": ["-Wno-unused-function"],
            "//build_config:windows_x86_64": [],
            "//conditions:default": ["-Wno-unused-function"],
        }) + copts,
        linkopts = select({
            "//build_config:emscripten": xnnpack_emscripten_test_linkopts(),
            "//conditions:default": [],
        }) + linkopts,
        linkstatic = True,
        defines = defines,
        deps = [
            "@com_google_googletest//:gtest_main",
        ] + deps + select({
            "//build_config:emscripten": xnnpack_emscripten_deps(),
            "//conditions:default": [],
        }),
        tags = tags,
        timeout = timeout,
        shard_count = shard_count,
        **kwargs,
    )

def xnnpack_binary(name, srcs, copts = [], deps = [], linkopts = []):
    """Minimal binary

    Args:
      name: The name of the binary target to define.
      srcs: The list of source and header files.
      copts: The list of additional compiler flags for the target.
      deps: The list of libraries to be linked.
      linkopts: The list of additional linker options
    """
    native.cc_binary(
        name = name,
        srcs = srcs,
        copts = copts,
        linkopts = select({
            "//build_config:emscripten": xnnpack_emscripten_minimal_linkopts(),
            "//conditions:default": [],
        }) + linkopts,
        linkstatic = True,
        deps = deps,
    )

def xnnpack_benchmark(name, srcs, copts = [], deps = [], tags = [], defines = []):
    """Microbenchmark binary based on Google Benchmark

    Args:
      name: The name of the binary target to define.
      srcs: The list of source and header files.
      copts: The list of additional compiler flags for the target.
      deps: The list of additional libraries to be linked. Google Benchmark
            library is always added as a dependency and does not need to be
            explicitly specified.
      tags: The list of arbitrary text tags.
      defines: The list of arbitrary defines tags.
    """
    # Set the default defines.
    defines = defines or xnnpack_configurable_defines()

    native.cc_test(
        name = name,
        srcs = srcs,
        copts = xnnpack_std_cxxopts() + select({
            "//build_config:windows_x86_64_clang": ["/clang:-Wno-unused-function"],
            "//build_config:windows_x86_64_mingw": ["-Wno-unused-function"],
            "//build_config:windows_x86_64_msys": ["-Wno-unused-function"],
            "//build_config:windows_x86_64": [],
            "//conditions:default": ["-Wno-unused-function"],
        }) + copts,
        linkopts = select({
            "//build_config:emscripten": xnnpack_emscripten_benchmark_linkopts(),
            "//build_config:windows_x86_64_mingw": ["-lshlwapi"],
            "//build_config:windows_x86_64_msys": ["-lshlwapi"],
            "//conditions:default": [],
        }),
        linkstatic = True,
        deps = [
            "@com_google_benchmark//:benchmark",
        ] + deps + select({
            "//build_config:emscripten": xnnpack_emscripten_deps(),
            "//conditions:default": [],
        }),
        tags = tags,
        defines = defines,
        args = ["--benchmark_min_time=1x"],
    )

SrcListInfo = provider("A list of source files.", fields = {"srcs": "sources"})

def _source_list_aspect_impl(_target, ctx):
    srcs = []
    if hasattr(ctx.rule.attr, "srcs"):
        srcs += [s for src in ctx.rule.attr.srcs for s in src.files.to_list()]
    transitive = [dep[SrcListInfo].srcs for dep in ctx.rule.attr.deps]
    return [SrcListInfo(srcs = depset(srcs, transitive = transitive))]

source_list_aspect = aspect(
    implementation = _source_list_aspect_impl,
    attr_aspects = ["deps"],
)

def _transitive_source_list_rule_impl(ctx):
    get_repo_name = lambda x: getattr(x, "repo_name", getattr(x, "workspace_name"))
    files = [p for dep in ctx.attr.deps for p in dep[SrcListInfo].srcs.to_list() if get_repo_name(p.owner) == get_repo_name(ctx.label) and p.owner.package.startswith(ctx.label.package)]
    return [DefaultInfo(files = depset(files + ctx.files.srcs))]

xnnpack_transitive_source_list = rule(
    implementation = _transitive_source_list_rule_impl,
    attrs = {
        "deps": attr.label_list(aspects = [source_list_aspect]),
        "srcs": attr.label_list(allow_files = True),
    },
)

def _source_list_file_rule_impl(ctx):
    output_file = ctx.actions.declare_file(ctx.label.name + ".list")
    ctx.actions.write(
        output = output_file,
        content = "\n".join([s.path for s in ctx.files.srcs]),
    )
    return [DefaultInfo(files = depset([output_file]))]

xnnpack_source_list_file = rule(
    implementation = _source_list_file_rule_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
    },
)
