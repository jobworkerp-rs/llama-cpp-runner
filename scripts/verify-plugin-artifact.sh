#!/usr/bin/env bash
# Usage: verify-plugin-artifact.sh <library> <linux-x86_64|darwin-arm64>
#
# Smoke check of a built plugin, run by the public release workflow before
# anything is published. It does not load the plugin (no model or GPU is
# needed); it checks what makes a release unusable on the user's machine:
# file format, the FFI entry points the jobworkerp loader resolves, the
# platform floor the binary demands, and on macOS the code signature.
#
# Linux requires MAX_GLIBC and MAX_GLIBCXX (e.g. 2.39 and 3.4.33): the highest
# GLIBC / GLIBCXX symbol versions the caller allows the library to require
# (llama.cpp is C++ and links libstdc++ dynamically). Optional REQUIRED_NEEDED
# lists sonames that must be dynamic dependencies, e.g. libcuda.so.1 proves a
# CUDA build really carries the CUDA backend.
# macOS requires MACOSX_DEPLOYMENT_TARGET (e.g. 14.0) for the same reason.
#
# Kept compatible with the bash 3.2 shipped with macOS.
set -euo pipefail

lib=${1-}
platform=${2-}
symbols="load_multi_method_plugin_v2 free_multi_method_plugin_v2"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

# True when dotted version $1 is greater than $2. Compared field by field in
# awk because `sort -V` is not available on every macOS release.
version_gt() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    na = split(a, x, "."); nb = split(b, y, "."); n = (na > nb) ? na : nb
    for (i = 1; i <= n; i++) {
      if ((x[i] + 0) > (y[i] + 0)) exit 0
      if ((x[i] + 0) < (y[i] + 0)) exit 1
    }
    exit 1
  }'
}

# check_symver_floor <GLIBC|GLIBCXX> <max>: fail when the library requires a
# newer symbol version of that runtime than <max>. A pure C library has no
# GLIBCXX symbols, so finding none is not an error.
check_symver_floor() {
  local runtime=$1 max=$2 required
  required=$(objdump -T "${lib}" | { grep -o "${runtime}_[0-9][0-9.]*" || true; } \
    | sed "s/${runtime}_//" | sort -uV | tail -n 1)
  echo "highest required ${runtime}: ${required:-none}"
  if [[ -n "${required}" ]] && version_gt "${required}" "${max}"; then
    die "requires ${runtime} ${required}, above MAX_${runtime} ${max}"
  fi
}

[[ -n "${lib}" && -n "${platform}" ]] || die "usage: $0 <library> <linux-x86_64|darwin-arm64>"
[[ -f "${lib}" ]] || die "no such file: ${lib}"

kind=$(file -b "${lib}")
echo "file: ${kind}"

case "${platform}" in
  linux-x86_64)
    [[ "${kind}" == *"ELF 64-bit LSB shared object"*"x86-64"* ]] \
      || die "not an x86-64 ELF shared object"
    [[ -n "${MAX_GLIBC-}" ]] || die "MAX_GLIBC must be set for ${platform}"
    [[ -n "${MAX_GLIBCXX-}" ]] || die "MAX_GLIBCXX must be set for ${platform}"
    exported=$(nm -D --defined-only "${lib}" | awk '{ print $NF }')
    prefix=
    echo "dynamic dependencies:"
    ldd "${lib}" || true
    check_symver_floor GLIBC "${MAX_GLIBC}"
    check_symver_floor GLIBCXX "${MAX_GLIBCXX}"
    needed=$(objdump -p "${lib}" | awk '$1 == "NEEDED" { print $2 }')
    for dep in ${needed}; do
      # Unloadable on the documented runtimes: Rust std as a shared library (a
      # Rust dylib build), NCCL (unused, absent from the runtime images) and
      # the CUDA toolkit libraries, which are linked statically.
      case "${dep}" in
        libstd-* | libnccl* | libcudart* | libcublas*) die "forbidden dynamic dependency: ${dep}" ;;
      esac
    done
    for dep in ${REQUIRED_NEEDED-}; do
      printf '%s\n' "${needed}" | grep -qxF "${dep}" || die "missing required dynamic dependency: ${dep}"
      echo "required dependency: ${dep}"
    done
    ;;
  darwin-arm64)
    [[ "${kind}" == *"Mach-O 64-bit dynamically linked shared library arm64"* ]] \
      || die "not an arm64 Mach-O dylib"
    [[ -n "${MACOSX_DEPLOYMENT_TARGET-}" ]] || die "MACOSX_DEPLOYMENT_TARGET must be set for ${platform}"
    exported=$(nm -gU "${lib}" | awk '{ print $NF }')
    # Mach-O prefixes C symbols with an underscore.
    prefix=_
    echo "dynamic dependencies:"
    otool -L "${lib}"
    minos=$(otool -l "${lib}" | awk '$1 == "minos" { print $2; exit }')
    echo "minimum macOS: ${minos:-unknown}"
    [[ -n "${minos}" ]] || die "no LC_BUILD_VERSION minos found"
    if version_gt "${minos}" "${MACOSX_DEPLOYMENT_TARGET}"; then
      die "requires macOS ${minos}, above MACOSX_DEPLOYMENT_TARGET ${MACOSX_DEPLOYMENT_TARGET}"
    fi
    codesign -v --verbose=2 "${lib}" || die "code signature is invalid"
    ;;
  *)
    die "unknown platform: ${platform}"
    ;;
esac

for sym in ${symbols}; do
  printf '%s\n' "${exported}" | grep -qx "${prefix}${sym}" || die "missing exported symbol: ${sym}"
  echo "exported: ${sym}"
done

echo "OK: ${lib} (${platform})"
