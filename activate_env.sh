# 在项目目录执行：source activate_env.sh
_SUPERSHIP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_SUPERSHIP_ROOT/.venv/bin/activate"
_SUPERSHIP_LIBS=""
for _SUPERSHIP_LIB in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib; do
    if [ -d "$_SUPERSHIP_LIB" ]; then
        _SUPERSHIP_LIBS="${_SUPERSHIP_LIBS}${_SUPERSHIP_LIB}:"
    fi
done
export LD_LIBRARY_PATH="${_SUPERSHIP_LIBS}${LD_LIBRARY_PATH:-}"
# Prefer the locally validated profiler when installed.
_SUPERSHIP_NSYS_BIN="$_SUPERSHIP_ROOT/.local-tools/nsight-systems/opt/nvidia/nsight-systems/2024.6.2/bin"
if [ -x "$_SUPERSHIP_NSYS_BIN/nsys" ]; then
    case ":$PATH:" in
        *":$_SUPERSHIP_NSYS_BIN:"*) ;;
        *) export PATH="$_SUPERSHIP_NSYS_BIN:$PATH" ;;
    esac
fi
unset _SUPERSHIP_ROOT _SUPERSHIP_LIBS _SUPERSHIP_LIB _SUPERSHIP_NSYS_BIN
