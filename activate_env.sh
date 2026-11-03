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
unset _SUPERSHIP_ROOT _SUPERSHIP_LIBS _SUPERSHIP_LIB
