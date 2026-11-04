# 运行参数

主线默认参数在 `../fastest.env`。下列 JSON 仅含运行环境参数，已去除原机器 Python 和缓存路径；使用时选择对应代码分支。

| 参数 | 对应分支 | 用途 |
| --- | --- | --- |
| eightcore-thread6.json | main / exp/supply-eightcore-20261027 | 固定8核新候选，独立8对确认快15.35%，原始三轮0.256022902秒已Accepted（timing parsed only） |
| thread100-light.json | main / exp/continued-sweep-20261026 | 原10线程参考集候选，保留作对照及其他环境复现 |
| exit-thp544.json | exp/exit-rootcause-20261026 | 已测完整 validation 的 THP 版本 |
| kernel544.json | exp/kernel-sweep-20261024 | 历史内核融合版本 |
| conservative640.json / .env | exp/rtx4090-e2e | 640 输入、完整 JPEG 解码的保守版本 |

历史分支不会自动读取这些 JSON，需将键值作为环境变量传入。保守版本可在对应分支执行 `source configs/presets/conservative640.env` 后运行 worker。较早分支仍使用根目录评测脚本；主线使用 `tools/benchmark/`。历史版本协议与依赖以该分支源码为准，结果不能跨比较组直接排名。
