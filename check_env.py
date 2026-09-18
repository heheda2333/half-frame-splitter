#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动前环境检查：确认 Python 及所需的 Pillow、numpy 两个库均可正常导入。

所有中文提示集中在本文件中输出，以避免批处理文件（.bat）因编码问题显示为乱码。
"""

from __future__ import annotations

import sys


def main() -> int:
    print("=" * 60)
    print("  半格相机照片自动切割工具")
    print("=" * 60)
    print("Python 版本：{}".format(sys.version.split()[0]))

    missing = []
    for name, pkg in (("PIL", "Pillow"), ("numpy", "numpy")):
        try:
            mod = __import__(name)
            print("  [正常] {:<8}{}".format(pkg, getattr(mod, "__version__", "版本未知")))
        except Exception as exc:  # noqa: BLE001
            print("  [缺失] {:<8}{}".format(pkg, exc))
            missing.append(pkg)

    if missing:
        print()
        print("缺失依赖：{}".format("、".join(missing)))
        print("请运行本目录中的【安装依赖.bat】，或手动执行：")
        print("    python -m pip install -r requirements.txt")
        return 1

    print("环境检查通过。")
    print("  待处理照片请置于  input 目录")
    print("  切割结果将写入  output 目录")
    print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
