#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""半格相机照片切割工具 —— 图形界面

运行本目录中的“运行-图形界面.bat”，或执行 python gui.py，即可打开操作窗口：
选定输入目录与输出目录后点击“开始处理”，切割结果将写入输出目录。
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox, ttk
from tkinter import END, NORMAL, DISABLED, W, E, N, S, Text
from tkinter import font as tkfont
import tkinter as tk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from half_frame_splitter import (  # noqa: E402
    DEFAULT_INPUT, DEFAULT_OUTPUT, HALF_FRAME_ASPECT, MIN_PROMINENCE,
    IMAGE_EXTS, Options, iter_input_files, output_dir_for, process_batch,
    relative_subdir,
)

SETTINGS_FILE = Path(__file__).resolve().parent / "_gui_settings.json"

ORIENT_LABELS = {
    "自动判断": "auto",
    "左右两幅（左 | 右）": "lr",
    "上下两幅（上 / 下）": "tb",
}
ORIENT_KEYS = {v: k for k, v in ORIENT_LABELS.items()}

EDGE_LABELS = {
    "整段舍去间隙（推荐）": "gap",
    "自间隙正中切开": "center",
}
EDGE_KEYS = {v: k for k, v in EDGE_LABELS.items()}


class App:
    def __init__(self, root: Tk):
        self.root = root
        self.queue: "queue.Queue" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_flag = threading.Event()

        root.title("半格相机照片切割工具")
        root.minsize(760, 560)
        self._set_style()

        cfg = self._load_settings()

        self.var_in = StringVar(value=cfg.get("input", str(DEFAULT_INPUT)))
        self.var_out = StringVar(value=cfg.get("output", str(DEFAULT_OUTPUT)))
        self.var_orient = StringVar(value=ORIENT_KEYS.get(cfg.get("orient", "auto"), "自动判断"))
        self.var_edge = StringVar(value=EDGE_KEYS.get(cfg.get("edge", "gap"),
                                                     "整段舍去间隙（推荐）"))
        self.var_trim = BooleanVar(value=cfg.get("trim", False))
        self.var_aspect = BooleanVar(value=cfg.get("aspect_fix", False))
        self.var_subdirs = BooleanVar(value=cfg.get("subdirs", True))
        self.var_overwrite = BooleanVar(value=cfg.get("overwrite", False))
        self.var_skip = BooleanVar(value=cfg.get("skip_uncertain", False))
        self.var_debug = BooleanVar(value=cfg.get("debug", True))
        self.var_quality = StringVar(value=str(cfg.get("quality", 95)))
        self.var_status = StringVar(
            value="就绪：将照片或照片文件夹放入输入目录，然后点击“开始处理”"
                  "（原图只读，不会修改或删除）")
        self.var_count = StringVar(value="")

        self._build()
        self._drain_queue()

    # ------------------------------------------------------------------ UI --
    def _set_style(self) -> None:
        try:  # 在高分辨率屏幕上显示更清晰（失败时不影响功能）
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except Exception:
            pass
        fams = set(tkfont.families())
        for name in ("Microsoft YaHei UI", "微软雅黑", "Microsoft YaHei"):
            if name in fams:
                style.configure(".", font=(name, 10))
                break

    def _build(self) -> None:
        pad = dict(padx=8, pady=4)

        # ---- 目录 ----
        box_dir = ttk.LabelFrame(self.root, text=" 目录 ")
        box_dir.pack(fill="x", padx=10, pady=(10, 4))

        ttk.Label(box_dir, text="输入目录（待处理照片）").grid(row=0, column=0, sticky=W, **pad)
        ttk.Entry(box_dir, textvariable=self.var_in).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(box_dir, text="浏览…", width=8,
                   command=lambda: self._pick_dir(self.var_in)).grid(row=0, column=2, **pad)
        ttk.Button(box_dir, text="打开", width=6,
                   command=lambda: self._open_dir(self.var_in.get())).grid(row=0, column=3, **pad)

        ttk.Label(box_dir, text="输出目录（切割结果）").grid(row=1, column=0, sticky=W, **pad)
        ttk.Entry(box_dir, textvariable=self.var_out).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(box_dir, text="浏览…", width=8,
                   command=lambda: self._pick_dir(self.var_out)).grid(row=1, column=2, **pad)
        ttk.Button(box_dir, text="打开", width=6,
                   command=lambda: self._open_dir(self.var_out.get())).grid(row=1, column=3, **pad)
        box_dir.columnconfigure(1, weight=1)

        # ---- 选项 ----
        box_opt = ttk.LabelFrame(self.root, text=" 选项 ")
        box_opt.pack(fill="x", padx=10, pady=4)

        ttk.Label(box_opt, text="画面排布").grid(row=0, column=0, sticky=W, **pad)
        ttk.Combobox(box_opt, textvariable=self.var_orient, state="readonly", width=22,
                     values=list(ORIENT_LABELS.keys())).grid(row=0, column=1, sticky=W, **pad)

        ttk.Label(box_opt, text="间隙处理").grid(row=0, column=2, sticky=W, **pad)
        ttk.Combobox(box_opt, textvariable=self.var_edge, state="readonly", width=22,
                     values=list(EDGE_LABELS.keys())).grid(row=0, column=3, sticky=W, **pad)

        ttk.Label(box_opt, text="JPEG 质量").grid(row=0, column=4, sticky=E, **pad)
        ttk.Spinbox(box_opt, from_=60, to=100, width=5,
                    textvariable=self.var_quality).grid(row=0, column=5, sticky=W, **pad)

        ttk.Checkbutton(box_opt, text="预处理：去除扫描边框（适用于含片基边框的整卷扫描）",
                        variable=self.var_trim).grid(row=1, column=0, columnspan=3, sticky=W, **pad)
        ttk.Checkbutton(box_opt, text="按半格画幅比例（2:3）修正构图",
                        variable=self.var_aspect).grid(row=1, column=3, columnspan=3, sticky=W, **pad)

        ttk.Checkbutton(box_opt, text="识别子文件夹（在输出目录生成同名文件夹）",
                        variable=self.var_subdirs
                        ).grid(row=2, column=0, columnspan=3, sticky=W, **pad)
        ttk.Checkbutton(box_opt, text="覆盖同名文件", variable=self.var_overwrite
                        ).grid(row=2, column=3, columnspan=2, sticky=W, **pad)
        ttk.Checkbutton(box_opt, text="置信度不足时跳过", variable=self.var_skip
                        ).grid(row=3, column=0, columnspan=2, sticky=W, **pad)
        ttk.Checkbutton(box_opt, text="输出切割线预览图（保存于结果目录的 _debug 中）",
                        variable=self.var_debug
                        ).grid(row=3, column=2, columnspan=3, sticky=W, **pad)

        # ---- 操作 ----
        box_run = ttk.Frame(self.root)
        box_run.pack(fill="x", padx=10, pady=6)
        self.btn_start = ttk.Button(box_run, text="开始处理", command=self.on_start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(box_run, text="停止", command=self.on_stop, state=DISABLED)
        self.btn_stop.pack(side="left", padx=6)
        ttk.Button(box_run, text="统计数量",
                   command=self.on_count).pack(side="left", padx=6)
        ttk.Button(box_run, text="清除日志",
                   command=lambda: self.txt.delete("1.0", END)).pack(side="left", padx=6)
        ttk.Label(box_run, textvariable=self.var_count).pack(side="left", padx=10)

        self.bar = ttk.Progressbar(self.root, mode="determinate", maximum=100)
        self.bar.pack(fill="x", padx=10)
        ttk.Label(self.root, textvariable=self.var_status, foreground="#0a5"
                  ).pack(fill="x", padx=10, pady=(2, 4))

        # ---- 日志 ----
        box_log = ttk.LabelFrame(self.root, text=" 日志 ")
        box_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.txt = Text(box_log, height=14, wrap="word", background="#1e1e1e",
                        foreground="#dcdcdc", insertbackground="#dcdcdc")
        sb = ttk.Scrollbar(box_log, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        self.txt.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        sb.pack(side="right", fill="y", padx=(0, 6), pady=6)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------- helpers --
    def _pick_dir(self, var: StringVar) -> None:
        cur = var.get() or str(Path.home())
        d = filedialog.askdirectory(initialdir=cur, title="选择目录")
        if d:
            var.set(os.path.normpath(d))

    def _open_dir(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            messagebox.showwarning("提示", "该目录不存在：\n{}".format(path))
            return
        try:
            os.startfile(str(p))          # Windows
        except Exception:
            subprocess.Popen(["explorer", str(p)])

    def log(self, msg: str) -> None:
        self.queue.put(("log", msg))

    # ---------------------------------------------------------------- run ---
    def _make_options(self) -> Options:
        try:
            q = int(float(self.var_quality.get()))
        except Exception:
            q = 95
        return Options(
            input_dir=Path(self.var_in.get()),
            output_dir=Path(self.var_out.get()),
            orient=ORIENT_LABELS.get(self.var_orient.get(), "auto"),
            edge=EDGE_LABELS.get(self.var_edge.get(), "gap"),
            frame_aspect=HALF_FRAME_ASPECT,
            trim=bool(self.var_trim.get()),
            aspect="auto" if self.var_aspect.get() else "off",
            quality=max(60, min(100, q)),
            subdirs=bool(self.var_subdirs.get()),
            overwrite=bool(self.var_overwrite.get()),
            skip_uncertain=bool(self.var_skip.get()),
            min_confidence=MIN_PROMINENCE,
            debug=bool(self.var_debug.get()),
        )

    def on_count(self) -> None:
        try:
            opts = self._make_options()
            files = iter_input_files(Path(opts.input_dir), opts.subdirs)
        except Exception as exc:
            messagebox.showerror("错误", str(exc))
            return
        subs = {relative_subdir(f, opts) for f in files}
        subs.discard(Path())
        self.var_count.set("待处理 {} 个文件{}".format(
            len(files), "（分布于 {} 个子文件夹）".format(len(subs)) if subs else ""))

    def on_start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        opts = self._make_options()
        if not Path(opts.input_dir).exists():
            Path(opts.input_dir).mkdir(parents=True, exist_ok=True)
        files = iter_input_files(Path(opts.input_dir), opts.subdirs)
        if not files:
            messagebox.showinfo("未找到照片",
                                "输入目录中未找到可处理的图片：\n{}\n\n"
                                "可直接放入照片，也可放入整个照片文件夹。\n"
                                "支持的格式：{}".format(
                                    opts.input_dir,
                                    "、".join(sorted(e.lstrip('.') for e in IMAGE_EXTS))))
            return

        self.stop_flag.clear()
        self.btn_start.configure(state=DISABLED)
        self.btn_stop.configure(state=NORMAL)
        self.bar.configure(value=0, maximum=len(files))
        self.var_status.set("正在处理…… 0/{}".format(len(files)))
        self._save_settings()
        self.txt.delete("1.0", END)
        self.log("输入目录：{}".format(opts.input_dir))
        self.log("输出目录：{}".format(opts.output_dir))
        self.log("共 {} 个文件，开始处理……".format(len(files)))
        self.log("-" * 60)

        def work() -> None:
            try:
                reps = process_batch(
                    opts,
                    log=lambda m: self.queue.put(("log", m)),
                    progress=lambda i, n, r: self.queue.put(("prog", (i, n))),
                    should_stop=self.stop_flag.is_set,
                )
                self.queue.put(("done", reps))
            except Exception:
                self.queue.put(("error", traceback.format_exc()))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def on_stop(self) -> None:
        self.stop_flag.set()
        self.var_status.set("正在中止……（当前文件处理完成后停止）")

    def on_close(self) -> None:
        self.stop_flag.set()
        self._save_settings()
        self.root.destroy()

    # --------------------------------------------------------------- queue --
    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self.txt.insert(END, str(payload) + "\n")
                    self.txt.see(END)
                elif kind == "prog":
                    i, n = payload
                    self.bar.configure(value=i, maximum=n)
                    self.var_status.set("正在处理…… {}/{}".format(i, n))
                elif kind == "done":
                    self._finish(payload)
                elif kind == "error":
                    self.txt.insert(END, payload + "\n")
                    self.txt.see(END)
                    self.var_status.set("发生错误，详见日志")
                    self.btn_start.configure(state=NORMAL)
                    self.btn_stop.configure(state=DISABLED)
        except queue.Empty:
            pass
        self.root.after(120, self._drain_queue)

    def _finish(self, reports) -> None:
        self.btn_start.configure(state=NORMAL)
        self.btn_stop.configure(state=DISABLED)
        ok = [r for r in reports if r.ok]
        skipped = [r for r in reports if r.skipped]
        failed = [r for r in reports if r.error]
        low = [r for r in ok if r.method == "center"]
        self.var_status.set("处理结束：成功 {} 个扫描文件（{} 张照片），失败 {}".format(
            len(ok), len(ok) * 2, len(failed)))
        self.log("-" * 60)
        if low:
            self.log("有 {} 个文件未检测到明确间隙（已按中心线分割），"
                     "建议查看预览图确认。".format(len(low)))
        if failed:
            for r in failed:
                self.log("失败：{} -> {}".format(r.src.name, r.error))
        if ok:
            folders = []
            for r in ok:
                d = output_dir_for(r.src, self._make_options())
                if d not in folders:
                    folders.append(d)
            self.log("结果已保存至：")
            for d in folders:
                self.log("  " + str(d))
            msg = "处理完成：{} 个扫描文件 -> {} 张照片。".format(len(ok), len(ok) * 2)
            if len(folders) > 1:
                msg += "\n\n结果分布于 {} 个文件夹（与输入目录同名）：".format(len(folders))
                msg += "\n" + "\n".join("  " + str(d) for d in folders[:8])
                if len(folders) > 8:
                    msg += "\n  ……"
            msg += "\n\n是否立即打开输出目录？"
            if messagebox.askyesno("处理完成", msg):
                self._open_dir(self.var_out.get())
        else:
            messagebox.showwarning("处理结果为空", "未能成功处理任何文件，请查看日志。")

    # ------------------------------------------------------------ settings --
    def _load_settings(self) -> dict:
        try:
            if SETTINGS_FILE.exists():
                return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _save_settings(self) -> None:
        data = {
            "input": self.var_in.get(), "output": self.var_out.get(),
            "orient": ORIENT_LABELS.get(self.var_orient.get(), "auto"),
            "edge": EDGE_LABELS.get(self.var_edge.get(), "gap"),
            "trim": bool(self.var_trim.get()), "aspect_fix": bool(self.var_aspect.get()),
            "subdirs": bool(self.var_subdirs.get()),
            "overwrite": bool(self.var_overwrite.get()),
            "skip_uncertain": bool(self.var_skip.get()),
            "debug": bool(self.var_debug.get()), "quality": self.var_quality.get(),
        }
        try:
            SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        except Exception:
            pass


def main() -> int:
    root = Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
