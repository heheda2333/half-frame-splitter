#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""半格相机照片自动切割工具（Half-Frame Photo Splitter）

功能：将一张同时包含左右（或上下）两幅半格画面的扫描照片，自动定位两幅画面之间的
片基间隙，并分割为两张独立照片。

    input/   待处理的扫描照片
    output/  分割结果

命令行使用示例：
    python half_frame_splitter.py                  # 处理 input 目录，结果写入 output 目录
    python half_frame_splitter.py --trim           # 先去除扫描边框，再定位间隙
    python half_frame_splitter.py --debug          # 同时输出带切割线的预览图，便于核查
    python half_frame_splitter.py --cut 0.51       # 指定切割位置（0~1 为比例，大于 1 为像素）
    python half_frame_splitter.py --orient tb      # 强制按上下两幅处理
    python half_frame_splitter.py --dry-run        # 仅识别，不写入文件

算法要点（仅依赖 Pillow 与 numpy，不需要 OpenCV）：
    1. 灰度剖面与色度剖面结合的多尺度条带检测，用于定位狭窄且均匀的间隙；
       间隙既可暗于周围（正片扫描），也可亮于周围或偏橙（负片片基）。
    2. 以间隙方向的均匀度、半格画幅 18:24 的几何约束对候选位置加权评分。
    3. 以峰值显著度作为置信度；证据不足时退回图像中心线。
    4. 可选后处理：去除四周纯黑或纯白的扫描边框、按画幅比例修正构图、输出核查用预览图。

本工具仅读取输入目录、仅写入输出目录，不修改、不移动、不删除任何原始文件。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageOps

# 扫描文件通常体积较大，且偶有数据截断，此处放宽相关限制
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    Image.MAX_IMAGE_PIXELS = None
except Exception:  # pragma: no cover
    pass

# 保证在非 UTF-8 控制台下，个别字符不会导致输出中断
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "input"
DEFAULT_OUTPUT = BASE_DIR / "output"

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".jpe", ".png", ".tif", ".tiff", ".bmp",
    ".webp", ".ppm", ".pgm", ".gif",
}

# 半格画幅：18mm x 24mm -> 宽/高 = 0.75
HALF_FRAME_ASPECT = 18.0 / 24.0

# 分析用缩略图的最长边（数值越大结果越精细，处理速度相应降低）
ANALYSIS_MAX_DIM = 1400

# 间隙搜索范围（相对主边长度的比例）
DEFAULT_SEARCH = (0.22, 0.78)

# 间隙宽度搜索范围（相对主边长度的比例）
GAP_MIN_FRAC = 0.0015
GAP_MAX_FRAC = 0.050

# 峰值显著度低于此值时，视为未检测到明确间隙，退回图像中心线。
# 依据自检数据：真实间隙为 3.2~12.3；无间隙或无线索的干扰项为 1.3~1.4，取 2.0 可完全区分。
MIN_PROMINENCE = 2.00

# 间隙证据强度的绝对下限（归一化后的条带响应值），过低表示不存在间隙
MIN_PEAK_SCORE = 0.06

# 整图长宽比恰为 3:2 时（两幅等宽，切割位置即在中心），要求更强证据方允许偏离中心
TIGHT_MIN_PROMINENCE = 2.60

# 半格对片的主边长度至少应为垂直边的该倍数（36mm x 24mm，即 1.5）
MIN_PAIR_RATIO = 1.15

# 单个分块至少占主边的比例，避免切割结果过窄
MIN_PART_FRAC = 0.12

VERBOSE_PREFIX = "  "


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #

@dataclass
class Options:
    """一次批处理的全部参数（命令行界面与图形界面共用）。"""

    input_dir: Path = DEFAULT_INPUT
    output_dir: Path = DEFAULT_OUTPUT
    orient: str = "auto"                 # auto | lr | tb
    edge: str = "gap"                    # gap（整段舍去间隙） | center（自间隙正中切开）
    frame_aspect: float = HALF_FRAME_ASPECT
    search: tuple = DEFAULT_SEARCH
    trim: bool = False                   # 去除四周纯黑或纯白的扫描边框
    trim_tol: float = 0.10               # 判定为边框的亮度阈值
    trim_max: float = 0.12               # 每侧最多去除的比例
    aspect: str = "off"                  # off | auto | 数值（宽/高）
    out_format: str = "same"             # same | jpg | png | tif | webp
    quality: int = 95
    subdirs: bool = True                 # 识别输入目录的子文件夹，并在输出目录生成同名文件夹
    overwrite: bool = False
    skip_uncertain: bool = False         # 置信度不足时跳过处理
    min_confidence: float = MIN_PROMINENCE
    manual_cut: Optional[float] = None   # 手动指定的切割位置
    debug: bool = False                  # 输出带切割线的预览图
    dry_run: bool = False


@dataclass
class SplitResult:
    orient: str                  # lr = 左右两幅；tb = 上下两幅
    method: str                  # seam | center | manual
    confidence: float
    peak: int                    # 分析尺度下的间隙中心
    band: tuple                  # 分析尺度下的间隙范围（半开区间）
    cut_a: int                   # 第一幅的结束位置（分析尺度）
    cut_b: int                   # 第二幅的起始位置（分析尺度）
    main_len: int
    perp_len: int
    scale: float                 # 分析尺度 -> 原图尺度
    searched: tuple
    peak_score: float = 0.0      # 缝隙证据的绝对强度（调试用）


@dataclass
class FileReport:
    src: Path
    ok: bool = False
    skipped: bool = False
    orient: Optional[str] = None
    method: str = ""
    confidence: float = 0.0
    cut_a: Optional[int] = None
    cut_b: Optional[int] = None
    src_size: Optional[tuple] = None
    out_sizes: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    note: str = ""
    error: str = ""
    elapsed: float = 0.0


# --------------------------------------------------------------------------- #
# 图像基础工具
# --------------------------------------------------------------------------- #

def _is_high_depth(mode: str) -> bool:
    return mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F")


def high_depth_to_L(img: Image.Image) -> Image.Image:
    """将 16 位或浮点灰度图按百分位拉伸为 8 位灰度图，避免直接转换造成高光截断。"""
    arr = np.asarray(img).astype(np.float32)
    if arr.size == 0:
        return Image.new("L", (1, 1), 0)
    lo, hi = (float(v) for v in np.percentile(arr, [0.5, 99.5]))
    if hi - lo < 1e-6:
        lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        hi = lo + 1.0
    u8 = np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    if u8.ndim == 3:                      # 少见情形：高深度彩色图像
        return Image.fromarray(u8[..., :3], "RGB").convert("L")
    return Image.fromarray(u8, "L")


def _small_image(img: Image.Image, max_dim: int) -> tuple:
    """返回（缩略图, 缩放比例）。缩放比例 = 缩略图尺寸 / 原图尺寸。"""
    w, h = img.size
    if max(w, h) <= max_dim:
        return img, 1.0
    s = max_dim / float(max(w, h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    try:
        small = img.resize((nw, nh), Image.BILINEAR)
    except Exception:
        small = img.convert("RGB").resize((nw, nh), Image.BILINEAR)
    return small, nw / float(w)


def analysis_arrays(img: Image.Image, max_dim: int = ANALYSIS_MAX_DIM) -> tuple:
    """生成分析所需的灰度数组、彩色数组（float32，取值 0~1）及缩放比例。"""
    base = img
    if _is_high_depth(img.mode):
        base = high_depth_to_L(img)
    elif img.mode not in ("L", "LA", "La", "RGB", "RGBA"):
        base = img.convert("RGB")

    small, scale = _small_image(base, max_dim)

    gray = np.asarray(small.convert("L"), dtype=np.float32) / 255.0
    if small.mode in ("L", "LA", "La", "1"):
        rgb = np.repeat(gray[..., None], 3, axis=2)
    else:
        rgb = np.asarray(small.convert("RGB"), dtype=np.float32) / 255.0

    if gray.ndim != 2:
        gray = gray.reshape(gray.shape[0], -1)
    return gray, rgb, scale


def exif_for_save(img: Image.Image) -> Optional[bytes]:
    """提取 EXIF 数据并归一化方向标记（图像在处理前已校正方向，不可重复旋转）。"""
    try:
        ex = img.getexif()
        if not ex:
            return None
        ex[274] = 1  # Orientation = 正常方向
        return ex.tobytes()
    except Exception:
        return None


def save_image(img: Image.Image, path: Path, fmt: str, quality: int = 95) -> None:
    fmt = (fmt or "JPEG").upper()
    if fmt in ("JPG", "JPEG"):
        fmt = "JPEG"
    if fmt == "TIF":
        fmt = "TIFF"

    work = img
    if fmt == "JPEG" and work.mode not in ("L", "RGB", "CMYK"):
        work = work.convert("RGB")
    if fmt == "PNG" and work.mode in ("CMYK", "I;16", "I", "F"):
        work = work.convert("RGB")
    if fmt == "WEBP" and work.mode not in ("RGB", "RGBA", "L"):
        work = work.convert("RGB")

    kwargs: dict = {}
    if fmt == "JPEG":
        kwargs.update(quality=int(quality), optimize=True,
                      subsampling=0 if quality >= 90 else 2)
    elif fmt == "PNG":
        kwargs.update(optimize=True)
    elif fmt == "WEBP":
        kwargs.update(quality=int(quality), method=4)
    elif fmt == "TIFF":
        kwargs.update(compression="tiff_lzw")

    exif = exif_for_save(work)
    if exif and fmt in ("JPEG", "PNG", "TIFF", "WEBP"):
        kwargs["exif"] = exif
    icc = work.info.get("icc_profile")
    if icc and fmt in ("JPEG", "PNG", "TIFF", "WEBP"):
        kwargs["icc_profile"] = icc
    dpi = work.info.get("dpi")
    if dpi and fmt in ("JPEG", "PNG", "TIFF"):
        kwargs["dpi"] = dpi

    work.save(str(path), format=fmt, **kwargs)


def ext_to_format(suffix: str) -> str:
    s = suffix.lower()
    if s in (".jpg", ".jpeg", ".jpe"):
        return "JPEG"
    if s == ".png":
        return "PNG"
    if s in (".tif", ".tiff"):
        return "TIFF"
    if s == ".webp":
        return "WEBP"
    if s == ".bmp":
        return "BMP"
    return "PNG"


FORMAT_EXT = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif", "WEBP": ".webp", "BMP": ".bmp"}


# --------------------------------------------------------------------------- #
# 缝隙检测
# --------------------------------------------------------------------------- #

def _centered_box_mean(a: np.ndarray, k: int) -> np.ndarray:
    """以 k（奇数）为窗口长度的居中滑动平均，边界按 edge 方式填充。"""
    k = int(max(1, k))
    if k % 2 == 0:
        k += 1
    half = k // 2
    a = np.asarray(a, dtype=np.float64)
    if half == 0:
        return a.copy()
    ap = np.pad(a, half, mode="edge")
    c = np.concatenate(([0.0], np.cumsum(ap)))
    return (c[k:] - c[:-k]) / float(k)


def _bar_response(prof: np.ndarray, k: int) -> np.ndarray:
    """条带检测器：计算中心条带均值与左右两侧条带均值之差的绝对值。

    狭窄、均匀且与两侧明显不同的条带响应最强，此类条带即两幅半格画面之间的片基间隙。
    """
    n = prof.size
    inner = _centered_box_mean(prof, k)
    off = max(1, int(k))
    idx = np.arange(n)
    left = inner[np.clip(idx - off, 0, n - 1)]
    right = inner[np.clip(idx + off, 0, n - 1)]
    return np.abs(inner - 0.5 * (left + right))


def _profiles(gray: np.ndarray, rgb: np.ndarray, orient: str) -> tuple:
    if orient == "lr":
        prof = gray.mean(axis=0)
        spread = gray.std(axis=0)
        chroma = rgb[..., 0].mean(axis=0) - rgb[..., 2].mean(axis=0)
        main, perp = prof.size, gray.shape[0]
    else:
        prof = gray.mean(axis=1)
        spread = gray.std(axis=1)
        chroma = rgb[..., 0].mean(axis=1) - rgb[..., 2].mean(axis=1)
        main, perp = prof.size, gray.shape[1]
    return (prof.astype(np.float64), spread.astype(np.float64),
            chroma.astype(np.float64), int(main), int(perp))


def _candidate_widths(main: int) -> list:
    wmin = max(2, int(round(main * GAP_MIN_FRAC)))
    wmax = max(wmin + 1, int(round(main * GAP_MAX_FRAC)))
    widths = {1, 3, 5, 7, 9}
    for v in np.geomspace(wmin, wmax, 10):
        k = int(round(float(v)))
        widths.add(k if k % 2 else k + 1)
    return sorted(k for k in widths if 1 <= k <= max(9, wmax + 2))


def _refine_band(prof: np.ndarray, peak: int, k: int, max_mult: float = 2.5) -> tuple:
    """将候选条带向两侧扩展为实际的间隙范围（半开区间 [l, r)）。"""
    n = prof.size
    half = max(0, k // 2)
    off = max(1, k)

    def box_mean(center: int) -> float:
        a = max(0, center - half)
        b = min(n, center + half + 1)
        return float(prof[a:b].mean()) if b > a else float(prof[peak])

    inner = box_mean(peak)
    outer = 0.5 * (box_mean(max(0, peak - off)) + box_mean(min(n - 1, peak + off)))
    contrast = abs(inner - outer)
    tol = max(0.25 * contrast, 1e-4)

    lim = int(min(max(k * max_mult, 2), max(2, n * 0.06)))
    l = peak
    r = peak
    while l > 0 and (peak - l) < lim and abs(float(prof[l - 1]) - inner) <= tol:
        l -= 1
    while r < n - 1 and (r - peak) < lim and abs(float(prof[r + 1]) - inner) <= tol:
        r += 1
    if r <= l:
        r = l + 1
    return l, r


def scan_axis(gray: np.ndarray, rgb: np.ndarray, orient: str,
              frame_aspect: float = HALF_FRAME_ASPECT,
              search: tuple = DEFAULT_SEARCH,
              content: tuple = None) -> dict:
    """在指定方向上搜索间隙，返回候选信息（均为分析尺度）。"""
    prof, spread, chroma, main, perp = _profiles(gray, rgb, orient)
    if content is None:
        content = (0, 0, gray.shape[1], gray.shape[0])

    a0, a1 = (content[0], content[2]) if orient == "lr" else (content[1], content[3])
    cperp = (content[3] - content[1]) if orient == "lr" else (content[2] - content[0])
    cmain = max(4, a1 - a0)
    cperp = max(4, cperp)

    lo = int(round(a0 + cmain * max(0.0, min(search[0], search[1]))))
    hi = int(round(a0 + cmain * min(0.95, max(search[0], search[1]))))
    lo = max(1, lo)
    hi = min(main - 2, hi)
    if hi <= lo:
        lo, hi = max(1, main // 3), max(2, main - main // 3)

    center = (a0 + a1) / 2.0
    img_center = (main - 1) / 2.0
    x = np.arange(main, dtype=np.float64)
    prior = np.exp(-0.5 * ((x - center) / (0.24 * cmain)) ** 2)

    # 半格几何约束：单幅画面沿主边的长度应约等于 frame_aspect 与垂直边长之积
    r_exp = max(1.0, frame_aspect * cperp)
    total_ratio = cmain / (2.0 * r_exp)
    if 0.80 <= total_ratio <= 1.30:
        d1 = np.abs(x - (a0 + r_exp)) / r_exp
        d2 = np.abs((a1 - r_exp) - x) / r_exp
        f1 = 0.55 + 0.45 * np.clip(1.0 - d1 / 0.35, 0.0, 1.0)
        f2 = 0.55 + 0.45 * np.clip(1.0 - d2 / 0.35, 0.0, 1.0)
        geom = np.sqrt(f1 * f2)
    else:
        geom = np.ones(main)

    # 整图长宽比接近 3:2（1.5）时，可判定图幅即为并排的两幅半格画面，
    # 两幅必然等宽，切割位置唯一确定于图像中心；此时间隙证据仅用于确定舍去的区段。
    tight = 1.40 <= (main / float(perp)) <= 1.60

    p_scale = float(np.std(prof)) + 1e-6
    c_scale = float(np.std(chroma)) + 1e-6
    s_scale = float(np.median(spread)) + 1e-6

    # evidence：仅由图像自身证据构成（条带响应 × 均匀度），不含任何先验，
    #           用于衡量该位置与间隙的相似程度；
    # score   ：在 evidence 基础上乘以几何约束与中心先验，用于确定最终位置。
    evidence = np.zeros(main, dtype=np.float64)
    score_curve = np.zeros(main, dtype=np.float64)
    best = (-1.0, main // 2, 3)
    for k in _candidate_widths(main):
        rg = _bar_response(prof, k) / p_scale
        rc = _bar_response(chroma, k) / c_scale
        resp = np.maximum(rg, 0.85 * rc)
        unif = _centered_box_mean(spread, k) / s_scale
        # 实际间隙沿长度方向高度均匀，画面内的强边缘（如栏杆、门框）则不然，
        # 因此对均匀度不足的候选施加惩罚。
        unif_f = 0.55 + 0.45 * (1.0 - np.clip(unif, 0.0, 1.0))
        ev = resp * unif_f
        np.maximum(evidence, ev, out=evidence)
        score = ev * geom * prior
        np.maximum(score_curve, score, out=score_curve)
        seg = score[lo:hi + 1]
        m = float(seg.max())
        if m > best[0]:
            best = (m, lo + int(seg.argmax()), k)

    peak, k = best[1], best[2]

    # 置信度：所选位置的证据强度相对于搜索区间内整体证据水平的突出程度
    seg_ev = evidence[lo:hi + 1]
    med = float(np.median(seg_ev))
    p90 = float(np.percentile(seg_ev, 90))
    floor = 0.01 * max(1.0, float(seg_ev.max()))     # 避免平坦曲线使比值异常放大
    prominence = (float(evidence[peak]) - med) / max(p90 - med, floor)
    peak_score = float(evidence[peak])
    band = _refine_band(prof, peak, k)
    return {
        "orient": orient,
        "peak": int(peak),
        "k": int(k),
        "band": (int(band[0]), int(band[1])),
        "prominence": float(prominence),
        "peak_score": float(peak_score),
        "main": int(main),
        "perp": int(cperp),
        "content": tuple(int(v) for v in content),
        "a0": int(a0),
        "a1": int(a1),
        "cmain": int(cmain),
        "center": float(center),
        "img_center": float(img_center),
        "tight": bool(tight),
        "searched": (int(lo), int(hi)),
        "score": float(best[0]),
    }


def detect_split(img: Image.Image, opts: Options) -> SplitResult:
    """检测切割位置，返回分析尺度下的结果，由调用方换算回原图坐标。"""
    gray, rgb, scale = analysis_arrays(img)

    # 内容区：去除四周纯黑或纯白边框后的有效画面（仅用于几何推断，不改动图像）
    try:
        bx = detect_border_box(img, tol=0.10, max_frac=0.15)
    except Exception:
        bx = (0, 0, img.size[0], img.size[1])
    content = tuple(int(math.floor(v * scale)) for v in bx)

    orient = opts.orient
    if orient == "auto":
        res_lr = scan_axis(gray, rgb, "lr", opts.frame_aspect, opts.search, content)
        res_tb = scan_axis(gray, rgb, "tb", opts.frame_aspect, opts.search, content)
        p_lr, p_tb = res_lr["prominence"], res_tb["prominence"]
        # 几何可行性：两幅半格并排的总长度为 24mm x 1.5，故主边必然明显长于垂直边。
        # 左右排布要求 宽/高 >= 1.15，上下排布要求 高/宽 >= 1.15，二者互斥。
        feas_lr = res_lr["main"] >= res_lr["perp"] * MIN_PAIR_RATIO
        feas_tb = res_tb["main"] >= res_tb["perp"] * MIN_PAIR_RATIO
        if feas_lr != feas_tb:
            orient, res = ("lr", res_lr) if feas_lr else ("tb", res_tb)
        elif p_lr >= 1.20 * p_tb:
            orient, res = "lr", res_lr
        elif p_tb >= 1.20 * p_lr:
            orient, res = "tb", res_tb
        else:  # 近似正方形或两类证据相当，按长边方向处理
            long_h = (res_lr["main"] >= res_lr["perp"])
            orient, res = ("lr", res_lr) if long_h else ("tb", res_tb)
    else:
        res = scan_axis(gray, rgb, orient, opts.frame_aspect, opts.search, content)

    main = res["main"]
    band_l, band_r = res["band"]
    peak = res["peak"]
    a0, a1 = res["a0"], res["a1"]
    center = int(round(res["center"]))
    img_center = int(round(res["img_center"]))
    tight = res["tight"]

    # 判定为间隙的条件：相对突出程度与绝对证据强度同时达标
    strong = (res["prominence"] >= max(opts.min_confidence, 0.0)
              and res["peak_score"] >= MIN_PEAK_SCORE)
    # 整图长宽比恰为 3:2 时两幅必然等宽，切割位置只可能位于图像中心附近：
    # 其一，峰值与中心的距离受限；其二，偏离中心需要更强的证据支持。
    if tight:
        if abs(peak - img_center) > 0.05 * main:
            strong = False
        elif res["prominence"] < TIGHT_MIN_PROMINENCE:
            strong = False

    if opts.manual_cut is not None:
        pos = opts.manual_cut
        if 0.0 < pos <= 1.0:
            pos = a0 + pos * (a1 - a0)
        pos = int(round(min(max(pos, 1), main - 2)))
        cut_a = cut_b = pos
        band = (pos, pos + 1)
        method, conf = "manual", 9.99
    elif strong:
        method, conf = "seam", float(res["prominence"])
        if opts.edge == "gap" and band_r > band_l:
            inset = max(1, int(round((band_r - band_l) * 0.15)))
            cut_a = band_l + inset
            cut_b = band_r - inset
        else:
            cut_a = cut_b = peak
        band = (band_l, band_r)
    else:
        # 未检出间隙时的稳妥处理：按图像中心切割（两幅等宽时该位置即为分界）
        method, conf = "center", float(res["prominence"])
        cut_a = cut_b = img_center
        band = (cut_a, cut_a + 1)

    # 限制单侧最小尺寸，避免切割结果过窄
    lim_lo = max(1, int(round(main * MIN_PART_FRAC)))
    lim_hi = main - lim_lo
    cut_a = min(max(cut_a, lim_lo), lim_hi)
    cut_b = min(max(cut_b, lim_lo), lim_hi)
    if cut_b < cut_a:
        cut_a = cut_b = min(max((cut_a + cut_b) // 2, lim_lo), lim_hi)

    return SplitResult(
        orient=orient, method=method, confidence=conf, peak=int(peak),
        band=band, cut_a=int(cut_a), cut_b=int(cut_b),
        main_len=int(main), perp_len=int(res["perp"]), scale=float(scale),
        searched=res["searched"], peak_score=float(res.get("peak_score", 0.0)),
    )


# --------------------------------------------------------------------------- #
# 边框处理 / 画幅比例
# --------------------------------------------------------------------------- #

def detect_border_box(img: Image.Image, tol: float = 0.10, max_frac: float = 0.12,
                      max_std: float = 0.06, max_dim: int = 700) -> tuple:
    """检测四周纯黑或纯白的均匀扫描边框，返回原图坐标下的裁剪框。"""
    w, h = img.size
    base = high_depth_to_L(img) if _is_high_depth(img.mode) else img.convert("L")
    small, scale = _small_image(base, max_dim)
    g = np.asarray(small, dtype=np.float32) / 255.0
    H, W = g.shape
    if H < 8 or W < 8:
        return (0, 0, w, h)

    def line_ok(vals: np.ndarray) -> bool:
        m = float(vals.mean())
        return (m < tol or m > 1.0 - tol) and float(vals.std()) < max_std

    top_lim = int(min(H * max_frac, H // 3))
    left_lim = int(min(W * max_frac, W // 3))

    top = 0
    while top < top_lim and line_ok(g[top, :]):
        top += 1
    bottom = 0
    while bottom < top_lim and line_ok(g[H - 1 - bottom, :]):
        bottom += 1
    left = 0
    while left < left_lim and line_ok(g[:, left]):
        left += 1
    right = 0
    while right < left_lim and line_ok(g[:, W - 1 - right]):
        right += 1

    if top + bottom + left + right == 0:
        return (0, 0, w, h)

    inv = 1.0 / scale
    box = (int(round(left * inv)), int(round(top * inv)),
           w - int(round(right * inv)), h - int(round(bottom * inv)))
    if box[2] - box[0] < w * 0.5 or box[3] - box[1] < h * 0.5:
        return (0, 0, w, h)          # 裁剪幅度过大，放弃该结果
    return box


def crop_to_aspect(img: Image.Image, ratio: float) -> Image.Image:
    """按目标宽高比居中裁剪（仅裁剪，不补边）。"""
    if ratio <= 0:
        return img
    w, h = img.size
    cur = w / float(h)
    if abs(cur - ratio) / ratio < 0.01:
        return img
    if cur > ratio:
        nw = max(1, int(round(h * ratio)))
        x0 = (w - nw) // 2
        return img.crop((x0, 0, x0 + nw, h))
    nh = max(1, int(round(w / ratio)))
    y0 = (h - nh) // 2
    return img.crop((0, y0, w, y0 + nh))


# --------------------------------------------------------------------------- #
# 核查用预览图
# --------------------------------------------------------------------------- #

def write_debug_preview(img: Image.Image, res: SplitResult, path: Path,
                        parts: tuple = None, quality: int = 88) -> None:
    """在缩略图上标注检测到的间隙（绿框）与实际切割位置（红线），便于人工核查。"""
    max_dim = 1400
    w, h = img.size
    # 采用与识别阶段一致的缩放比例，标注线才能落在正确位置
    s = res.scale if 0 < res.scale <= 1 else min(1.0, max_dim / float(max(w, h)))
    prev = img.convert("RGB").resize((max(1, int(round(w * s))), max(1, int(round(h * s)))),
                                     Image.BILINEAR)
    d = ImageDraw.Draw(prev)
    pw, ph = prev.size

    def sc(v: int) -> float:
        return v * s

    color_band = (0, 200, 0)
    color_cut = (255, 0, 0)
    if res.orient == "lr":
        d.rectangle([sc(res.band[0]), 0, sc(res.band[1]), ph], outline=color_band, width=2)
        for c in (res.cut_a, res.cut_b):
            d.line([sc(c), 0, sc(c), ph], fill=color_cut, width=3)
    else:
        d.rectangle([0, sc(res.band[0]), pw, sc(res.band[1])], outline=color_band, width=2)
        for c in (res.cut_a, res.cut_b):
            d.line([0, sc(c), pw, sc(c)], fill=color_cut, width=3)

    label = "{} {} conf={:.2f} cut={}/{}".format(
        res.orient, res.method, res.confidence, res.cut_a, res.cut_b)
    d.rectangle([0, 0, min(pw, 8 + 7 * len(label)), 18], fill=(0, 0, 0))
    d.text((4, 4), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    prev.save(str(path), format="JPEG", quality=quality)


# --------------------------------------------------------------------------- #
# 单文件处理
# --------------------------------------------------------------------------- #

def process_file(src: Path, opts: Options) -> FileReport:
    t0 = time.perf_counter()
    rep = FileReport(src=src)
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im) or im
            im.load()
            work = im.copy()
        rep.src_size = work.size

        # 1）可选：先去除扫描边框
        if opts.trim:
            box = detect_border_box(work, tol=opts.trim_tol, max_frac=opts.trim_max)
            if box != (0, 0, work.size[0], work.size[1]):
                work = work.crop(box)

        # 2）检测间隙位置
        res = detect_split(work, opts)
        rep.orient = res.orient
        rep.method = res.method
        rep.confidence = res.confidence

        inv = 1.0 / res.scale if res.scale else 1.0
        main_px = work.size[0] if res.orient == "lr" else work.size[1]
        c1 = int(round(res.cut_a * inv))
        c2 = int(round(res.cut_b * inv))
        c1 = min(max(c1, 1), main_px - 2)
        c2 = min(max(c2, 1), main_px - 2)
        if c2 < c1:
            c1 = c2 = (c1 + c2) // 2
        rep.cut_a, rep.cut_b = c1, c2

        # 3）按检测位置分割
        if res.orient == "lr":
            parts = [work.crop((0, 0, c1, work.size[1])),
                     work.crop((c2, 0, work.size[0], work.size[1]))]
        else:
            parts = [work.crop((0, 0, work.size[0], c1)),
                     work.crop((0, c2, work.size[0], work.size[1]))]

        # 4）可选：对每个分块再次去边框并按画幅比例修正
        if opts.aspect and str(opts.aspect).lower() != "off":
            if str(opts.aspect).lower() == "auto":
                ratio = opts.frame_aspect if res.orient == "lr" else 1.0 / opts.frame_aspect
            else:
                ratio = float(opts.aspect)
            parts = [crop_to_aspect(p, ratio) for p in parts]
        if opts.trim:
            trimmed = []
            for p in parts:
                box = detect_border_box(p, tol=opts.trim_tol, max_frac=opts.trim_max)
                trimmed.append(p.crop(box) if box != (0, 0, p.size[0], p.size[1]) else p)
            parts = trimmed

        rep.out_sizes = [p.size for p in parts]

        # 5）整理结果说明，标注需要人工核查的情形
        notes = []
        if res.method == "center":
            notes.append("未检测到明确间隙（置信度 {:.2f}），已按中心线分割，"
                         "建议查看预览图确认".format(res.confidence))
        if res.method == "manual":
            notes.append("切割位置由手动指定")
        gap_px = max(0, c2 - c1)
        if gap_px > 0:
            notes.append("已舍去间隙 {} 像素".format(gap_px))
        long_side, short_side = max(work.size), min(work.size)
        ratio = long_side / float(short_side)
        if ratio < 1.22:
            notes.append("原图近似正方形，可能并非半格对片")
        elif res.orient == "tb" and res.method == "center" and 1.24 <= ratio <= 1.45:
            # 仅在未能确认间隙时才提示：整体比例接近 3:4 的文件，也可能是单幅画面
            notes.append("原图比例接近 3:4，若该文件本身即为单幅画面，"
                         "建议将其移出输入目录")
        rep.note = "；".join(notes)

        if opts.skip_uncertain and res.method != "manual" and \
                res.confidence < opts.min_confidence:
            rep.skipped = True
            rep.ok = False
            rep.elapsed = time.perf_counter() - t0
            return rep

        # 6）写出结果文件
        fmt = opts.out_format
        if fmt == "same":
            fmt = ext_to_format(src.suffix)
            out_ext = src.suffix if src.suffix.lower() in (".jpg", ".jpeg", ".png",
                                                           ".tif", ".tiff", ".webp") else FORMAT_EXT[fmt]
        else:
            fmt = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "tif": "TIFF",
                   "tiff": "TIFF", "webp": "WEBP"}.get(fmt.lower(), "JPEG")
            out_ext = FORMAT_EXT.get(fmt, ".jpg")

        if opts.debug:
            write_debug_preview(work, res, debug_dir_for(src, opts) /
                                (src.stem + "_split.jpg"))

        if opts.dry_run:
            rep.ok = True
            rep.outputs = []
            rep.elapsed = time.perf_counter() - t0
            return rep

        out_dir = output_dir_for(src, opts)
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for i, part in enumerate(parts, start=1):
            dst = out_dir / "{}_{}{}".format(src.stem, i, out_ext)
            if dst.exists() and not opts.overwrite:
                rep.note = (rep.note + "；" if rep.note else "") + \
                    "{}_{} 已存在，跳过写入".format(src.stem, i)
                continue
            save_image(part, dst, fmt, opts.quality)
            outputs.append(dst)
        rep.outputs = outputs
        rep.ok = True
    except Exception as exc:  # noqa: BLE001
        rep.ok = False
        rep.error = "{}: {}".format(type(exc).__name__, exc)
    rep.elapsed = time.perf_counter() - t0
    return rep


def relative_subdir(src: Path, opts: Options) -> Path:
    """返回 src 相对输入根目录的子目录；位于根目录时返回空路径。

    例如：input/A/B/1.jpg 返回 A/B；input/2.jpg 返回空。
    输出目录据此生成同名文件夹 A\\B，并将结果写入其中。
    """
    root = Path(opts.input_dir)
    if not opts.subdirs or root.is_file():
        return Path()
    try:
        rel = src.parent.resolve().relative_to(root.resolve())
    except Exception:
        return Path()
    return Path() if str(rel) in (".", "") else rel


def output_dir_for(src: Path, opts: Options) -> Path:
    """返回该文件的切割结果应写入的目录（自动镜像输入目录的子文件夹结构）。"""
    base = Path(opts.output_dir if opts.output_dir else DEFAULT_OUTPUT)
    sub = relative_subdir(src, opts)
    return base / sub if str(sub) not in (".", "") else base


def debug_dir_for(src: Path, opts: Options) -> Path:
    return output_dir_for(src, opts) / "_debug"


def display_name(src: Path, opts: Options) -> str:
    """返回日志中显示的相对路径，例如 A\\IMG_0001.jpg。"""
    sub = relative_subdir(src, opts)
    return str(sub / src.name) if str(sub) not in (".", "") else src.name


def iter_input_files(root: Path, subdirs: bool = True) -> list:
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTS else []
    if not root.exists():
        return []
    it = root.rglob("*") if subdirs else root.glob("*")
    files = [p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS
             and not p.name.startswith(".")
             and not any(part.startswith(".") for part in p.relative_to(root).parts)]
    return sorted(files, key=lambda p: str(p).lower())


# --------------------------------------------------------------------------- #
# 批处理
# --------------------------------------------------------------------------- #

def process_batch(opts: Options,
                  log: Optional[Callable[[str], None]] = None,
                  progress: Optional[Callable[[int, int, FileReport], None]] = None,
                  should_stop: Optional[Callable[[], bool]] = None) -> list:
    def emit(msg: str) -> None:
        if log:
            log(msg)
        else:
            print(msg)

    in_dir = Path(opts.input_dir)
    out_dir = Path(opts.output_dir)
    in_dir.mkdir(parents=True, exist_ok=True)
    if not opts.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_input_files(in_dir, opts.subdirs)
    if not files:
        emit("输入目录中未找到可处理的图片：{}".format(in_dir))
        emit("请将待处理的照片或照片文件夹放入该目录，支持的格式：{}".format(
            "、".join(sorted(e.lstrip('.') for e in IMAGE_EXTS))))
        return []

    # 统计各子文件夹中的文件数量（排序规则：根目录在前，其余按名称）
    folder_counts: dict = {}
    for f in files:
        sub = relative_subdir(f, opts)
        key = str(sub) if str(sub) not in (".", "") else "."
        folder_counts[key] = folder_counts.get(key, 0) + 1
    ordered = sorted(folder_counts.items(), key=lambda kv: (kv[0] != ".", kv[0].lower()))

    emit("共找到 {} 个待处理文件{}".format(
        len(files), "，分布于 {} 个文件夹".format(len(ordered)) if len(ordered) > 1 else ""))
    for key, n in ordered:
        if key == ".":
            emit("  {}（输入目录根）: {} 个 -> 输出至 {}".format(in_dir, n, out_dir))
        else:
            emit("  {}: {} 个 -> 输出至 {}".format(key, n, out_dir / key))
    emit("-" * 68)

    reports: list = []
    total = len(files)
    for idx, src in enumerate(files, start=1):
        if should_stop and should_stop():
            emit("已中止，其余 {} 个文件未处理".format(total - idx + 1))
            break
        rep = process_file(src, opts)
        reports.append(rep)
        name = display_name(src, opts)
        o = "左右" if rep.orient == "lr" else ("上下" if rep.orient == "tb" else "-")
        if rep.error:
            emit("[{}/{}] {}  处理失败：{}".format(idx, total, name, rep.error))
        elif rep.skipped:
            emit("[{}/{}] {}  已跳过（置信度 {:.2f} 低于阈值 {:.2f}）".format(
                idx, total, name, rep.confidence, opts.min_confidence))
        else:
            size = "{}x{}".format(*rep.src_size) if rep.src_size else "-"
            outs = "、".join(p.name for p in rep.outputs) if rep.outputs else "（未写入文件）"
            emit("[{}/{}] {}  [{}] {} 切割位置 {}|{}  置信度 {:.2f}  用时 {:.2f}s".format(
                idx, total, name, size, o, rep.cut_a, rep.cut_b,
                rep.confidence, rep.elapsed))
            emit(VERBOSE_PREFIX + "输出：" + outs)
        if rep.note:
            emit(VERBOSE_PREFIX + "说明：" + rep.note)
        if progress:
            progress(idx, total, rep)

    # 汇总
    ok = [r for r in reports if r.ok]
    skipped = [r for r in reports if r.skipped]
    failed = [r for r in reports if r.error]
    low = [r for r in ok if r.method == "center"]
    emit("-" * 68)
    emit("处理结束：成功 {}，跳过 {}，失败 {}".format(len(ok), len(skipped), len(failed)))
    if opts.dry_run:
        emit("（--dry-run：仅执行识别，未写入任何文件）")
    if ok and not opts.dry_run:
        out_folders = []
        for r in ok:
            d = output_dir_for(r.src, opts)
            if d not in out_folders:
                out_folders.append(d)
        emit("输出目录：")
        for d in out_folders:
            n = sum(1 for r in ok if output_dir_for(r.src, opts) == d)
            emit("  {}   （{} 个扫描文件 -> {} 张照片）".format(d, n, n * 2))
    if low:
        emit("以下文件未检测到明确间隙（已按中心线分割），"
             "建议查看其所在目录 _debug 中的预览图：")
        for r in low:
            emit(VERBOSE_PREFIX + display_name(r.src, opts))
    if failed:
        for r in failed:
            emit("失败：" + display_name(r.src, opts) + " -> " + r.error)
    return reports


def make_logo() -> str:
    return (
        "=" * 68 + "\n"
        "  半格相机照片自动切割工具  Half-Frame Photo Splitter\n"
        "  单个扫描文件 -> 两张独立照片\n"
        "  本工具仅读取输入、仅写入输出，不修改、不删除任何原始文件\n"
        + "=" * 68
    )


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="half_frame_splitter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="将半格相机扫描文件（单幅含左右或上下两幅画面）自动分割为两张照片。",
        epilog="""示例：
  python half_frame_splitter.py                     处理 input 目录，结果写入 output 目录
  python half_frame_splitter.py --trim --debug      去除扫描边框，并输出核查用预览图
  python half_frame_splitter.py -i <输入目录> -o <输出目录>
  python half_frame_splitter.py --no-subdirs        仅处理输入目录根层，忽略子文件夹
  python half_frame_splitter.py --orient tb         强制按上下两幅处理
  python half_frame_splitter.py --cut 0.5           按图像中心手动指定切割位置

文件夹规则：
  照片直接置于输入目录                结果写入输出目录
  整个文件夹 A 置于输入目录           结果写入输出目录\\A\\
  多层结构（如 A/B）                  结果写入输出目录\\A\\B\\，目录结构完整镜像
""")
    p.add_argument("-i", "--input", default=str(DEFAULT_INPUT),
                   help="输入目录（默认 ./input）")
    p.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT),
                   help="输出目录（默认 ./output）")
    p.add_argument("--orient", choices=["auto", "lr", "tb"], default="auto",
                   help="画面排布：auto 自动判断（默认）/ lr 左右两幅 / tb 上下两幅")
    p.add_argument("--edge", choices=["gap", "center"], default="gap",
                   help="间隙处理：gap 整段舍去间隙（默认）/ center 自间隙正中切开")
    p.add_argument("--frame-aspect", type=float, default=HALF_FRAME_ASPECT,
                   help="单幅画面宽高比，半格画幅为 0.75（即 18:24）")
    p.add_argument("--search", type=float, nargs=2, metavar=("LO", "HI"),
                   default=list(DEFAULT_SEARCH),
                   help="间隙搜索范围（比例，默认 0.22 0.78）")
    p.add_argument("--trim", action="store_true", help="去除四周纯黑或纯白的扫描边框")
    p.add_argument("--trim-tol", type=float, default=0.10,
                   help="判定为边框的亮度阈值（默认 0.10）")
    p.add_argument("--trim-max", type=float, default=0.12,
                   help="每侧最多去除的比例（默认 0.12）")
    p.add_argument("--aspect", default="off",
                   help="按画幅比例居中裁剪：off / auto / 数值（如 0.75）")
    p.add_argument("--format", dest="out_format", default="same",
                   choices=["same", "jpg", "png", "tif", "webp"],
                   help="输出格式（默认与输入一致）")
    p.add_argument("--quality", type=int, default=95, help="JPEG/WEBP 质量（默认 95）")
    p.add_argument("--no-subdirs", dest="subdirs", action="store_false",
                   help="仅处理输入目录根层，忽略子文件夹")
    p.add_argument("--subdirs", dest="subdirs", action="store_true",
                   help="识别子文件夹并在输出目录生成同名文件夹（默认行为）")
    p.add_argument("--recursive", dest="subdirs", action="store_true",
                   help=argparse.SUPPRESS)   # 兼容旧参数：该行为现为默认
    p.add_argument("--keep-structure", dest="subdirs", action="store_true",
                   help=argparse.SUPPRESS)   # 兼容旧参数：该行为现为默认
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")
    p.add_argument("--skip-uncertain", action="store_true",
                   help="置信度不足时跳过该文件")
    p.add_argument("--min-confidence", type=float, default=MIN_PROMINENCE,
                   help="置信度阈值（默认 {:.2f}）".format(MIN_PROMINENCE))
    p.add_argument("--cut", type=float, default=None,
                   help="手动指定切割位置：0~1 为比例，大于 1 为像素")
    p.add_argument("--debug", action="store_true",
                   help="输出带切割线的预览图（保存于结果目录的 _debug 中）")
    p.add_argument("--dry-run", action="store_true", help="仅执行识别，不写入文件")
    p.add_argument("--quiet", action="store_true", help="不输出启动信息")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    opts = Options(
        input_dir=Path(args.input).expanduser(),
        output_dir=Path(args.output).expanduser(),
        orient=args.orient,
        edge=args.edge,
        frame_aspect=args.frame_aspect,
        search=(float(args.search[0]), float(args.search[1])),
        trim=args.trim,
        trim_tol=args.trim_tol,
        trim_max=args.trim_max,
        aspect=args.aspect,
        out_format=args.out_format,
        quality=args.quality,
        subdirs=args.subdirs,
        overwrite=args.overwrite,
        skip_uncertain=args.skip_uncertain,
        min_confidence=args.min_confidence,
        manual_cut=args.cut,
        debug=args.debug,
        dry_run=args.dry_run,
    )

    if not args.quiet:
        print(make_logo())
    reports = process_batch(opts)
    if not reports:
        return 2
    failed = [r for r in reports if r.error]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
