#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自检脚本：合成各类半格扫描样本，检验切割位置的准确性。

用法：
    python selftest.py                        # 使用程序生成的图像作为测试素材
    python selftest.py --photos a.jpg b.jpg   # 使用指定的真实照片作为测试素材

运行结果写入 _selftest 目录（含样本、预览图与切割结果），并在控制台输出误差对照表。
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from half_frame_splitter import (  # noqa: E402
    BASE_DIR, MIN_PROMINENCE, Options, detect_border_box, detect_split,
    process_batch, process_file, write_debug_preview,
)

SELFTEST_DIR = BASE_DIR / "_selftest"


# --------------------------------------------------------------------------- #
# 素材
# --------------------------------------------------------------------------- #

def random_photo(seed: int, size: tuple, colorful: bool = True) -> Image.Image:
    """生成具有纹理结构的模拟照片（噪声、模糊、渐变与色块），用作画幅内容。"""
    w, h = size
    rng = np.random.default_rng(seed)
    base = rng.random((h // 8, w // 8, 3), dtype=np.float32)
    base = np.array(Image.fromarray((base * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC))
    img = Image.fromarray(base).filter(ImageFilter.GaussianBlur(3))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    yy, xx = np.mgrid[0:h, 0:w]
    grad = (xx / max(1, w - 1)) * rng.uniform(-0.4, 0.4) + \
           (yy / max(1, h - 1)) * rng.uniform(-0.4, 0.4)
    arr = np.clip(arr + grad[..., None], 0, 1)
    if not colorful:
        arr = np.repeat(arr.mean(axis=2, keepdims=True), 3, axis=2)

    img = Image.fromarray((arr * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(rng.integers(3, 9)):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        x1, y1 = x0 + rng.integers(w // 12, w // 3), y0 + rng.integers(h // 12, h // 3)
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        d.ellipse([x0, y0, x1, y1], fill=col)
    return img


def flat_photo(seed: int, size: tuple, level: float) -> Image.Image:
    """生成近似纯色的画幅，用于模拟天空、雪地等低纹理内容。"""
    rng = np.random.default_rng(seed)
    w, h = size
    arr = np.full((h, w, 3), level, dtype=np.float32)
    arr += rng.normal(0, 0.012, size=arr.shape).astype(np.float32)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# 合成半格扫描样本
# --------------------------------------------------------------------------- #

@dataclass
class Case:
    name: str
    desc: str
    expect: str          # seam | center
    image: Image.Image
    band: tuple          # 实际间隙区间（主边坐标，半开区间）；gap=0 时为两幅的分界位置
    main_axis: str       # lr | tb
    expect_orient: str
    trim: bool = False   # 该用例需启用 --trim


def build_pair(a: Image.Image, b: Image.Image, main_axis: str = "lr",
               frame_long: int = 1300, gap: int = 24, gap_color=(0, 0, 0),
               border: int = 0, border_color=(0, 0, 0), crop_start: int = 0,
               jpeg_quality: Optional[int] = None, blur: float = 0.0) -> tuple:
    """将两幅画面合成为一张模拟扫描件，返回（图像, 间隙区间, 主边方向）。"""
    f = int(round(frame_long * 0.75))          # 画幅沿主边方向的长度（18mm）
    perp = frame_long                          # 画幅垂直方向长度（24mm）

    if main_axis == "lr":
        fa = a.resize((f, perp), Image.LANCZOS)
        fb = b.resize((f, perp), Image.LANCZOS)
    else:                                      # 上下排布：等同于整卷扫描时旋转 90 度
        fa = a.resize((f, perp), Image.LANCZOS).rotate(90, expand=True)   # -> perp x f
        fb = b.resize((f, perp), Image.LANCZOS).rotate(90, expand=True)

    if main_axis == "lr":
        W, H = 2 * f + gap, perp
        canvas = Image.new("RGB", (W, H), tuple(int(c) for c in gap_color))
        canvas.paste(fa, (0, 0))
        canvas.paste(fb, (f + gap, 0))
    else:
        W, H = perp, 2 * f + gap
        canvas = Image.new("RGB", (W, H), tuple(int(c) for c in gap_color))
        canvas.paste(fa, (0, 0))
        canvas.paste(fb, (0, f + gap))

    band = (f, f + gap)

    if border > 0:
        bw, bh = canvas.size
        big = Image.new("RGB", (bw + 2 * border, bh + 2 * border),
                        tuple(int(c) for c in border_color))
        big.paste(canvas, (border, border))
        canvas = big
        band = (band[0] + border, band[1] + border)

    if crop_start > 0:
        if main_axis == "lr":
            canvas = canvas.crop((crop_start, 0, canvas.size[0], canvas.size[1]))
        else:
            canvas = canvas.crop((0, crop_start, canvas.size[0], canvas.size[1]))
        band = (band[0] - crop_start, band[1] - crop_start)

    if blur > 0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(blur))

    if jpeg_quality:
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=jpeg_quality)
        buf.seek(0)
        canvas = Image.open(buf).convert("RGB")

    return canvas, band, main_axis


def build_cases(photos: list) -> list:
    """构造测试用例；photos 为空时使用程序生成的图像。"""
    def photo(i: int, size=(900, 1200), colorful=True):
        if photos:
            p = photos[i % len(photos)]
            with Image.open(p) as im:
                im = im.convert("RGB")
                # 先按目标尺寸缩放，确保相邻两幅内容不同
                tw, th = size
                im = im.resize((tw, th), Image.LANCZOS)
                return im.copy()
        return random_photo(1000 + i, size, colorful)

    a, b = photo(0), photo(1)
    c, d = photo(2), photo(3)
    e, g = photo(4), photo(5)

    cases = []
    img, band, axis = build_pair(a, b, "lr", gap=24)
    cases.append(Case("lr_gap_dark", "左右排布 + 黑色间隙 24px", "seam", img, band, axis, "lr"))

    img, band, axis = build_pair(c, d, "lr", frame_long=1300, gap=64)
    cases.append(Case("lr_gap_wide", "左右排布 + 宽间隙 64px", "seam", img, band, axis, "lr"))

    img, band, axis = build_pair(e, g, "lr", gap=18, gap_color=(255, 176, 96))
    cases.append(Case("lr_gap_orange", "左右排布 + 橙色间隙（负片片基）", "seam", img, band, axis, "lr"))

    img, band, axis = build_pair(a, b, "lr", gap=30, gap_color=(250, 250, 250))
    cases.append(Case("lr_gap_bright", "左右排布 + 白色间隙（正片扫描）", "seam", img, band, axis, "lr"))

    img, band, axis = build_pair(c, d, "lr", gap=7)
    cases.append(Case("lr_gap_tiny", "左右排布 + 极细间隙 7px", "seam", img, band, axis, "lr"))

    img, band, axis = build_pair(a, b, "lr", gap=0)
    cases.append(Case("lr_no_gap", "左右排布 + 无间隙（已裁至画幅边缘）", "center", img, band, axis, "lr"))

    img, band, axis = build_pair(c, d, "lr", gap=20, border=70)
    cases.append(Case("lr_border", "左右排布 + 70px 黑色扫描边框", "seam", img, band, axis, "lr"))

    img, band, axis = build_pair(a, b, "lr", gap=24, crop_start=90)
    cases.append(Case("lr_asym_crop", "左右排布 + 左侧裁去 90px（不对称）", "seam", img, band, axis, "lr"))

    img, band, axis = build_pair(c, d, "lr", gap=22, jpeg_quality=55, blur=0.6)
    cases.append(Case("lr_jpeg_heavy", "左右排布 + 重度 JPEG 压缩 + 轻度模糊", "seam", img, band, axis, "lr"))

    img, band, axis = build_pair(e, g, "tb", gap=24)
    cases.append(Case("tb_gap_dark", "上下排布 + 黑色间隙", "seam", img, band, axis, "tb"))

    img, band, axis = build_pair(a, b, "tb", gap=20, border=60)
    cases.append(Case("tb_border", "上下排布 + 60px 黑色边框", "seam", img, band, axis, "tb"))

    fa = flat_photo(7, (900, 1200), 0.72)
    fb = flat_photo(8, (900, 1200), 0.35)
    img, band, axis = build_pair(fa, fb, "lr", gap=22)
    cases.append(Case("lr_flat_content", "左右排布 + 低纹理内容（天空、雪地）", "seam", img, band, axis, "lr"))

    # 干扰用例：画幅内部含贯穿全高的强竖线（如栏杆、门框），用于检验是否会被误判为间隙
    img, band, axis = build_pair(a, b, "lr", gap=24)
    dr = ImageDraw.Draw(img)
    lx = int(img.size[0] * 0.30)
    dr.rectangle([lx, 0, lx + 11, img.size[1]], fill=(0, 0, 0))
    cases.append(Case("lr_false_line", "左右排布 + 画面内 12px 强竖线干扰", "seam", img, band, axis, "lr"))

    # 重度不对称裁切：左幅被裁去大半，间隙位置明显偏左
    img, band, axis = build_pair(c, d, "lr", gap=24, crop_start=400)
    cases.append(Case("lr_crop_heavy", "左右排布 + 左侧裁去 400px（间隙偏左）", "seam", img, band, axis, "lr"))

    # 无间隙且四周含黑边：中心线仍应落在两幅的分界处
    img, band, axis = build_pair(a, b, "lr", gap=0, border=60)
    cases.append(Case("lr_no_gap_border", "左右排布 + 无间隙 + 60px 黑边", "center", img, band, axis, "lr"))

    # 低对比度间隙：间隙仅略暗于画面
    img, band, axis = build_pair(flat_photo(11, (900, 1200), 0.62),
                                 flat_photo(12, (900, 1200), 0.55), "lr", gap=26,
                                 gap_color=(120, 120, 120))
    cases.append(Case("lr_gap_weak", "左右排布 + 低对比度灰色间隙", "seam", img, band, axis, "lr"))

    # 含片基齿孔的整卷扫描：需先执行 --trim 去除黑边
    img, band, axis = build_pair(e, g, "lr", gap=22, border=90)
    dr = ImageDraw.Draw(img)
    for x in range(120, img.size[0] - 120, 160):
        dr.rectangle([x, 18, x + 90, 72], fill=(245, 245, 245))
    cases.append(Case("trim_sprocket", "含齿孔黑边的整卷扫描（需 --trim）", "seam", img, band, axis, "lr", trim=True))

    # 压力用例：同时施加细间隙、压缩、模糊与不对称裁切
    img, band, axis = build_pair(random_photo(21, (900, 1200)), random_photo(22, (900, 1200)),
                                 "lr", gap=14, crop_start=70, jpeg_quality=40, blur=1.0)
    cases.append(Case("lr_gap_hard", "细间隙 14px + JPEG 40 + 模糊 + 偏裁", "seam", img, band, axis, "lr"))

    # 浅灰色间隙，两侧画面为中等亮度
    img, band, axis = build_pair(random_photo(23, (900, 1200)), random_photo(24, (900, 1200)),
                                 "lr", gap=30, gap_color=(205, 205, 205), jpeg_quality=70)
    cases.append(Case("lr_gap_light", "左右排布 + 浅灰色间隙", "seam", img, band, axis, "lr"))

    return cases


# --------------------------------------------------------------------------- #
# 执行测试
# --------------------------------------------------------------------------- #

def run_case(case: Case, opts: Options, out_dir: Path, idx: int) -> dict:
    src = out_dir / "{:02d}_{}.jpg".format(idx, case.name)
    case.image.save(src, format="JPEG", quality=94)

    with Image.open(src) as im:
        work = im.convert("RGB")
    band = case.band
    if case.trim:                      # 与 process_file 一致：先去除边框，基准坐标同步平移
        box = detect_border_box(work, tol=opts.trim_tol, max_frac=opts.trim_max)
        work = work.crop(box) if box != (0, 0, work.size[0], work.size[1]) else work
        off = box[0] if case.main_axis == "lr" else box[1]
        band = (band[0] - off, band[1] - off)

    t0 = time.perf_counter()
    res = detect_split(work, opts)
    dt = time.perf_counter() - t0

    main_len = res.main_len / res.scale
    c1 = res.cut_a / res.scale
    c2 = res.cut_b / res.scale
    true_l, true_r = band
    tol_center = max(4.0, 0.006 * main_len)
    tol_eat = max(2.0, 0.003 * main_len)

    if true_r - true_l == 0:                      # 无间隙：仅要求切割位置接近实际分界
        center_err = abs(((c1 + c2) / 2.0) - true_l)
        ok = center_err <= max(6.0, 0.012 * main_len)
        ate = center_err
    else:
        center_err = abs(((c1 + c2) / 2.0) - (true_l + true_r) / 2.0)
        # 不得切入画面内容：左幅右边界不应越过间隙左沿，右幅左边界不应越过间隙右沿
        ate = max(true_l - c1, c2 - true_r)
        ok = (center_err <= tol_center) and (ate <= tol_eat)

    res_out = {
        "name": case.name, "desc": case.desc, "ok": bool(ok),
        "orient": res.orient, "expect_orient": case.expect_orient,
        "method": res.method, "expect": case.expect,
        "conf": res.confidence, "peak_score": res.peak_score,
        "center_err": center_err, "ate": ate,
        "main_len": main_len, "elapsed": dt, "src": src,
    }
    write_debug_preview(work, res, out_dir / (case.name + "_cut.jpg"))
    return res_out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="半格照片切割工具自检")
    ap.add_argument("--photos", nargs="*", default=[], help="指定用作素材的真实照片（可选）")
    ap.add_argument("--keep", action="store_true", help="保留 _selftest 目录（默认清空后重建）")
    args = ap.parse_args(argv)

    if SELFTEST_DIR.exists() and not args.keep:
        shutil.rmtree(SELFTEST_DIR, ignore_errors=True)
    SELFTEST_DIR.mkdir(parents=True, exist_ok=True)
    shots = SELFTEST_DIR / "scans"
    shots.mkdir(parents=True, exist_ok=True)

    photos = [Path(p) for p in args.photos if Path(p).exists()]
    cases = build_cases(photos)
    opts = Options()

    print("=" * 96)
    print("半格照片切割工具自检    素材来源：{}".format(
        "真实照片 {} 张".format(len(photos)) if photos else "程序生成图像"))
    print("=" * 96)
    print("{:<16}{:<34}{:>4}{:>6}{:>7}{:>7}{:>9}{:>8}{:>7}".format(
        "用例", "说明", "方向", "方法", "置信度", "证据强度", "中心误差", "侵入量", "结果"))
    print("-" * 96)

    results = []
    for i, case in enumerate(cases, start=1):
        c_opts = replace(opts, trim=True) if case.trim else opts
        r = run_case(case, c_opts, shots, i)
        results.append(r)
        flag = "通过" if r["ok"] else "未通过"
        orient_flag = "" if r["orient"] == r["expect_orient"] else "(方向误判)"
        print("{:<16}{:<34}{:>4}{:>6}{:>7.2f}{:>7.2f}{:>8.1f}px{:>7.1f}px{:>8}".format(
            r["name"], r["desc"][:32], r["orient"] + orient_flag, r["method"],
            r["conf"], r["peak_score"], r["center_err"], r["ate"], flag))

    # 端到端：完整执行一次文件处理流程
    e2e_in = SELFTEST_DIR / "e2e_in"
    e2e_out = SELFTEST_DIR / "e2e_out"
    e2e_in.mkdir(parents=True, exist_ok=True)
    if e2e_out.exists():
        shutil.rmtree(e2e_out, ignore_errors=True)
    case = cases[0]
    case.image.save(e2e_in / "sample_half.jpg", format="JPEG", quality=94)
    eopts = Options(input_dir=e2e_in, output_dir=e2e_out)
    rep = process_file(e2e_in / "sample_half.jpg", eopts)
    e2e_ok = rep.ok and len(rep.outputs) == 2 and all(p.exists() for p in rep.outputs)
    sizes = []
    if e2e_ok:
        for p in rep.outputs:
            with Image.open(p) as im:
                sizes.append(im.size)
    print("-" * 96)
    print("端到端写入测试：{}    输出 {} 个文件，尺寸 {}".format(
        "通过" if e2e_ok else "未通过", len(rep.outputs), sizes or rep.error))

    # 单幅画面（已切割完成的照片被误放入输入目录）：应正常处理并给出提示
    single = cases[0].image.crop((0, 0, 975, 1300))
    single_path = e2e_in / "single_half.jpg"
    single.save(single_path, format="JPEG", quality=94)
    srep = process_file(single_path, Options(input_dir=e2e_in, output_dir=SELFTEST_DIR / "e2e_single"))
    single_ok = srep.ok and "单幅" in srep.note
    print("单幅画面提示测试：{}    （{} / {}）".format(
        "通过" if single_ok else "未通过", srep.orient, srep.note or "-"))

    # 目录结构：输入目录中的子文件夹应在输出目录生成同名文件夹
    tree_ok, tree_msg = check_folder_tree(cases, opts)
    print("文件夹镜像测试：{}    {}".format("通过" if tree_ok else "未通过", tree_msg))

    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    confs = [r["conf"] for r in results]
    print("-" * 96)
    print("通过 {}/{}    置信度范围 {:.2f} ~ {:.2f}    （当前阈值 {:.2f}）".format(
        passed, total, min(confs), max(confs), MIN_PROMINENCE))
    print("样本、预览图与切割结果位于：{}".format(SELFTEST_DIR.name))
    return 0 if (passed == total and e2e_ok and single_ok and tree_ok) else 1


def check_folder_tree(cases: list, opts: Options) -> tuple:
    """检验输入目录中的子文件夹是否在输出目录生成同名文件夹（含多层嵌套）。"""
    tin = SELFTEST_DIR / "tree_in"
    tout = SELFTEST_DIR / "tree_out"
    for d in (tin, tout):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    # 目录布局：根目录 1 个、A/ 2 个、A/B/ 1 个、B/ 1 个，另含非图片文件
    plan = {
        "top.jpg": cases[0],
        "A/a1.jpg": cases[0],
        "A/a2.jpg": cases[1],
        "A/B/deep.jpg": cases[2],
        "B/b1.jpg": cases[3],
    }
    for rel, case in plan.items():
        p = tin / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        case.image.save(p, format="JPEG", quality=92)
    (tin / "A" / "readme.txt").write_text("not an image", encoding="utf-8")
    # 已切割完成的单幅画面置于子文件夹中，不应影响其它文件的处理
    single = cases[0].image.crop((0, 0, 975, 1300))
    single.save(tin / "B" / "single.jpg", format="JPEG", quality=92)

    topts = replace(opts, input_dir=tin, output_dir=tout, debug=True)
    reps = process_batch(topts)
    ok_reps = [r for r in reps if r.ok]

    expected = [
        tout / "top_1.jpg", tout / "top_2.jpg",
        tout / "A" / "a1_1.jpg", tout / "A" / "a1_2.jpg",
        tout / "A" / "a2_1.jpg", tout / "A" / "a2_2.jpg",
        tout / "A" / "B" / "deep_1.jpg", tout / "A" / "B" / "deep_2.jpg",
        tout / "B" / "b1_1.jpg", tout / "B" / "b1_2.jpg",
        tout / "B" / "single_1.jpg", tout / "B" / "single_2.jpg",
    ]
    missing = [str(p.relative_to(tout)) for p in expected if not p.exists()]
    debug_expected = [tout / "_debug", tout / "A" / "_debug", tout / "A" / "B" / "_debug",
                      tout / "B" / "_debug"]
    miss_dbg = [str(p.relative_to(tout)) for p in debug_expected if not p.is_dir()]
    stray_top = sorted(p.name for p in tout.glob("*_1.jpg")
                       if p.name not in ("top_1.jpg",))

    if missing or miss_dbg or stray_top or len(ok_reps) != len(plan) + 1:
        return False, "缺失文件={} 缺失预览目录={} 根目录多余文件={} 成功数={}/{}".format(
            missing, miss_dbg, stray_top, len(ok_reps), len(plan) + 1)
    return True, "根目录 1 个、A 2 个、A\\B 1 个、B 2 个全部就位，_debug 目录同步分文件夹"


if __name__ == "__main__":
    raise SystemExit(main())
