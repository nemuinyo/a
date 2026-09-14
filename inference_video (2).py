# diag_frames.py — 使い方:
#   python diag_frames.py aaa.mkv 出力.mp4 --multi 2
import os
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('input')
ap.add_argument('output')
ap.add_argument('--multi', type=int, default=2)
ap.add_argument('--save', type=int, default=10, help='最悪フレームの比較画像保存枚数')
ap.add_argument('--outdir', default='diag')
ap.add_argument('--range', dest='rng', type=str, default=None,
                help='例 200:210 — その範囲の全フレームをPNG保存')
args = ap.parse_args()


def gen(cap):
    while True:
        ok, f = cap.read()
        if not ok:
            return
        yield f


def psnr(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = (d * d).mean()
    return 99.0 if mse == 0 else 10 * np.log10(255.0 * 255.0 / mse)


src_cap = cv2.VideoCapture(args.input)
out_cap = cv2.VideoCapture(args.output)
src_it = gen(src_cap)
os.makedirs(args.outdir, exist_ok=True)

m = args.multi
scores = []          # (psnr, input_idx, output_idx)
cur_k, src_cur = -1, None
n_out = 0

for j, outf in enumerate(gen(out_cap)):
    n_out = j + 1
    k = j // m                       # 出力jに対応する入力フレーム番号
    while cur_k < k:                 # 入力を順に読み進める(メモリ安全)
        src_cur = next(src_it, None)
        cur_k += 1
    if src_cur is None:
        break
    if j % m == 0:                   # 素通しフレームのみ検査
        scores.append((psnr(outf, src_cur), k, j))

    if args.rng:
        a, b = map(int, args.rng.split(':'))
        if a <= j < b:
            cv2.imwrite('%s/out_%06d.png' % (args.outdir, j), outf)

print('output frames: %d' % n_out)
print('checked %d pass-through frames (output even-index vs input)' % len(scores))
arr = np.array([s[0] for s in scores])
print('PSNR: median=%.1f dB, >=35dB: %d, 20-35dB: %d, <20dB: %d' % (
    np.median(arr), (arr >= 35).sum(), ((arr >= 20) & (arr < 35)).sum(), (arr < 20).sum()))

scores.sort()
print('--- worst 15 ---')
for p, k, j in scores[:15]:
    mark = '  <== 破損' if p < 20 else ''
    print('  input#%-6d output#%-7d PSNR=%6.2f dB%s' % (k, j, p, mark))

for i, (p, k, j) in enumerate(scores[:args.save]):
    if src_cur is None or k > cur_k:
        continue
    # 該当入力フレームを再読み込み
    cap = cv2.VideoCapture(args.input)
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, a = cap.read()
    cap.release()
    if not ok:
        continue
    b = cv2.imread  # noop(型 предупрежд防止)
    out_cap.set(cv2.CAP_PROP_POS_FRAMES, j)
    ok2, bb = out_cap.read()
    if not ok2:
        continue
    diff = np.clip(np.abs(bb.astype(int) - a.astype(int)) * 4, 0, 255).astype(np.uint8)
    canvas = np.concatenate((a, bb, diff), 1)
    cv2.imwrite('%s/cmp_%02d_in%d_out%d_%04ddB.jpg' % (args.outdir, i, k, j, int(p)), canvas)
print('比較画像を %s/ に保存しました(左=入力, 中=出力, 右=差分x4)' % args.outdir)
