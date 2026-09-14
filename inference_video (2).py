# diag_fast.py — 使い方: python diag_fast.py aaa.mkv aaa_2X_48fps.mp4 --multi 2
import os, sys, argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('input'); ap.add_argument('output')
ap.add_argument('--multi', type=int, default=2)
ap.add_argument('--outdir', default='diag')
args = ap.parse_args()

src = cv2.VideoCapture(args.input)
out = cv2.VideoCapture(args.output)
if not src.isOpened() or not out.isOpened():
    print('ERROR: 動画を開けません'); sys.exit(1)
os.makedirs(args.outdir, exist_ok=True)

def psnr(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = (d * d).mean()
    return 99.0 if mse == 0 else 10 * np.log10(255.0 * 255.0 / mse)

m, scores, n_in, j = args.multi, [], 0, 0
while True:
    ok, outf = out.read()
    if not ok:
        break
    j += 1
    k = (j - 1) // m
    if (j - 1) % m == 0:          # 素通しフレームのみ検査
        while n_in < k:
            if src.read()[0]:
                n_in += 1
        ok2, srcf = src.read()
        n_in += 1
        if not ok2:
            break
        p = psnr(outf, srcf)
        scores.append((p, k, j - 1))
        if p < 25:                # 破損フレームは即保存
            diff = np.clip(np.abs(outf.astype(int) - srcf.astype(int)) * 4, 0, 255).astype(np.uint8)
            cv2.imwrite('%s/bad_in%d_out%d_%04ddB.jpg' % (args.outdir, k, j - 1, int(p)),
                        np.concatenate((srcf, outf, diff), 1))

print('output frames: %d, input consumed: %d' % (j, n_in))
if not scores:
    print('比較できず'); sys.exit(1)
arr = np.array([s[0] for s in scores])
print('PSNR: median=%.1f dB | >=35dB: %d | 20-35dB: %d | <20dB: %d' % (
    np.median(arr), int((arr >= 35).sum()), int(((arr >= 20) & (arr < 35)).sum()), int((arr < 20).sum())))
scores.sort()
print('--- worst 10 ---')
for p, k, jo in scores[:10]:
    print('  input#%-6d output#%-7d %6.2f dB%s' % (k, jo, p, '  <== 破損' if p < 20 else ''))
print('破損画像(あれば): %s/ (左=入力, 中=出力, 右=差分x4)' % args.outdir)
