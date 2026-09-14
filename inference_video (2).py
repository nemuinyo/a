# diag_frames.py
# 使い方: python diag_frames.py 入力動画 出力動画 --multi 2
import os, sys, argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('input')
ap.add_argument('output')
ap.add_argument('--multi', type=int, default=2)
ap.add_argument('--save', type=int, default=8)
ap.add_argument('--outdir', default='diag')
args = ap.parse_args()

def open_or_die(path, tag):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print('ERROR: %s を開けません: %s (パスを確認してください)' % (tag, path))
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('%s: %s  fps=%.3f  frames(概算)=%d' % (tag, path, fps, n))
    return cap, fps

src_cap, _ = open_or_die(args.input, 'input ')
out_cap, ofps = open_or_die(args.output, 'output')
os.makedirs(args.outdir, exist_ok=True)

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

m = args.multi
scores, cur_k, src_cur, n_out = [], -1, None, 0
for j, outf in enumerate(gen(out_cap)):
    n_out = j + 1
    k = j // m
    while cur_k < k:
        src_cur = next(src_it, None) if False else None
        cur_k += 1
    # 入力は都度シーク(シンプル&確実)
    if j % m == 0:
        src_cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, srcf = src_cap.read()
        if not ok:
            print('input frame %d が読めず中断' % k)
            break
        scores.append((psnr(outf, srcf), k, j))

print('output frames actually read: %d' % n_out)
exp = None
if n_out > 0:
    exp = (n_out - 1) // m + 1
    print('expected input-frame coverage: %d' % exp)
if not scores:
    print('比較可能なフレームがありません(出力が空/開けない)')
    sys.exit(1)

arr = np.array([s[0] for s in scores])
print('PSNR: median=%.1f dB | >=35dB: %d | 20-35dB: %d | <20dB: %d' % (
    np.median(arr), int((arr >= 35).sum()), int(((arr >= 20) & (arr < 35)).sum()), int((arr < 20).sum())))
scores.sort()
print('--- worst 15 (input#, output#, PSNR) ---')
for p, k, j in scores[:15]:
    print('  input#%-6d output#%-7d %6.2f dB%s' % (k, j, p, '  <== 破損' if p < 20 else ''))

for i, (p, k, j) in enumerate(scores[:args.save]):
    src_cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, a = src_cap.read()
    out_cap.set(cv2.CAP_PROP_POS_FRAMES, j)
    ok2, bb = out_cap.read()
    if not (ok and ok2):
        continue
    diff = np.clip(np.abs(bb.astype(int) - a.astype(int)) * 4, 0, 255).astype(np.uint8)
    cv2.imwrite('%s/cmp_%02d_in%d_out%d_%04ddB.jpg' % (args.outdir, i, k, j, int(p)),
                np.concatenate((a, bb, diff), 1))
print('比較画像: %s/ (左=入力, 中=出力, 右=差分x4)' % args.outdir)
