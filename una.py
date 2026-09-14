import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import time
import shutil
import argparse
import inspect
import warnings
import traceback
import subprocess
import threading
import multiprocessing as mp
from fractions import Fraction
from queue import Empty, Queue

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # ※ローカル変数にFを使わないこと(過去バグの教訓)
from tqdm import tqdm

warnings.filterwarnings("ignore")
from model.pytorch_msssim import ssim_matlab


def ffprobe_meta(path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'json', path]
    st = json.loads(subprocess.check_output(cmd).decode())['streams'][0]
    num, den = st['r_frame_rate'].split('/')
    fps = Fraction(int(num), int(den))
    try:
        a = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'a',
                                     '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', path]).decode().strip()
        audio = len(a) > 0
    except Exception:
        audio = False
    return int(st['width']), int(st['height']), fps, audio


def probe_nvenc():
    try:
        r = subprocess.run(
            ['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i', 'color=c=black:s=256x256:d=0.2:r=30',
             '-c:v', 'h264_nvenc', '-preset', 'p4', '-rc', 'vbr', '-cq', '19', '-b:v', '0',
             '-pix_fmt', 'yuv420p', '-f', 'null', '-'],
            stderr=subprocess.DEVNULL, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def probe_nvdec():
    tmp = '/tmp/rife_nvdec_probe.mp4'
    try:
        r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-nostdin', '-f', 'lavfi',
                            '-i', 'testsrc2=s=256x256:d=0.5:r=30', '-c:v', 'libx264',
                            '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', tmp],
                           stderr=subprocess.DEVNULL, timeout=60)
        if r.returncode != 0:
            return False
        p = subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-hwaccel', 'cuda', '-i', tmp,
                              '-map', '0:v:0', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        data = p.stdout.read()
        p.wait(timeout=30)
        return len(data) == 15 * 256 * 256 * 3
    except Exception:
        return False
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


class VideoFrames:
    """trim=start_frame でフレーム番号指定(時間シークを使わないのでVFR/start_timeオフセットでもずれない)"""
    def __init__(self, path, h, w_full, start_frame, nvdec):
        self.h, self.w = h, w_full
        self.frame_bytes = h * w_full * 3
        cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1']
        if nvdec:
            cmd += ['-hwaccel', 'cuda']
        cmd += ['-i', path, '-map', '0:v:0', '-vf', 'trim=start_frame=%d' % max(0, start_frame),
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def __iter__(self):
        n = self.frame_bytes
        while True:
            buf = bytearray(n)
            view = memoryview(buf)
            got = 0
            while got < n:
                r = self.proc.stdout.readinto(view[got:])
                if not r:
                    return
                got += r
            yield np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 3)

    def close(self):
        try:
            self.proc.stdout.close()
            self.proc.terminate()
        except Exception:
            pass


class ImageFrames:
    def __init__(self, paths):
        self.paths = paths

    def __iter__(self):
        for p in self.paths:
            f = cv2.imread(p, cv2.IMREAD_COLOR)
            if f is not None:
                yield f

    def close(self):
        pass


def _probe_pair(base_cfg):
    """AMPセルフチェック用に実コンテンツ2フレームを取得"""
    if base_cfg['mode'] == 'video':
        r = VideoFrames(base_cfg['video_path'], base_cfg['h'], base_cfg['w_full'], 0, base_cfg['nvdec'])
        it = iter(r)
        a = next(it, None)
        b = next(it, None)
        r.close()
    else:
        a = cv2.imread(base_cfg['img_paths'][0], cv2.IMREAD_COLOR)
        b = cv2.imread(base_cfg['img_paths'][1], cv2.IMREAD_COLOR) if len(base_cfg['img_paths']) > 1 else None
    if a is None:
        raise RuntimeError('probe frame read failed')
    if base_cfg['left']:
        a = a[:, base_cfg['left']:base_cfg['left'] + base_cfg['w']]
        if b is not None:
            b = b[:, base_cfg['left']:base_cfg['left'] + base_cfg['w']]
    if b is None:
        b = a
    return a, b


def _seg_writer(outq, cfg, msg_q):
    try:
        if cfg['png']:
            i = cfg['png_start']
            while True:
                img = outq.get()
                if img is None:
                    break
                cv2.imwrite('vid_out/%07d.png' % i, img)
                i += 1
        else:
            fn, fd = cfg['ofps_frac']
            fr = '%d/%d' % (fn, fd)  # ofps_fracは既にmulti反映済み。二重掛け禁止(過去のスピードバグの教訓)
            cmd = ['ffmpeg', '-v', 'error', '-y', '-nostdin',
                   '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (cfg['ow'], cfg['h']),
                   '-framerate', fr, '-i', '-', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-r', fr]
            if cfg['nvenc']:
                cmd += ['-c:v', 'h264_nvenc', '-preset', 'p4', '-rc', 'vbr',
                        '-cq', str(cfg['crf']), '-b:v', '0', '-pix_fmt', 'yuv420p']
            else:
                cmd += ['-c:v', 'libx264', '-preset', cfg['preset'], '-crf', str(cfg['crf']),
                        '-pix_fmt', 'yuv420p', '-threads', '1']
            cmd += [cfg['seg_path']]
            enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            while True:
                img = outq.get()
                if img is None:
                    break
                if not img.flags.c_contiguous:
                    img = np.ascontiguousarray(img)
                enc.stdin.write(img.data)
            enc.stdin.close()
            if enc.wait() != 0:
                raise RuntimeError('encoder failed: ' + cfg['seg_path'])
    except Exception:
        traceback.print_exc()
        msg_q.put(('error', 'segment writer failed'))


def make_inference(model, I0, I1, n, scale, ver):
    if n <= 0:
        return []
    if ver >= 3.9:
        if n == 1:
            return [model.inference(I0, I1, 0.5, scale)]
        outs = []
        CH = 4
        for s0 in range(0, n, CH):
            m = min(CH, n - s0)
            if m == 1:
                outs.append(model.inference(I0, I1, (s0 + 1) / (n + 1), scale))
                continue
            try:
                ts = (torch.arange(s0 + 1, s0 + m + 1, device=I0.device, dtype=I0.dtype) / (n + 1)).view(-1, 1, 1, 1)
                mm = model.inference(I0.repeat(m, 1, 1, 1), I1.repeat(m, 1, 1, 1), ts, scale)
                outs.extend(mm[i:i + 1] for i in range(m))
            except Exception:
                outs.extend(model.inference(I0, I1, (i + 1) / (n + 1), scale) for i in range(s0, s0 + m))
        return outs
    middle = model.inference(I0, I1, scale)
    if n == 1:
        return [middle]
    first = make_inference(model, I0, middle, n // 2, scale, ver)
    second = make_inference(model, middle, I1, n // 2, scale, ver)
    return [*first, middle, *second] if n % 2 else [*first, *second]


def infer_mid(model, I0, I1, scale, ver):
    return model.inference(I0, I1, 0.5, scale) if ver >= 3.9 else model.inference(I0, I1, scale)


def gpu_ssim(a, b):
    sa = F.interpolate(a, (32, 32), mode='bilinear', align_corners=False)
    sb = F.interpolate(b, (32, 32), mode='bilinear', align_corners=False)
    return float(ssim_matlab(sa[:, :3], sb[:, :3]))


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower()


def gpu_worker(gid, base_cfg, chunk_q, msg_q):
    tag = '[gpu%d]' % gid
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        dev = torch.device('cuda')

        from train_log.RIFE_HDv3 import Model
        model = Model()
        if not hasattr(model, 'version'):
            model.version = 0
        model.load_model(base_cfg['model_dir'], -1)
        model.eval()
        for name in list(vars(model)):
            o = getattr(model, name)
            if isinstance(o, torch.nn.Module):
                o.to(dev)
        ver = getattr(model, 'version', 0)
        if ver < 3.9:
            try:
                if 'timestep' in inspect.signature(model.inference).parameters:
                    ver = 3.9
            except Exception:
                pass
        print(tag, 'model version:', ver)

        h, w = base_cfg['h'], base_cfg['w']
        padding = tuple(base_cfg['padding'])
        montage, multi = base_cfg['montage'], base_cfg['multi']
        pin = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)

        def load(u8):
            np.copyto(pin.numpy(), np.ascontiguousarray(u8))
            x = pin.to(dev)
            x = x.permute(2, 0, 1)[[2, 1, 0]].unsqueeze(0).float().div_(255.)  # BGR→RGB
            return F.pad(x, padding)

        def img_from(t):
            x = t[0, :, :h, :w].permute(1, 2, 0).float().mul_(255.).clamp_(0, 255).byte().flip(2)  # RGB→BGR
            return x.contiguous().cpu().numpy()

        # ---- 起動時メモリプローブ ----
        ph, pw = h + padding[3], w + padding[1]
        a = torch.zeros(1, 3, ph, pw, device=dev)
        b = torch.zeros_like(a)
        sc = base_cfg['scale']
        while True:
            try:
                with torch.autocast('cuda', dtype=torch.float16, enabled=not base_cfg['fp32']):
                    infer_mid(model, a, b, sc, ver)
                break
            except Exception as e:
                if not is_oom(e) or sc <= 0.2501:
                    raise
                sc = max(0.25, sc / 2)
                torch.cuda.empty_cache()
        del a, b
        torch.cuda.empty_cache()

        # ---- AMPセルフチェック: 実コンテンツ2フレームで判定(乱数は病的すぎるため) ----
        amp_mode = base_cfg['amp_mode']
        amp_ok = (amp_mode != 'off')
        if amp_ok and amp_mode == 'auto':
            try:
                a8, b8 = _probe_pair(base_cfg)
                A, B = load(a8), load(b8)
                with torch.autocast('cuda', enabled=False):
                    r1 = infer_mid(model, A, B, sc, ver)
                with torch.autocast('cuda', dtype=torch.float16):
                    r2 = infer_mid(model, A, B, sc, ver)
                d = (r1.double() - r2.double()).abs().max().item()
                if (not np.isfinite(d)) or d > 0.02:
                    amp_ok = False
                    print(tag, 'AMP check FAILED on real frames (maxdiff=%.4f) -> fp32' % d)
                else:
                    print(tag, 'AMP check ok (maxdiff=%.5f) -> fp16 autocast' % d)
                del A, B, r1, r2
            except Exception as e:
                amp_ok = False
                print(tag, 'AMP check error -> fp32 :', repr(e))
        elif amp_ok:
            print(tag, 'AMP forced ON (--fp16)')

        def ac():
            return torch.autocast('cuda', dtype=torch.float16, enabled=amp_ok)

        # ---- バッチ推論チェック(multi==2のpair連結バッチ) ----
        batch = base_cfg['batch']
        if batch > 1 and multi == 2 and ver >= 3.9:
            try:
                g = torch.Generator().manual_seed(7)
                x = torch.rand((1, 3, 256, 256), generator=g).to(dev)
                y = torch.rand((1, 3, 256, 256), generator=g).to(dev)
                z = torch.rand((1, 3, 256, 256), generator=g).to(dev)
                with ac():
                    bat = model.inference(torch.cat([x, y]), torch.cat([y, z]), 0.5, 1.0)
                    s0 = model.inference(x, y, 0.5, 1.0)
                    s1 = model.inference(y, z, 0.5, 1.0)
                d = max((bat[0:1] - s0).abs().max().item(), (bat[1:2] - s1).abs().max().item())
                if not np.isfinite(d) or d > 0.05:
                    print(tag, 'batch check FAILED (maxdiff=%.4f) -> batch=1' % d)
                    batch = 1
                else:
                    print(tag, 'batch check ok (maxdiff=%.5f, batch=%d)' % (d, batch))
                del x, y, z, bat, s0, s1
            except Exception as e:
                print(tag, 'batch unsupported -> batch=1 :', repr(e))
                batch = 1
        elif batch > 1 and multi != 2:
            batch = 1  # バッチパスはmulti==2専用

        print(tag, 'ready (scale=%g, amp=%s, batch=%d)' % (sc, amp_ok, batch))
        msg_q.put(('ready', gid))

        def run_oom_safe(fn):
            s = sc
            while True:
                try:
                    with ac():
                        return fn(s)
                except Exception as e:
                    if not is_oom(e) or s <= 0.2501:
                        raise
                    s = max(0.25, s / 2)
                    torch.cuda.empty_cache()
                    print(tag, 'OOM -> scale=%g' % s)

        # ---- チャンクループ(動的割当) ----
        while True:
            ch = chunk_q.get()
            if ch is None:
                msg_q.put(('done', gid))  # ★修正: 正常終了を親に通知(前回の誤検知の原因)
                return
            t0 = time.time()
            segcfg = dict(base_cfg)
            segcfg['seg_path'] = ch['seg']
            segcfg['png_start'] = ch['png_start']
            outq = Queue(maxsize=64)
            wt = threading.Thread(target=_seg_writer, args=(outq, segcfg, msg_q), daemon=True)
            wt.start()

            if base_cfg['mode'] == 'video':
                reader = VideoFrames(base_cfg['video_path'], h, base_cfg['w_full'],
                                     ch['first_iter'] - 1, base_cfg['nvdec'])
            else:
                reader = ImageFrames(base_cfg['img_paths'][ch['first_iter'] - 1:])
            it = iter(reader)

            def next_crop():
                fr = next(it)
                if base_cfg['left']:
                    fr = fr[:, base_cfg['left']:base_cfg['left'] + w]
                return fr

            state_img = next_crop()
            Fc = load(state_img)
            pending = None
            ops = ch['ops']
            i, M = 0, len(ops)

            while i < M:
                kind = ops[i]

                # 連続pairの一括バッチ(multi==2のみ)
                if batch > 1 and pending is None and kind == 'pair':
                    r, frs = 0, []
                    while i + r < M and ops[i + r] == 'pair' and r < batch:
                        frs.append(next_crop())
                        r += 1
                    Ts = [load(x) for x in frs]
                    I0 = torch.cat([Fc] + Ts[:-1], 0)
                    I1 = torch.cat(Ts, 0)
                    outs = run_oom_safe(lambda s: model.inference(I0, I1, 0.5, s))
                    for j in range(r):
                        li = state_img
                        outq.put(np.concatenate((li, li), 1) if montage else li)
                        m = img_from(outs[j:j + 1])
                        outq.put(np.concatenate((li, m), 1) if montage else m)
                        state_img = frs[j]
                        Fc = Ts[j]
                    i += r
                    msg_q.put(('prog', r))
                    continue

                fk_img = pending if pending is not None else next_crop()
                pending = None
                li = state_img
                outq.put(np.concatenate((li, li), 1) if montage else li)

                if kind == 'dup':
                    mids = [li] * (multi - 1)  # ★修正: multi>2対応
                    state_img = fk_img
                    Fc = load(fk_img)
                elif kind == 'pair':
                    Fk = load(fk_img)
                    outs = run_oom_safe(lambda s: make_inference(model, Fc, Fk, multi - 1, s, ver))
                    mids = [img_from(o) for o in outs]
                    state_img = fk_img
                    Fc = Fk
                elif kind in ('static', 'laststatic'):
                    if kind == 'static':
                        nxt = next_crop()      # f_{k+1}
                        src = nxt
                    else:
                        src = fk_img           # 動画末尾: f_k自身(元コードのbreak経路)
                    Fsrc = load(src)
                    D = run_oom_safe(lambda s: infer_mid(model, Fc, Fsrc, s, ver))
                    if gpu_ssim(Fc, D) < 0.2:
                        mids = [li] * (multi - 1)  # ★修正: multi>2対応
                    else:
                        outs = run_oom_safe(lambda s: make_inference(model, Fc, D, multi - 1, s, ver))
                        mids = [img_from(o) for o in outs]
                    state_img = img_from(D)
                    Fc = D
                    if kind == 'static' and i < M - 1:
                        pending = nxt          # 元コード同様、次iterationはf_{k+1}を再利用
                else:
                    raise RuntimeError('unknown op: ' + kind)

                for m in mids:
                    outq.put(np.concatenate((state_img, m), 1) if montage else m)
                msg_q.put(('prog', 1))
                i += 1

            if ch['is_last']:
                outq.put(np.concatenate((state_img, state_img), 1) if montage else state_img)
                msg_q.put(('prog', 1))
            outq.put(None)
            wt.join(timeout=300)
            reader.close()
            msg_q.put(('segdone', ch['cid'], gid, time.time() - t0))
            print(tag, 'chunk %d done in %.1fs' % (ch['cid'], time.time() - t0))
    except Exception:
        traceback.print_exc()
        msg_q.put(('error', 'gpu%d worker crashed' % gid))


# ---------------- プランナ(親・32x32プロキシのみ) ----------------

def read_proxy32(path):
    proc = subprocess.Popen(
        ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-i', path, '-map', '0:v:0',
         '-vf', 'scale=32:32:flags=area', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = 32 * 32 * 3
    frames = []
    while True:
        buf = proc.stdout.read(n)
        if not buf or len(buf) < n:
            break
        frames.append(np.frombuffer(buf, dtype=np.uint8).reshape(32, 32, 3))
    proc.stdout.close()
    proc.wait()
    return frames


def _t32(u8):
    return torch.from_numpy(np.ascontiguousarray(u8.transpose(2, 0, 1))[None]).float().div_(255.)


def build_ops(frames32, multi):
    N = len(frames32)
    ops = [None] * N
    Fp = frames32[0].astype(np.float32)
    for k in range(1, N):
        s = float(ssim_matlab(_t32(Fp), _t32(frames32[k])))
        if s > 0.996:
            if k + 1 <= N - 1:
                ops[k] = 'static'
                Fp = (Fp + frames32[k + 1].astype(np.float32)) * 0.5
            else:
                ops[k] = 'laststatic'
        elif s < 0.2:
            ops[k] = 'dup'
            Fp = frames32[k].astype(np.float32)
        else:
            ops[k] = 'pair'
            Fp = frames32[k].astype(np.float32)
    return ops[1:]


def split_chunks(ops, nsplit):
    M = len(ops)
    if nsplit <= 1 or M == 0:
        return [(1, ops)]
    wmap = {'pair': 1.0, 'dup': 0.05, 'static': 2.0, 'laststatic': 2.0}
    target = max(sum(wmap[o] for o in ops), 1e-9) / nsplit
    cuts, acc, nxt_t = [], 0.0, target
    for i in range(M - 1):
        acc += wmap[ops[i]]
        remain = nsplit - len(cuts)
        if ops[i] in ('pair', 'dup') and acc >= nxt_t and (M - (i + 1)) >= (remain - 1):
            cuts.append(i + 1)
            nxt_t += target
    bounds = [0] + cuts + [M]
    return [(bounds[c] + 1, ops[bounds[c]:bounds[c + 1]]) for c in range(len(bounds) - 1)
            if bounds[c] < bounds[c + 1]]


def parse_args():
    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--video', dest='video', type=str, default=None)
    parser.add_argument('--output', dest='output', type=str, default=None)
    parser.add_argument('--img', dest='img', type=str, default=None)
    parser.add_argument('--montage', dest='montage', action='store_true')
    parser.add_argument('--model', dest='modelDir', type=str, default='train_log')
    parser.add_argument('--fp16', dest='fp16', action='store_true', help='チェック無視でAMP強制ON')
    parser.add_argument('--fp32', dest='fp32', action='store_true', help='AMP無効')
    parser.add_argument('--UHD', dest='UHD', action='store_true')
    parser.add_argument('--scale', dest='scale', type=float, default=1.0)
    parser.add_argument('--fps', dest='fps', type=int, default=None)
    parser.add_argument('--png', dest='png', action='store_true')
    parser.add_argument('--ext', dest='ext', type=str, default='mp4')
    parser.add_argument('--exp', dest='exp', type=int, default=1)
    parser.add_argument('--multi', dest='multi', type=int, default=2)
    parser.add_argument('--ngpu', dest='ngpu', type=int, default=0)
    parser.add_argument('--crf', dest='crf', type=int, default=18)
    parser.add_argument('--preset', dest='preset', type=str, default='veryfast')
    parser.add_argument('--batch', dest='batch', type=int, default=1, help='multi==2時のpair連続バッチ(2〜4)')
    parser.add_argument('--split', dest='split', type=int, default=16, help='チャンク分割数')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.exp != 1:
        args.multi = 2 ** args.exp
    assert (args.video is not None) != (args.img is not None)
    if args.UHD and args.scale == 1.0:
        args.scale = 0.5
    assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
    if args.img is not None:
        args.png = True
    amp_mode = 'off' if args.fp32 else ('on' if args.fp16 else 'auto')

    assert torch.cuda.is_available(), 'GPU required'
    ngpu = args.ngpu if args.ngpu > 0 else min(2, torch.cuda.device_count())
    ngpu = max(1, min(ngpu, torch.cuda.device_count()))

    files = None
    fpsNotAssigned = False
    video_path_wo_ext = None
    if args.video is not None:
        assert shutil.which('ffmpeg'), 'ffmpeg が必要です'
        w_full, h, fps, audio = ffprobe_meta(args.video)
        ofps = fps * args.multi
        if args.fps is None:
            fpsNotAssigned = True
        else:
            ofps = Fraction(args.fps, 1)
        video_path_wo_ext, _ = os.path.splitext(args.video)
        frames32 = read_proxy32(args.video)
        N = len(frames32)
        print('input: {}, {} frames, {}FPS -> {}FPS'.format(args.video, N, float(fps), float(ofps)))
        print('The audio will be merged after interpolation process' if (not args.png and fpsNotAssigned)
              else 'Will not merge audio because using png or fps flag!')
    else:
        files = sorted([f for f in os.listdir(args.img) if 'png' in f], key=lambda x: int(x[:-4]))
        f0 = cv2.imread(os.path.join(args.img, files[0]), cv2.IMREAD_COLOR)
        assert f0 is not None
        h, w_full = f0.shape[:2]
        frames32 = [cv2.resize(cv2.imread(os.path.join(args.img, f), cv2.IMREAD_COLOR), (32, 32),
                               interpolation=cv2.INTER_AREA) for f in files]
        N = len(frames32)
        ofps, audio = None, False
        print('image sequence: {} frames'.format(N))

    nvenc = (not args.png) and probe_nvenc()
    nvdec = (args.video is not None) and probe_nvdec()
    print('hardware: nvenc={} nvdec={}'.format(nvenc, nvdec))

    left = (w_full // 4) if args.montage else 0
    w = (w_full // 2) if args.montage else w_full
    ow = w * 2 if args.montage else w

    tmp = max(128, int(128 / args.scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    ops = build_ops(frames32, args.multi)
    M = len(ops)
    nsplit = max(1, min(args.split, M))
    parts = split_chunks(ops, nsplit)
    chunks = []
    for cid, (fi, cop) in enumerate(parts):
        chunks.append(dict(cid=cid, first_iter=fi, ops=cop, is_last=(cid == len(parts) - 1),
                           seg=os.path.abspath('./rife_seg_%d_%d.%s' % (os.getpid(), cid, args.ext)),
                           png_start=(fi - 1) * args.multi))
    for c in chunks:
        print('  chunk %d: iters %d..%d (%d ops)' % (c['cid'], c['first_iter'],
                                                     c['first_iter'] + len(c['ops']) - 1, len(c['ops'])))

    ctx = mp.get_context('spawn')
    msg_q = ctx.Queue()
    chunk_q = ctx.Queue()
    base_cfg = dict(mode='video' if args.video is not None else 'img',
                    video_path=args.video,
                    img_paths=[os.path.join(args.img, f) for f in files] if files else None,
                    ofps_frac=(ofps.numerator, ofps.denominator) if ofps else (1, 1),
                    h=h, w=w, w_full=w_full, left=left, ow=ow, montage=args.montage,
                    multi=args.multi, model_dir=args.modelDir, amp_mode=amp_mode,
                    padding=padding, png=args.png, scale=args.scale,
                    crf=args.crf, preset=args.preset, batch=args.batch,
                    nvenc=nvenc, nvdec=nvdec)

    procs = []
    for g in range(ngpu):
        p = ctx.Process(target=gpu_worker, daemon=True, args=(g, base_cfg, chunk_q, msg_q))
        p.start()
        procs.append(p)
    for c in chunks:
        chunk_q.put(c)
    for _ in procs:
        chunk_q.put(None)

    pbar = tqdm(total=max(1, N))
    pending = set(c['cid'] for c in chunks)
    by_id = {c['cid']: c for c in chunks}
    busy = {g: 0.0 for g in range(len(procs))}
    ccount = {g: 0 for g in range(len(procs))}
    requeued, err, t0, last_ev = False, None, time.time(), time.time()
    while True:
        try:
            m = msg_q.get(timeout=2)
            last_ev = time.time()
        except Empty:
            alive = sum(1 for p in procs if p.is_alive())
            if pending and alive:
                if alive < len(procs) and not requeued:
                    requeued = True
                    print('[scheduler] worker died -> requeue %d chunk(s)' % len(pending))
                    for cid in pending:
                        chunk_q.put(by_id[cid])
                continue
            if pending and alive == 0:
                err = err or 'gpu worker(s) died unexpectedly'
                break
            if not pending and alive == 0:
                break
            if time.time() - last_ev > 600:
                err = err or 'stalled (no progress for 600s)'
                break
            continue
        k = m[0]
        if k == 'prog':
            pbar.update(m[1])
        elif k == 'segdone':
            pending.discard(m[1])
            busy[m[2]] += m[3]
            ccount[m[2]] += 1
        elif k == 'done':
            pass
        elif k == 'error':
            err = m[1]
            break
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=60)
    pbar.close()
    if err:
        raise RuntimeError(err)
    for g in range(len(procs)):
        print('  gpu%d: %d chunks, busy %.1fs' % (g, ccount[g], busy[g]))
    elapsed = time.time() - t0
    print('Interpolation: %.1f sec (%.2f input frames/sec)' % (elapsed, N / max(elapsed, 1e-6)))

    if args.png:
        print('done: vid_out/*.png')
        return

    for c in chunks:
        assert os.path.exists(c['seg']) and os.path.getsize(c['seg']) > 0, 'segment missing: ' + c['seg']

    vid_out_name = args.output if args.output is not None else \
        '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(float(ofps))), args.ext)
    listf = './rife_concat_%d.txt' % os.getpid()
    with open(listf, 'w') as f:
        for c in chunks:
            f.write("file '%s'\n" % c['seg'])

    ok = False
    if args.video is not None and fpsNotAssigned and audio:
        try:
            r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-nostdin', '-f', 'concat', '-safe', '0',
                                '-i', listf, '-i', args.video, '-map', '0:v:0', '-map', '1:a:0',
                                '-c', 'copy', '-movflags', '+faststart', vid_out_name],
                               stderr=subprocess.DEVNULL)
            ok = r.returncode == 0 and os.path.getsize(vid_out_name) > 0
        except Exception:
            ok = False
        if not ok:
            try:
                r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-nostdin', '-f', 'concat', '-safe', '0',
                                    '-i', listf, '-i', args.video, '-map', '0:v:0', '-map', '1:a:0',
                                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k',
                                    '-movflags', '+faststart', vid_out_name],
                                   stderr=subprocess.DEVNULL)
                ok = r.returncode == 0 and os.path.getsize(vid_out_name) > 0
                if ok:
                    print('Lossless audio transfer failed. Audio was transcoded to AAC instead.')
            except Exception:
                ok = False
    if not ok:
        r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-nostdin', '-f', 'concat', '-safe', '0',
                            '-i', listf, '-c', 'copy', '-movflags', '+faststart', vid_out_name],
                           stderr=subprocess.DEVNULL)
        if not (r.returncode == 0 and os.path.getsize(vid_out_name) > 0) and fpsNotAssigned and args.video is not None:
            print('Audio transfer failed. Interpolated video will have no audio')
    os.remove(listf)
    for c in chunks:
        try:
            os.remove(c['seg'])
        except OSError:
            pass
    print('done:', vid_out_name)


if __name__ == '__main__':
    main()
