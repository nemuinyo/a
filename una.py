import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import time
import shutil
import argparse
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
import torch.nn.functional as F  # ※テンソル変数名とは別物。ワーカー内ではFcを使用
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


class VideoFrames:
    def __init__(self, path, h, w_full, start_idx, fps):
        self.h, self.w = h, w_full
        self.frame_bytes = h * w_full * 3
        cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1']
        if start_idx > 0:
            cmd += ['-ss', '%.6f' % max(0.0, (start_idx - 0.5) / float(fps))]
        cmd += ['-i', path, '-map', '0:v:0', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
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


class ImageFrames:
    def __init__(self, paths):
        self.paths = paths

    def __iter__(self):
        for p in self.paths:
            f = cv2.imread(p, cv2.IMREAD_COLOR)
            if f is not None:
                yield f


def _output_writer(outq, cfg, msg_q):
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
            fr = '%d/%d' % (fn * cfg['multi'], fd) if cfg['ofps_auto'] else '%d' % fn
            enc = subprocess.Popen(
                ['ffmpeg', '-v', 'error', '-y', '-nostdin',
                 '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (cfg['ow'], cfg['h']),
                 '-framerate', fr, '-i', '-',
                 '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                 '-c:v', 'libx264', '-preset', cfg['preset'], '-crf', str(cfg['crf']),
                 '-pix_fmt', 'yuv420p', '-threads', '1', '-r', fr, cfg['seg_path']],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            while True:
                img = outq.get()
                if img is None:
                    break
                if not img.flags.c_contiguous:
                    img = np.ascontiguousarray(img)
                enc.stdin.write(img.data)
            enc.stdin.close()
            if enc.wait() != 0:
                raise RuntimeError('encoder failed')
    except Exception:
        traceback.print_exc()
        msg_q.put(('error', 'output writer failed'))


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


def gpu_worker(gid, cfg, ops, first_iter, is_last, msg_q):
    tag = '[gpu%d]' % gid
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        cv2.setNumThreads(1)
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(1)
        dev = torch.device('cuda')

        outq = Queue(maxsize=64)
        wt = threading.Thread(target=_output_writer, args=(outq, cfg, msg_q), daemon=True)
        wt.start()

        from train_log.RIFE_HDv3 import Model
        model = Model()
        if not hasattr(model, 'version'):
            model.version = 0
        model.load_model(cfg['model_dir'], -1)
        print(tag + ' Loaded 3.x/4.x HD model.')
        model.eval()
        for name in list(vars(model)):
            o = getattr(model, name)
            if isinstance(o, torch.nn.Module):
                o.to(dev)
        ver = getattr(model, 'version', 0)

        h, w, padding, multi = cfg['h'], cfg['w'], tuple(cfg['padding']), cfg['multi']
        pin = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)

        def load(u8):
            np.copyto(pin.numpy(), np.ascontiguousarray(u8))
            x = pin.to(dev)
            x = x.permute(2, 0, 1)[[2, 1, 0]].unsqueeze(0).float().div_(255.)  # BGR→RGB
            return F.pad(x, padding)

        def img_from(t):
            x = t[0, :, :h, :w].permute(1, 2, 0).float().mul_(255.).clamp_(0, 255).byte().flip(2)  # RGB→BGR
            return x.contiguous().cpu().numpy()

        ph, pw = h + padding[3], w + padding[1]
        a = torch.zeros(1, 3, ph, pw, device=dev)
        b = torch.zeros_like(a)
        sc = cfg['scale']
        while True:
            try:
                with torch.autocast('cuda', dtype=torch.float16, enabled=not cfg['fp32']):
                    infer_mid(model, a, b, sc, ver)
                break
            except Exception as e:
                if not is_oom(e) or sc <= 0.2501:
                    raise
                sc = max(0.25, sc / 2)
                torch.cuda.empty_cache()
        del a, b
        torch.cuda.empty_cache()
        print(tag + ' ready (scale=%g)' % sc)
        msg_q.put(('ready',))

        if cfg['mode'] == 'video':
            fn, fd = cfg['fps_frac']
            reader = VideoFrames(cfg['video_path'], h, cfg['w_full'], first_iter - 1, Fraction(fn, fd))
        else:
            reader = ImageFrames([os.path.join(cfg['img_dir'], f) for f in cfg['files'][first_iter - 1:]])
        it = iter(reader)

        def next_crop():
            fr = next(it)
            if cfg['left']:
                fr = fr[:, cfg['left']:cfg['left'] + w]
            return fr

        state_img = next_crop()      # f_{first_iter-1}
        Fc = load(state_img)         # ★ 改名: F モジュールとの衝突を解消
        pending = None
        M = len(ops)

        def run_oom_safe(fn, sc):
            while True:
                try:
                    with torch.autocast('cuda', dtype=torch.float16, enabled=not cfg['fp32']):
                        return fn(sc)
                except Exception as e:
                    if not is_oom(e) or sc <= 0.2501:
                        raise
                    sc = max(0.25, sc / 2)
                    torch.cuda.empty_cache()
                    print(tag + ' OOM -> scale=%g' % sc)

        for idx, kind in enumerate(ops):
            fk_img = pending if pending is not None else next_crop()
            pending = None
            base = np.concatenate((state_img, state_img), 1) if cfg['montage'] else state_img
            outq.put(base)

            if kind == 'dup':
                mids = [state_img] * (multi - 1)
                state_img = fk_img
                Fc = load(fk_img)
            elif kind == 'pair':
                Fk = load(fk_img)
                outs = run_oom_safe(lambda s: make_inference(model, Fc, Fk, multi - 1, s, ver), sc)
                mids = [img_from(o) for o in outs]
                state_img = fk_img
                Fc = Fk
            elif kind == 'static':
                nxt_img = next_crop()
                Fnx = load(nxt_img)
                D = run_oom_safe(lambda s: infer_mid(model, Fc, Fnx, s, ver), sc)
                if gpu_ssim(Fc, D) < 0.2:
                    mids = [state_img] * (multi - 1)
                else:
                    outs = run_oom_safe(lambda s: make_inference(model, Fc, D, multi - 1, s, ver), sc)
                    mids = [img_from(o) for o in outs]
                state_img = img_from(D)
                Fc = D
                if idx < M - 1:
                    pending = nxt_img
            else:  # laststatic
                D = run_oom_safe(lambda s: infer_mid(model, Fc, Fc, s, ver), sc)
                outs = run_oom_safe(lambda s: make_inference(model, Fc, D, multi - 1, s, ver), sc)
                mids = [img_from(o) for o in outs]
                state_img = img_from(D)
                Fc = D

            for m in mids:
                outq.put(np.concatenate((state_img, m), 1) if cfg['montage'] else m)
            msg_q.put(('prog', 1))

        if is_last:
            outq.put(np.concatenate((state_img, state_img), 1) if cfg['montage'] else state_img)
            msg_q.put(('prog', 1))
        outq.put(None)
        wt.join(timeout=300)
        msg_q.put(('done',))
    except Exception:
        traceback.print_exc()
        msg_q.put(('error', 'gpu%d worker crashed' % gid))


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


def split_chunks(ops, nchunk, multi):
    M = len(ops)
    if nchunk <= 1 or M == 0:
        return [(1, ops)]
    wmap = {'pair': float(max(1, multi - 1)), 'dup': 0.05, 'static': float(multi), 'laststatic': float(multi)}
    target = max(sum(wmap[o] for o in ops), 1e-9) / nchunk
    chunks, start, acc, nxt_target = [], 0, 0.0, target
    for i in range(M - 1):
        acc += wmap[ops[i]]
        remain = nchunk - len(chunks)
        if ops[i] not in ('static', 'laststatic') and acc >= nxt_target and (M - (i + 1)) >= (remain - 1):
            chunks.append((start + 1, ops[start:i + 1]))
            start = i + 1
            nxt_target += target
    chunks.append((start + 1, ops[start:]))
    return chunks


def parse_args():
    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--video', dest='video', type=str, default=None)
    parser.add_argument('--output', dest='output', type=str, default=None)
    parser.add_argument('--img', dest='img', type=str, default=None)
    parser.add_argument('--montage', dest='montage', action='store_true')
    parser.add_argument('--model', dest='modelDir', type=str, default='train_log')
    parser.add_argument('--fp16', dest='fp16', action='store_true', help='(AMPはデフォルトON。互換用)')
    parser.add_argument('--fp32', dest='fp32', action='store_true', help='AMP(fp16)を無効化')
    parser.add_argument('--UHD', dest='UHD', action='store_true')
    parser.add_argument('--scale', dest='scale', type=float, default=1.0)
    parser.add_argument('--skip', dest='skip', action='store_true')
    parser.add_argument('--fps', dest='fps', type=int, default=None)
    parser.add_argument('--png', dest='png', action='store_true')
    parser.add_argument('--ext', dest='ext', type=str, default='mp4')
    parser.add_argument('--exp', dest='exp', type=int, default=1)
    parser.add_argument('--multi', dest='multi', type=int, default=2)
    parser.add_argument('--ngpu', dest='ngpu', type=int, default=0, help='0=自動(最大2)')
    parser.add_argument('--crf', dest='crf', type=int, default=18)
    parser.add_argument('--preset', dest='preset', type=str, default='veryfast')
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

    have_ffmpeg = shutil.which('ffmpeg') is not None
    assert torch.cuda.is_available(), 'GPU required'
    ngpu = args.ngpu if args.ngpu > 0 else min(2, torch.cuda.device_count())
    ngpu = max(1, min(ngpu, torch.cuda.device_count()))
    if args.video is not None:
        assert have_ffmpeg, 'ffmpeg が必要です'

    files = None
    fpsNotAssigned = False
    video_path_wo_ext = None
    if args.video is not None:
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
        if not args.png and fpsNotAssigned:
            print('The audio will be merged after interpolation process')
        else:
            print('Will not merge audio because using png or fps flag!')
    else:
        files = sorted([f for f in os.listdir(args.img) if 'png' in f], key=lambda x: int(x[:-4]))
        f0 = cv2.imread(os.path.join(args.img, files[0]), cv2.IMREAD_COLOR)
        assert f0 is not None
        h, w_full = f0.shape[:2]
        fps = None
        frames32 = [cv2.resize(cv2.imread(os.path.join(args.img, f), cv2.IMREAD_COLOR), (32, 32),
                               interpolation=cv2.INTER_AREA) for f in files]
        N = len(frames32)
        print('image sequence: {} frames'.format(N))

    left = (w_full // 4) if args.montage else 0
    w = (w_full // 2) if args.montage else w_full
    ow = w * 2 if args.montage else w

    tmp = max(128, int(128 / args.scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    ops = build_ops(frames32, args.multi)
    M = len(ops)
    nchunk = max(1, min(ngpu, max(1, M)))
    chunks = split_chunks(ops, nchunk, args.multi)
    for i, (fi, c) in enumerate(chunks):
        print('  gpu%d: iterations %d..%d (%d ops)' % (i, fi, fi + len(c) - 1, len(c)))

    ctx = mp.get_context('spawn')
    msg_q = ctx.Queue()
    base_cfg = dict(mode='video' if args.video is not None else 'img',
                    video_path=args.video, img_dir=args.img, files=files,
                    fps_frac=(fps.numerator, fps.denominator) if fps else (1, 1),
                    ofps_frac=(ofps.numerator, ofps.denominator), ofps_auto=fpsNotAssigned,
                    h=h, w=w, w_full=w_full, left=left, ow=ow, montage=args.montage,
                    multi=args.multi, fp32=args.fp32, model_dir=args.modelDir,
                    padding=padding, png=args.png, scale=args.scale,
                    crf=args.crf, preset=args.preset)

    procs, segs = [], []
    for i, (fi, cop) in enumerate(chunks):
        cfg = dict(base_cfg)
        cfg['seg_path'] = os.path.abspath('./rife_seg_%d_%d.%s' % (os.getpid(), i, args.ext))
        cfg['png_start'] = (fi - 1) * args.multi
        segs.append(cfg['seg_path'])
        p = ctx.Process(target=gpu_worker, daemon=True,
                        args=(i, cfg, cop, fi, i == len(chunks) - 1, msg_q))
        p.start()
        procs.append(p)

    pbar = tqdm(total=max(1, N))
    done, err, t0 = 0, None, time.time()
    while done < len(chunks):
        try:
            m = msg_q.get(timeout=2)
        except Empty:
            if not any(p.is_alive() for p in procs):
                err = 'a gpu worker died unexpectedly'
                break
            continue
        k = m[0]
        if k == 'prog':
            pbar.update(m[1])
        elif k == 'done':
            done += 1
        elif k == 'error':
            err = m[1]
            break
    if err:
        for p in procs:
            p.terminate()
        pbar.close()
        raise RuntimeError(err)
    for p in procs:
        p.join(timeout=300)
    pbar.close()
    elapsed = time.time() - t0
    print('Interpolation: %.1f sec (%.2f input frames/sec)' % (elapsed, N / max(elapsed, 1e-6)))

    if args.png:
        print('done: vid_out/*.png')
        return

    for s in segs:
        assert os.path.exists(s) and os.path.getsize(s) > 0, 'segment missing: ' + s

    vid_out_name = args.output if args.output is not None else \
        '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(float(ofps))), args.ext)
    listf = './rife_concat_%d.txt' % os.getpid()
    with open(listf, 'w') as f:
        for s in segs:
            f.write("file '%s'\n" % os.path.abspath(s))

    ok = False
    if args.video is not None and fpsNotAssigned and audio:
        try:
            r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-nostdin', '-f', 'concat', '-safe', '0',
                                '-i', listf, '-i', args.video, '-map', '0:v:0', '-map', '1:a:0',
                                '-c', 'copy', vid_out_name], stderr=subprocess.DEVNULL)
            ok = r.returncode == 0 and os.path.getsize(vid_out_name) > 0
        except Exception:
            ok = False
        if not ok:
            try:
                r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-nostdin', '-f', 'concat', '-safe', '0',
                                    '-i', listf, '-i', args.video, '-map', '0:v:0', '-map', '1:a:0',
                                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k', vid_out_name],
                                   stderr=subprocess.DEVNULL)
                ok = r.returncode == 0 and os.path.getsize(vid_out_name) > 0
                if ok:
                    print('Lossless audio transfer failed. Audio was transcoded to AAC instead.')
            except Exception:
                ok = False
    if not ok:
        r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-nostdin', '-f', 'concat', '-safe', '0',
                            '-i', listf, '-c', 'copy', vid_out_name], stderr=subprocess.DEVNULL)
        if not (r.returncode == 0 and os.path.getsize(vid_out_name) > 0) and fpsNotAssigned and args.video is not None:
            print('Audio transfer failed. Interpolated video will have no audio')
    os.remove(listf)
    for s in segs:
        try:
            os.remove(s)
        except OSError:
            pass
    print('done:', vid_out_name)


if __name__ == '__main__':
    main()
