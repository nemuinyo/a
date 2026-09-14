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

cv2.setNumThreads(2)


# ---------------- メタ / 検証 / HW検出 ----------------

def ffprobe_meta(path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,r_frame_rate,pix_fmt', '-of', 'json', path]
    st = json.loads(subprocess.check_output(cmd).decode())['streams'][0]
    num, den = st['r_frame_rate'].split('/')
    fps = Fraction(int(num), int(den))
    try:
        a = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'a',
                                     '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', path]).decode().strip()
        audio = len(a) > 0
    except Exception:
        audio = False
    return int(st['width']), int(st['height']), fps, audio, st.get('pix_fmt', 'yuv420p')


def video_frame_count(path):
    """出力検証用。count_packets(高速)→nb_framesの順で試す"""
    try:
        out = subprocess.check_output(['ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v:0',
                                       '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0',
                                       path]).decode().strip()
        v = int(out.splitlines()[0])
        if v > 0:
            return v
    except Exception:
        pass
    try:
        out = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                       '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0',
                                       path]).decode().strip()
        return int(out.splitlines()[0])
    except Exception:
        return -1


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


# ---------------- フレーム読み込み ----------------

class VideoFrames:
    """trim=start_frame でフレーム番号指定。※-hwaccel cudaは10bit等でクロマ破損を起こすため
       利用は8bit yuv420pかつ明示opt-in時のみ(デフォルトはCPUデコ)"""
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
            self.proc.wait()
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


class LegacyReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)

    def __iter__(self):
        while True:
            ok, f = self.cap.read()
            if not ok:
                return
            yield f

    def close(self):
        self.cap.release()


class LegacyWriter:
    def __init__(self, path, fps, w, h):
        self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), float(fps), (w, h))

    def write(self, img):
        self.vw.write(img)

    def close(self):
        self.vw.release()


class RawWriter:
    """ffmpegパイプ + libx264(デフォルト)/NVENC(opt-in)"""
    def __init__(self, path, ofps_frac, w, h, crf, preset, nvenc):
        fn, fd = ofps_frac
        fr = '%d/%d' % (fn, fd)
        cmd = ['ffmpeg', '-v', 'error', '-y', '-nostdin',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (w, h),
               '-framerate', fr, '-i', '-', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-r', fr]
        if nvenc:
            cmd += ['-c:v', 'h264_nvenc', '-preset', 'p4', '-rc', 'vbr',
                    '-cq', str(crf), '-b:v', '0', '-pix_fmt', 'yuv420p']
        else:
            cmd += ['-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
                    '-pix_fmt', 'yuv420p', '-threads', '2']
        cmd += [path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write(self, img):
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img)
        self.proc.stdin.write(img.data)

    def close(self):
        try:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError('encoder failed: ' + ' '.join(self.proc.args[:8]) + '...')
        except BrokenPipeError:
            raise RuntimeError('encoder died (broken pipe)')


# ---------------- 推論ユーティリティ ----------------

def make_inference(model, I0, I1, n, scale, ver):
    if n <= 0:
        return []
    if ver >= 3.9:
        return [model.inference(I0, I1, (i + 1) * 1. / (n + 1), scale) for i in range(n)]
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


def load_model(model_dir, dev):
    from train_log.RIFE_HDv3 import Model
    m = Model()
    if not hasattr(m, 'version'):
        m.version = 0
    m.load_model(model_dir, -1)
    print('Loaded 3.x/4.x HD model.')
    m.eval()
    for name in list(vars(m)):
        o = getattr(m, name)
        if isinstance(o, torch.nn.Module):
            o.to(dev)
    ver = getattr(m, 'version', 0)
    if ver < 3.9:
        try:
            if 'timestep' in inspect.signature(m.inference).parameters:
                ver = 3.9
        except Exception:
            pass
    return m, ver


def probe_scale(model, ver, ph, pw, dev, start):
    a = torch.zeros(1, 3, ph, pw, device=dev)
    b = torch.zeros_like(a)
    sc = start
    while True:
        try:
            with torch.autocast('cuda', enabled=False):
                infer_mid(model, a, b, sc, ver)
            del a, b
            torch.cuda.empty_cache()
            return sc
        except Exception as e:
            if not is_oom(e) or sc <= 0.2501:
                raise
            sc = max(0.25, sc / 2)
            torch.cuda.empty_cache()


# ---------------- シングルパス(デフォルト: 元コードの状態機械をそのまま) ----------------

def run_single(dev, model, ver, base, pbar):
    h, w = base['h'], base['w']
    padding = tuple(base['padding'])
    multi = base['multi']
    amp_ok = base['amp_mode'] == 'on'
    if amp_ok:
        print('AMP fp16 ON (--fp16): 実験用。高モーションで崩れる可能性あり')

    pin = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)

    def load(u8):
        np.copyto(pin.numpy(), np.ascontiguousarray(u8))
        x = pin.to(dev)
        x = x.permute(2, 0, 1)[[2, 1, 0]].unsqueeze(0).float().div_(255.)  # BGR→RGB
        return F.pad(x, padding)

    def img_from(t):
        x = t[0, :, :h, :w].permute(1, 2, 0).float().mul_(255.).clamp_(0, 255).byte().flip(2)  # RGB→BGR
        return x.contiguous().cpu().numpy()

    def run_oom_safe(fn):
        s = base['scale']
        while True:
            try:
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp_ok):
                    return fn(s)
            except Exception as e:
                if not is_oom(e) or s <= 0.2501:
                    raise
                s = max(0.25, s / 2)
                torch.cuda.empty_cache()
                print('OOM -> scale=%g' % s)

    if base['mode'] == 'video':
        if base['legacy_io']:
            reader = LegacyReader(base['video_path'])
        else:
            reader = VideoFrames(base['video_path'], h, base['w_full'], 0, base['nvdec'])
    else:
        reader = ImageFrames(base['img_paths'])
    it = iter(reader)

    def read_frame():
        """★修正: 終端でNoneを返す(StopIterationクラッシュ対策)"""
        fr = next(it, None)
        if fr is not None and base['left']:
            fr = fr[:, base['left']:base['left'] + w]
        return fr

    writer = None
    png_i = 0
    if base['png']:
        os.makedirs('vid_out', exist_ok=True)
    else:
        if base['legacy_io']:
            writer = LegacyWriter(base['tmp_out'], float(base['ofps_frac'][0]), base['ow'], h)
        else:
            writer = RawWriter(base['tmp_out'], base['ofps_frac'], base['ow'], h,
                               base['crf'], base['preset'], base['nvenc'])

    written = 0
    frames_read = 0

    def emit(img):
        nonlocal written, png_i
        if base['png']:
            cv2.imwrite('vid_out/%07d.png' % png_i, img)
            png_i += 1
        else:
            writer.write(img)
        written += 1

    try:
        lastframe = read_frame()
        frames_read += 1
        I1 = load(lastframe)
        temp = None

        while True:
            if temp is not None:
                frame = temp
                temp = None
            else:
                frame = read_frame()
                if frame is None:
                    break
            frames_read += 1
            I0 = I1
            I1 = load(frame)
            ssim = gpu_ssim(I0, I1)

            break_flag = False
            if ssim > 0.996:
                nxt = read_frame()
                if nxt is None:
                    break_flag = True
                    src = lastframe  # 原版準拠: 読めなければ最終フレームで代替
                else:
                    frames_read += 1
                    temp = nxt
                    src = nxt
                I1 = load(src)
                I1 = run_oom_safe(lambda s: infer_mid(model, I0, I1, s, ver))
                ssim = gpu_ssim(I0, I1)
                frame = img_from(I1)

            if ssim < 0.2:
                mids = [I0] * (multi - 1)
            else:
                mids = run_oom_safe(lambda s: make_inference(model, I0, I1, multi - 1, s, ver))

            if base['montage']:
                emit(np.concatenate((lastframe, lastframe), 1))
                for mid in mids:
                    emit(np.concatenate((lastframe, img_from(mid)), 1))
            else:
                emit(lastframe)
                for mid in mids:
                    emit(img_from(mid))
            pbar.update(1)
            lastframe = frame
            if break_flag:
                break

        if base['montage']:
            emit(np.concatenate((lastframe, lastframe), 1))
        else:
            emit(lastframe)
        pbar.update(1)
    finally:
        if writer is not None:
            writer.close()
        reader.close()

    print('frames read: %d, frames written: %d' % (frames_read, written))
    return written


# ---------------- fast(2GPUチャンク) ----------------

def _seg_writer(outq, cfg, msg_q):
    try:
        fn, fd = cfg['ofps_frac']
        fr = '%d/%d' % (fn, fd)
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
            raise RuntimeError('encoder failed')
    except Exception:
        traceback.print_exc()
        msg_q.put(('error', 'segment writer failed'))


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

        model, ver = load_model(base_cfg['model_dir'], dev)
        print(tag, 'model version:', ver)

        h, w = base_cfg['h'], base_cfg['w']
        padding = tuple(base_cfg['padding'])
        montage, multi = base_cfg['montage'], base_cfg['multi']
        pin = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)

        def load(u8):
            np.copyto(pin.numpy(), np.ascontiguousarray(u8))
            x = pin.to(dev)
            x = x.permute(2, 0, 1)[[2, 1, 0]].unsqueeze(0).float().div_(255.)
            return F.pad(x, padding)

        def img_from(t):
            x = t[0, :, :h, :w].permute(1, 2, 0).float().mul_(255.).clamp_(0, 255).byte().flip(2)
            return x.contiguous().cpu().numpy()

        sc = probe_scale(model, ver, h + padding[3], w + padding[1], dev, base_cfg['scale'])
        amp_ok = base_cfg['amp_mode'] == 'on'
        print(tag, 'ready (scale=%g, amp=%s)' % (sc, amp_ok))
        msg_q.put(('ready', gid))

        def ac():
            return torch.autocast('cuda', dtype=torch.float16, enabled=amp_ok)

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

        while True:
            ch = chunk_q.get()
            if ch is None:
                msg_q.put(('done', gid))
                return
            t0 = time.time()
            segcfg = dict(base_cfg)
            segcfg['seg_path'] = ch['seg']
            outq = Queue(maxsize=64)
            wt = threading.Thread(target=_seg_writer, args=(outq, segcfg, msg_q), daemon=True)
            wt.start()

            reader = VideoFrames(base_cfg['video_path'], h, base_cfg['w_full'],
                                 ch['first_iter'] - 1, base_cfg['nvdec'])
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
                fk_img = pending if pending is not None else next_crop()
                pending = None
                li = state_img
                outq.put(np.concatenate((li, li), 1) if montage else li)

                if kind == 'dup':
                    mids = [li] * (multi - 1)
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
                        nxt = next_crop()
                        src = nxt
                    else:
                        src = state_img
                    Fsrc = load(src)
                    D = run_oom_safe(lambda s: infer_mid(model, Fc, Fsrc, s, ver))
                    if gpu_ssim(Fc, D) < 0.2:
                        mids = [li] * (multi - 1)
                    else:
                        outs = run_oom_safe(lambda s: make_inference(model, Fc, D, multi - 1, s, ver))
                        mids = [img_from(o) for o in outs]
                    state_img = img_from(D)
                    Fc = D
                    if kind == 'static' and i < M - 1:
                        pending = nxt
                else:
                    raise RuntimeError('unknown op: ' + kind)

                for m in mids:
                    outq.put(np.concatenate((li, m), 1) if montage else m)
                msg_q.put(('prog', 1))
                i += 1

            if ch['is_last']:
                outq.put(np.concatenate((state_img, state_img), 1) if montage else state_img)
                msg_q.put(('prog', 1))
            outq.put(None)
            wt.join(timeout=300)
            reader.close()
            msg_q.put(('segdone', ch['cid'], gid, time.time() - t0))
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


def split_chunks(ops, nsplit, min_ops=4):
    M = len(ops)
    if nsplit <= 1 or M == 0:
        return [(1, ops)]
    wmap = {'pair': 1.0, 'dup': 0.05, 'static': 2.0, 'laststatic': 2.0}
    target = max(sum(wmap[o] for o in ops), 1e-9) / nsplit
    cuts, acc, nxt_t, start = [], 0.0, target, 0
    for i in range(M - 1):
        acc += wmap[ops[i]]
        remain = nsplit - len(cuts)
        if (ops[i] in ('pair', 'dup') and acc >= nxt_t
                and (M - (i + 1)) >= (remain - 1)
                and (i + 1 - start) >= min_ops):
            cuts.append(i + 1)
            start = i + 1
            nxt_t += target
    bounds = [0] + cuts + [M]
    return [(bounds[c] + 1, ops[bounds[c]:bounds[c + 1]]) for c in range(len(bounds) - 1)
            if bounds[c] < bounds[c + 1]]


def mux_audio(tmp_video, src, out_name, do_audio):
    if not do_audio:
        os.replace(tmp_video, out_name)
        return
    r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-nostdin', '-i', tmp_video, '-i', src,
                        '-map', '0:v:0', '-map', '1:a:0', '-c', 'copy',
                        '-movflags', '+faststart', out_name], stderr=subprocess.DEVNULL)
    if r.returncode == 0 and os.path.exists(out_name) and os.path.getsize(out_name) > 0:
        os.remove(tmp_video)
        return
    r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-nostdin', '-i', tmp_video, '-i', src,
                        '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k',
                        '-movflags', '+faststart', out_name], stderr=subprocess.DEVNULL)
    if r.returncode == 0 and os.path.exists(out_name) and os.path.getsize(out_name) > 0:
        os.remove(tmp_video)
        print('Lossless audio transfer failed. Audio was transcoded to AAC instead.')
        return
    print('Audio transfer failed. Interpolated video will have no audio')
    os.replace(tmp_video, out_name)


def parse_args():
    p = argparse.ArgumentParser(description='RIFE interpolation (robust build)')
    p.add_argument('--video', dest='video', type=str, default=None)
    p.add_argument('--output', dest='output', type=str, default=None)
    p.add_argument('--img', dest='img', type=str, default=None)
    p.add_argument('--montage', action='store_true')
    p.add_argument('--model', dest='modelDir', type=str, default='train_log')
    p.add_argument('--fp16', action='store_true', help='fp16 autocast(実験用・デフォルトOFF)')
    p.add_argument('--UHD', action='store_true')
    p.add_argument('--scale', dest='scale', type=float, default=1.0)
    p.add_argument('--fps', dest='fps', type=int, default=None)
    p.add_argument('--png', action='store_true')
    p.add_argument('--ext', dest='ext', type=str, default='mp4')
    p.add_argument('--exp', dest='exp', type=int, default=1)
    p.add_argument('--multi', dest='multi', type=int, default=2)
    p.add_argument('--crf', dest='crf', type=int, default=18)
    p.add_argument('--preset', dest='preset', type=str, default='veryfast')
    p.add_argument('--fast', action='store_true', help='2GPUチャンク並列')
    p.add_argument('--nvenc', action='store_true', help='HWｴﾝｺｰﾄﾞopt-in')
    p.add_argument('--nvdec', action='store_true', help='HWﾃﾞｺｰﾄﾞopt-in(8bit yuv420pのみ安全)')
    p.add_argument('--legacy-io', dest='legacy_io', action='store_true', help='cv2入出力(原版リファレンス)')
    p.add_argument('--split', dest='split', type=int, default=16)
    return p.parse_args()


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

    assert torch.cuda.is_available(), 'GPU required'
    dev = torch.device('cuda:0')
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    use_fast = args.fast and (args.video is not None) and (not args.png) and (not args.legacy_io)

    files = None
    fpsNotAssigned = False
    video_path_wo_ext = None
    pix_fmt = 'yuv420p'
    if args.video is not None:
        assert shutil.which('ffmpeg') or args.legacy_io, 'ffmpeg が必要です(legacy-io以外)'
        w_full, h, fps, audio, pix_fmt = ffprobe_meta(args.video)
        ofps = fps * args.multi
        if args.fps is None:
            fpsNotAssigned = True
        else:
            ofps = Fraction(args.fps, 1)
        video_path_wo_ext, _ = os.path.splitext(args.video)
        N = None
        print('input: {}, {}FPS -> {}FPS, pix_fmt={}'.format(args.video, float(fps), float(ofps), pix_fmt))
        print('The audio will be merged after interpolation process' if (not args.png and fpsNotAssigned)
              else 'Will not merge audio because using png or fps flag!')
    else:
        files = sorted([f for f in os.listdir(args.img) if 'png' in f], key=lambda x: int(x[:-4]))
        f0 = cv2.imread(os.path.join(args.img, files[0]), cv2.IMREAD_COLOR)
        assert f0 is not None
        h, w_full = f0.shape[:2]
        N = len(files)
        ofps, fps, audio = None, None, False
        print('image sequence: {} frames'.format(N))

    # ★ NVDEC安全ガード: 10bit/4:4:4等(白黒ノイズ破綻の温床)では自動無効化
    nvdec_ok_fmt = pix_fmt in ('yuv420p', 'yuvj420p', 'gray')
    if args.nvdec and args.video is not None and not nvdec_ok_fmt and not args.legacy_io:
        print('WARN: pix_fmt=%s はNVDECでクロマ破損の恐れがあるためCPUデコードに切り替え' % pix_fmt)
    nvenc = args.nvenc and (not args.png) and probe_nvenc()
    nvdec = (args.nvdec and (args.video is not None) and (not args.legacy_io)
             and nvdec_ok_fmt and probe_nvdec())
    if (args.nvenc or args.nvdec) and not (nvenc or nvdec):
        print('WARN: HW accel requested but unavailable -> software fallback')

    left = (w_full // 4) if args.montage else 0
    w = (w_full // 2) if args.montage else w_full
    ow = w * 2 if args.montage else w

    tmpsz = max(128, int(128 / args.scale))
    ph = ((h - 1) // tmpsz + 1) * tmpsz
    pw = ((w - 1) // tmpsz + 1) * tmpsz
    padding = (0, pw - w, 0, ph - h)

    vid_out_name = args.output if args.output is not None else \
        '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi,
                                 int(np.round(float(ofps))) if ofps else 0, args.ext)
    tmp_out = vid_out_name + '.tmp.' + args.ext

    base = dict(mode='video' if args.video is not None else 'img',
                video_path=args.video,
                img_paths=[os.path.join(args.img, f) for f in files] if files else None,
                ofps_frac=(ofps.numerator, ofps.denominator) if ofps else (30, 1),
                h=h, w=w, w_full=w_full, left=left, ow=ow, montage=args.montage,
                multi=args.multi, model_dir=args.modelDir,
                amp_mode=('on' if args.fp16 else 'off'),
                padding=padding, png=args.png, scale=args.scale,
                crf=args.crf, preset=args.preset,
                nvenc=nvenc, nvdec=nvdec, legacy_io=args.legacy_io, tmp_out=tmp_out)

    model, ver = load_model(args.modelDir, dev)
    print('model version:', ver, '| hw: nvenc={} nvdec={}'.format(nvenc, nvdec),
          '| io:', 'legacy-cv2' if args.legacy_io else ('ffmpeg' if shutil.which('ffmpeg') else 'none'))

    t0 = time.time()
    segs = None
    expected = None
    if use_fast:
        frames32 = read_proxy32(args.video)
        N = len(frames32)
        print('input frames: {}'.format(N))
        ops = build_ops(frames32, args.multi)
        parts = split_chunks(ops, max(1, min(args.split, len(ops))))
        chunks = [dict(cid=cid, first_iter=fi, ops=cop, is_last=(cid == len(parts) - 1),
                       seg=os.path.abspath('./rife_seg_%d_%d.%s' % (os.getpid(), cid, args.ext)))
                  for cid, (fi, cop) in enumerate(parts)]
        segs = [c['seg'] for c in chunks]
        for c in chunks:
            print('  chunk %d: iters %d..%d (%d ops)' % (c['cid'], c['first_iter'],
                                                         c['first_iter'] + len(c['ops']) - 1, len(c['ops'])))
        ctx = mp.get_context('spawn')
        msg_q, chunk_q = ctx.Queue(), ctx.Queue()
        procs = []
        for g in range(min(2, torch.cuda.device_count())):
            p = ctx.Process(target=gpu_worker, daemon=True, args=(g, base, chunk_q, msg_q))
            p.start()
            procs.append(p)
        for c in chunks:
            chunk_q.put(c)
        for _ in procs:
            chunk_q.put(None)

        pbar = tqdm(total=max(1, N), desc='fast')
        pending = set(c['cid'] for c in chunks)
        err = None
        while True:
            try:
                m = msg_q.get(timeout=3)
            except Empty:
                alive = sum(1 for p in procs if p.is_alive())
                if pending and alive == 0:
                    err = 'gpu worker died'
                    break
                if not pending and alive == 0:
                    break
                continue
            if m[0] == 'prog':
                pbar.update(m[1])
            elif m[0] == 'segdone':
                pending.discard(m[1])
            elif m[0] == 'error':
                err = m[1]
                break
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=30)
        pbar.close()
        if err:
            raise RuntimeError(err + ' (segments kept: ' + ', '.join(segs) + ')')
        for c, s in zip(chunks, segs):
            exp = len(c['ops']) * args.multi + (1 if c['is_last'] else 0)
            got = video_frame_count(s)
            print('  seg %d: frames=%d (expected %d)' % (c['cid'], got, exp))
            assert got == exp, 'segment %d frame mismatch (%d != %d)' % (c['cid'], got, exp)
        with open('./rife_concat_%d.txt' % os.getpid(), 'w') as f:
            for s in segs:
                f.write("file '%s'\n" % s)
        r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-nostdin', '-f', 'concat', '-safe', '0',
                            '-i', './rife_concat_%d.txt' % os.getpid(), '-c', 'copy',
                            '-movflags', '+faststart', tmp_out], stderr=subprocess.DEVNULL)
        assert r.returncode == 0, 'concat failed'
        os.remove('./rife_concat_%d.txt' % os.getpid())
        expected = (N - 1) * args.multi + 1
    else:
        pbar = tqdm(total=None, desc='single')
        written = run_single(dev, model, ver, base, pbar)
        pbar.close()
        expected = written  # 実測emit数が正

    # ---- 出力検証: 実測書き込み数とエンコード結果が一致しなければ完成形を出さない ----
    got = video_frame_count(tmp_out)
    print('output frames: %d (expected %d)' % (got, expected))
    if expected > 0 and got != expected:
        raise RuntimeError('FRAME COUNT MISMATCH: %d != %d (kept %s for inspection)' % (got, expected, tmp_out))

    print('Interpolation: %.1f sec' % (time.time() - t0))

    if args.png:
        print('done: vid_out/*.png')
        return

    mux_audio(tmp_out, args.video, vid_out_name,
              do_audio=(args.video is not None and fpsNotAssigned and audio))
    if segs:
        for s in segs:
            try:
                os.remove(s)
            except OSError:
                pass
    print('done:', vid_out_name)


if __name__ == '__main__':
    main()
