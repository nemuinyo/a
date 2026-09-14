import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import shutil
import argparse
import warnings
import traceback
import subprocess
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from queue import Empty

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore")
from model.pytorch_msssim import ssim_matlab

cv2.setNumThreads(2)


def transferAudio(sourceVideo, targetVideo):
    tempAudioFileName = "./temp/audio.mkv"
    if os.path.isdir("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")
    os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))
    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
    if os.path.getsize(targetVideo) == 0:
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if os.path.getsize(targetVideo) == 0:
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)
    shutil.rmtree("temp")


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
    parser.add_argument('--cv-writer', dest='cv_writer', action='store_true', help='旧cv2(mp4v)ライタを使う')
    return parser.parse_args()


# ---------------- ffmpeg / cv2 I/O ----------------

def ffprobe_meta(path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,r_frame_rate,nb_frames', '-of', 'json', path]
    st = json.loads(subprocess.check_output(cmd).decode())['streams'][0]
    num, den = st['r_frame_rate'].split('/')
    fps = float(num) / float(den)
    tot = int(st.get('nb_frames') or 0)
    if tot <= 0:
        try:
            dur = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                                                 'format=duration', '-of', 'default=nw=1:nk=1', path]).decode().strip())
            tot = max(0, int(round(dur * fps)))
        except Exception:
            tot = 0
    return st['width'], st['height'], fps, tot


class FFmpegReader:
    def __init__(self, path, h, w_full, left, w_out):
        self.h, self.w_full, self.left, self.w_out = h, w_full, left, w_out
        self.frame_bytes = h * w_full * 3
        self.proc = subprocess.Popen(
            ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '2', '-i', path,
             '-map', '0:v:0', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self):
        n = self.frame_bytes
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            r = self.proc.stdout.readinto(view[got:])
            if not r:
                return None
            got += r
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w_full, 3)
        if self.w_out != self.w_full:
            frame = frame[:, self.left:self.left + self.w_out]
        return frame

    def close(self):
        try:
            self.proc.stdout.close()
            self.proc.terminate()
            self.proc.wait()
        except Exception:
            pass


class CVReader:
    def __init__(self, path, left, w_out):
        self.cap = cv2.VideoCapture(path)
        self.left, self.w_out = left, w_out

    def read(self):
        ok, f = self.cap.read()
        if not ok:
            return None
        if self.w_out != f.shape[1]:
            f = f[:, self.left:self.left + self.w_out]
        return f

    def close(self):
        self.cap.release()


class ImageReader:
    def __init__(self, folder, files, left, w_out):
        self.paths = [os.path.join(folder, f) for f in files]
        self.i, self.left, self.w_out = 0, left, w_out

    def read(self):
        if self.i >= len(self.paths):
            return None
        f = cv2.imread(self.paths[self.i])
        self.i += 1
        if f is None:
            return None
        if self.w_out != f.shape[1]:
            f = f[:, self.left:self.left + self.w_out]
        return f

    def close(self):
        pass


class FFmpegWriter:
    def __init__(self, path, fps, w, h, crf=18, preset='veryfast'):
        cmd = ['ffmpeg', '-v', 'error', '-y', '-nostdin',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}',
               '-framerate', str(fps), '-i', '-',
               '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
               '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
               '-pix_fmt', 'yuv420p', '-threads', '2', '-r', str(fps), path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write(self, img):
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img)
        self.proc.stdin.write(img.data)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait()
        except Exception:
            pass


class CVWriter:
    def __init__(self, path, fps, w, h):
        self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), float(fps), (w, h))

    def write(self, img):
        self.vw.write(img)

    def close(self):
        self.vw.release()


# ---------------- ユーティリティ ----------------

def cpu_ssim(a_u8, b_u8):
    """SSIM判定をCPU完結(32x32なので誤差は実用上無視できる)"""
    a = cv2.resize(a_u8, (32, 32), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b_u8, (32, 32), interpolation=cv2.INTER_AREA)
    ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float().div_(255.)
    tb = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).float().div_(255.)
    return float(ssim_matlab(ta, tb))


class Slots:
    def __init__(self, n):
        self.n, self.used, self.aborted = n, 0, False
        self.cv = threading.Condition()

    def acquire(self):
        with self.cv:
            while self.used >= self.n and not self.aborted:
                self.cv.wait()
            self.used += 1

    def release(self):
        with self.cv:
            self.used -= 1
            self.cv.notify()

    def abort(self):
        with self.cv:
            self.aborted = True
            self.cv.notify_all()


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower()


def lower_scale(scale_val, s):
    with scale_val.get_lock():
        if s < scale_val.value:
            scale_val.value = s


def qget(q, procs, timeout=30):
    while True:
        try:
            return q.get(timeout=timeout)
        except Empty:
            if not any(p.is_alive() for p in procs):
                return ('abort', 'gpu worker died')


# ---------------- 推論(ワーカープロセス内) ----------------

def make_inference(model, I0, I1, n, scale, ver):
    if n <= 0:
        return []
    if ver >= 3.9:
        if n == 1:
            return [model.inference(I0, I1, 0.5, scale)]
        outs = []
        CH = 4  # バッチタイムステップ(メモリ爆発防止のため4区切り)
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
    if ver >= 3.9:
        return model.inference(I0, I1, 0.5, scale)
    return model.inference(I0, I1, scale)


def gpu_worker(gid, model_dir, n_mid, fp32, h, w, padding, task_q, res_q, static_q, scale_val, probe_evt):
    tag = f'[gpu{gid}]'
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)  # 自プロセスは物理GPU gid のみを見る
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        dev = torch.device('cuda')
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(1)

        from train_log.RIFE_HDv3 import Model
        model = Model()
        if not hasattr(model, 'version'):
            model.version = 0
        model.load_model(model_dir, -1)
        print(tag + " Loaded 3.x/4.x HD model.")
        model.eval()
        for name in list(vars(model)):
            o = getattr(model, name)
            if isinstance(o, torch.nn.Module):
                o.to(dev)
        ver = getattr(model, 'version', 0)

        pin = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)

        def to_gpu(u8):
            np.copyto(pin.numpy(), u8)
            x = pin.to(dev, non_blocking=True)
            x = x.permute(2, 0, 1).unsqueeze(0).float().div_(255.)
            return F.pad(x, padding)

        def img_from(t):
            x = t[0, :, :h, :w].permute(1, 2, 0).mul(255.).clamp_(0, 255).byte().contiguous()
            return x.cpu().numpy()

        # ---- 起動時メモリプローブ:実解像度でOOMならscaleを自動半減 ----
        ph, pw = h + padding[3], w + padding[1]
        a = torch.zeros(1, 3, ph, pw, device=dev)
        b = torch.zeros_like(a)
        scale = scale_val.value
        while True:
            try:
                with torch.autocast('cuda', dtype=torch.float16, enabled=not fp32):
                    infer_mid(model, a, b, scale, ver)
                break
            except Exception as e:
                if not is_oom(e) or scale <= 0.2501:
                    raise
                scale = max(0.25, scale / 2)
                lower_scale(scale_val, scale)
                torch.cuda.empty_cache()
        del a, b
        torch.cuda.empty_cache()
        probe_evt.set()
        print(tag + f' ready (scale={scale})')

        # ---- タスクループ ----
        while True:
            t = task_q.get()
            if t is None:
                return
            kind, s, x0, x1 = t
            I0 = to_gpu(x0)
            I1 = to_gpu(x1)
            sc = scale_val.value
            while True:
                try:
                    with torch.autocast('cuda', dtype=torch.float16, enabled=not fp32):
                        if kind == 'pair':
                            outs = make_inference(model, I0, I1, n_mid - 1, sc, ver)
                        else:
                            outs = [infer_mid(model, I0, I1, sc, ver)]
                    break
                except Exception as e:
                    if not is_oom(e) or sc <= 0.2501:
                        raise
                    sc = max(0.25, sc / 2)
                    lower_scale(scale_val, sc)
                    torch.cuda.empty_cache()
            imgs = [img_from(o) for o in outs]
            if kind == 'pair':
                res_q.put(('r', s, imgs))
            else:
                static_q.put(('sres', s, imgs[0]))
    except Exception:
        traceback.print_exc()
        try:
            probe_evt.set()
            res_q.put(('abort', f'gpu{gid} worker crashed'))
            static_q.put(('abort', f'gpu{gid} worker crashed'))
        except Exception:
            pass


# ---------------- 親プロセス側スレッド ----------------

def run_sequencer(args, reader, procs, events, scale_val, task_q, static_q, writer_q, slots, lastframe_np, state, pbar):
    try:
        for ev in events:
            ev.wait(timeout=600)
        sc = scale_val.value
        if abs(sc - args.scale) > 1e-6:
            print(f"VRAM insufficient at scale={args.scale}: auto-lowered to scale={sc}")
        temp = None
        seq = 0
        while not slots.aborted:
            frame = temp if temp is not None else reader.read()
            temp = None
            if frame is None:
                break
            I0u8 = lastframe_np
            ssim = cpu_ssim(I0u8, frame)
            break_flag = False

            if ssim > 0.996:  # 静的フレーム判定(元コードと同じロジック)
                nxt = reader.read()
                if nxt is None:
                    break_flag = True
                    src = lastframe_np
                else:
                    temp = nxt
                    src = nxt
                slots.acquire()
                if slots.aborted:
                    return
                task_q.put(('static', seq, I0u8, src))
                m = qget(static_q, procs)
                if m[0] == 'abort':
                    state['error'] = m[1]
                    slots.abort()
                    return
                mid_u8 = m[2]
                slots.release()
                ssim = cpu_ssim(I0u8, mid_u8)
                frame_out = mid_u8
            else:
                frame_out = frame

            if ssim < 0.2:  # シーンカット:GPU不要の複製パス
                writer_q.put(('full', seq, lastframe_np, [I0u8] * (args.multi - 1)))
            else:
                slots.acquire()
                if slots.aborted:
                    return
                writer_q.put(('half', seq, lastframe_np))
                task_q.put(('pair', seq, I0u8, frame_out))
            lastframe_np = frame_out
            pbar.update(1)
            seq += 1
            if break_flag:
                break
    except Exception:
        traceback.print_exc()
        state['error'] = state['error'] or 'sequencer error'
        slots.abort()
    finally:
        writer_q.put(('end',))


def run_writer(args, writer_q, res_q, out, png_pool, pbar, slots, state, procs):
    nxt = 0
    pend_lf, resb = {}, {}
    cnt = 0
    futs = []

    def emit(base, img):
        nonlocal cnt, futs
        final = np.concatenate((base, img), 1) if args.montage else img
        if args.png:
            futs.append(png_pool.submit(cv2.imwrite, 'vid_out/{:0>7d}.png'.format(cnt), final))
            if len(futs) > 64:
                futs = [f for f in futs if not f.done()]
            cnt += 1
        elif out is not None:
            out.write(final)

    def flush():
        nonlocal nxt
        while nxt in pend_lf and nxt in resb:
            lf, outs = pend_lf.pop(nxt), resb.pop(nxt)
            emit(lf, lf)
            for im in outs:
                emit(lf, im)
            nxt += 1

    def recv_res():
        nonlocal nxt
        m = res_q.get(timeout=30) if False else None
        return m

    while True:
        try:
            m = writer_q.get(timeout=30)
        except Empty:
            if not any(p.is_alive() for p in procs):
                state['error'] = state['error'] or 'gpu worker died unexpectedly'
                slots.abort()
                return
            continue
        t = m[0]
        if t == 'full':
            _, s, lf, outs = m
            pend_lf[s] = lf
            resb[s] = outs
            flush()
        elif t == 'half':
            _, s, lf = m
            pend_lf[s] = lf
            flush()
        elif t == 'r':
            _, s, outs = m
            resb[s] = outs
            slots.release()
            flush()
        elif t == 'abort':
            state['error'] = state['error'] or m[1]
            slots.abort()
        elif t == 'end':
            while pend_lf or resb:  # 残りの結果を排水してから終了
                try:
                    m2 = res_q.get(timeout=30)
                except Empty:
                    if not any(p.is_alive() for p in procs):
                        state['error'] = state['error'] or 'workers died before finishing'
                        break
                    continue
                if m2[0] == 'r':
                    resb[m2[1]] = m2[2]
                    slots.release()
                    flush()
                elif m2[0] == 'abort':
                    state['error'] = state['error'] or m2[1]
                    break
            flush()
            return


def build_main():
    pass


def main():
    args = parse_args()
    if args.exp != 1:
        args.multi = 2 ** args.exp
    assert (not args.video is None or not args.img is None)
    if args.skip:
        print("skip flag is abandoned, please refer to issue #207.")
    if args.UHD and args.scale == 1.0:
        args.scale = 0.5
    assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
    if not args.img is None:
        args.png = True

    assert torch.cuda.is_available(), "GPU required"
    ngpu = args.ngpu if args.ngpu > 0 else min(2, torch.cuda.device_count())
    ngpu = max(1, min(ngpu, torch.cuda.device_count()))
    have_ffmpeg = shutil.which('ffmpeg') is not None

    fpsNotAssigned = False
    video_path_wo_ext = None
    if args.video is not None:
        w_full, h_probe, fps, tot_frame = ffprobe_meta(args.video)
        if args.fps is None:
            fpsNotAssigned = True
            args.fps = fps * args.multi
        video_path_wo_ext, ext = os.path.splitext(args.video)
        print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
        if args.png == False and fpsNotAssigned == True:
            print("The audio will be merged after interpolation process")
        else:
            print("Will not merge audio because using png or fps flag!")
        left = w_full // 4 if args.montage else 0
        w_init = w_full // 2 if args.montage else w_full
        reader = FFmpegReader(args.video, h_probe, w_full, left, w_init) if have_ffmpeg \
            else CVReader(args.video, left, w_init)
    else:
        files = [f for f in os.listdir(args.img) if 'png' in f]
        files.sort(key=lambda x: int(x[:-4]))
        tot_frame = len(files)
        f0 = cv2.imread(os.path.join(args.img, files[0]))
        assert f0 is not None
        left = f0.shape[1] // 4 if args.montage else 0
        w_init = f0.shape[1] // 2 if args.montage else f0.shape[1]
        reader = ImageReader(args.img, files, left, w_init)
        print('image sequence: {} frames'.format(tot_frame))

    lastframe = reader.read()
    assert lastframe is not None, "cannot read the first frame"
    h, w = lastframe.shape[:2]

    tmp = max(128, int(128 / args.scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    vid_out_name = None
    out = None
    out_w = w * 2 if args.montage else w  # 元コードのmontage時の出力サイズ不具合も修正
    if args.png:
        os.makedirs('vid_out', exist_ok=True)
    else:
        vid_out_name = args.output if args.output is not None else \
            '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(args.fps)), args.ext)
        if args.cv_writer or not have_ffmpeg:
            out = CVWriter(vid_out_name, args.fps, out_w, h)
        else:
            out = FFmpegWriter(vid_out_name, args.fps, out_w, h, args.crf, args.preset)

    ctx = mp.get_context('spawn')
    scale_val = ctx.Value('d', float(args.scale))
    events = [ctx.Event() for _ in range(ngpu)]
    task_q = ctx.Queue(maxsize=8)
    res_q = ctx.Queue()
    static_q = ctx.Queue()
    writer_q = ctx.Queue()
    slots = Slots(2 * ngpu + 4)

    procs = []
    for g in range(ngpu):
        p = ctx.Process(target=gpu_worker, daemon=True,
                        args=(g, args.modelDir, args.multi, args.fp32, h, w, padding,
                              task_q, res_q, static_q, scale_val, events[g]))
        p.start()
        procs.append(p)

    state = {'error': None}
    pbar = tqdm(total=max(1, tot_frame))
    png_pool = ThreadPoolExecutor(max_workers=4) if args.png else None
    wt = threading.Thread(target=run_writer, daemon=True,
                          args=(args, writer_q, res_q, out, png_pool, pbar, slots, state, procs))
    wt.start()

    run_sequencer(args, reader, procs, events, scale_val, task_q, static_q, writer_q, slots, lastframe, state, pbar)

    for _ in procs:
        try:
            task_q.put_nowait(None)
        except Exception:
            pass
    for p in procs:
        p.join(timeout=60)
    wt.join(timeout=600)
    if png_pool:
        png_pool.shutdown(wait=True)
    if out is not None:
        out.close()
    reader.close()
    pbar.close()

    if state['error']:
        raise RuntimeError(state['error'])

    if args.png == False and fpsNotAssigned == True and args.video is not None:
        try:
            transferAudio(args.video, vid_out_name)
        except Exception:
            print("Audio transfer failed. Interpolated video will have no audio")
            targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
            os.rename(targetNoAudio, vid_out_name)


if __name__ == '__main__':
    main()
