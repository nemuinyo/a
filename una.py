import os
import json
import shutil
import argparse
import warnings
import traceback
import subprocess
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from tqdm import tqdm
from torch.nn import functional as F

warnings.filterwarnings("ignore")
from model.pytorch_msssim import ssim_matlab


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
    parser.add_argument('--fp16', dest='fp16', action='store_true', help='(AMPはデフォルトON。互換用フラグ)')
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


# ---------------- ffmpeg I/O (CPU処理をサブプロセスへオフロード) ----------------

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
    """rawvideoをpipeで受ける。BGRのまま流すのでチャンネル反転コストなし"""
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
        if self.w_out != self.w_full:  # montage用クロップ
            frame = frame[:, self.left:self.left + self.w_out]
        return frame

    def close(self):
        try:
            self.proc.stdout.close()
            self.proc.terminate()
            self.proc.wait()
        except Exception:
            pass


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
               '-pix_fmt', 'yuv420p', '-r', str(fps), path]
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


# ---------------- GPU前処理 ----------------

class FrameLoader:
    """uint8 HWC のままpinned ring bufferで非同期H2Dし、permute/float/padは全てGPU上で実行"""
    def __init__(self, dev, h, w, padding, depth=3):
        self.dev, self.padding, self.i = dev, padding, 0
        self.pin = [torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True) for _ in range(depth)]
        self.evt = [torch.cuda.Event() for _ in range(depth)]

    def load(self, frame_np):
        i = self.i
        self.i = (i + 1) % len(self.pin)
        self.evt[i].synchronize()
        self.pin[i].copy_(torch.from_numpy(frame_np))
        gpu = self.pin[i].to(self.dev, non_blocking=True)
        self.evt[i].record()
        x = gpu.permute(2, 0, 1).unsqueeze(0).float().div_(255.)
        return F.pad(x, self.padding)


def compute_ssim(I0, I1):
    s0 = F.interpolate(I0.float(), (32, 32), mode='bilinear', align_corners=False)
    s1 = F.interpolate(I1.float(), (32, 32), mode='bilinear', align_corners=False)
    return float(ssim_matlab(s0[:, :3], s1[:, :3]))


def tensor_to_img(t, h, w):
    # 転置/クロップ/スケールをGPU側で実行し、CPUには連続なuint8 HWCのみを落とす
    x = t[0].permute(1, 2, 0)[:h, :w].float().mul(255.).byte().contiguous()
    return x.cpu().numpy()


_batch_ok = True

def make_inference(model, I0, I1, n, scale):
    """タイムステップ違いを1つのバッチforwardにまとめる(非対応モデルは自動フォールバック)"""
    global _batch_ok
    if n <= 0:
        return []
    if getattr(model, 'version', 0) >= 3.9:
        if n == 1:
            return [model.inference(I0, I1, 0.5, scale)]
        if _batch_ok:
            try:
                ts = torch.arange(1, n + 1, device=I0.device, dtype=I0.dtype).div_(n + 1).view(-1, 1, 1, 1)
                m = model.inference(I0.repeat(n, 1, 1, 1), I1.repeat(n, 1, 1, 1), ts, scale)
                return [m[i:i + 1] for i in range(n)]
            except Exception:
                _batch_ok = False
        return [model.inference(I0, I1, (i + 1) * 1. / (n + 1), scale) for i in range(n)]
    middle = model.inference(I0, I1, scale=scale)
    if n == 1:
        return [middle]
    first = make_inference(model, I0, middle, n // 2, scale)
    second = make_inference(model, middle, I1, n // 2, scale)
    if n % 2:
        return [*first, middle, *second]
    return [*first, *second]


class Slots:
    """バックプレッシャー(メモリ爆発防止)"""
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


# ---------------- パイプライン各スレッド ----------------

def worker(gid, dev, model, args, h, w, task_q, result_q):
    amp = torch.autocast('cuda', dtype=torch.float16, enabled=not args.fp32)
    while True:
        t = task_q.get()
        if t is None:
            return
        _, seq, I0, I1, base, lf = t
        try:
            I0 = I0.to(dev)
            I1 = I1.to(dev)
            with amp:
                outs = make_inference(model, I0, I1, args.multi - 1, args.scale)
            mids = []
            for o in outs:
                img = tensor_to_img(o, h, w)
                mids.append(np.concatenate((base, img), 1) if args.montage else img)
            result_q.put(('r', seq, lf, mids))
        except Exception:
            traceback.print_exc()
            result_q.put(('abort', f'worker {gid} error'))
            return


def sequencer(args, reader, model, dev, loader, task_q, result_q, slots, h, w, workers, lastframe_np):
    """フレーム読み込み・SSIM判定・スタティック処理(元コードの逐次ロジックを正確に再現)"""
    I1 = loader.load(lastframe_np)
    temp = None
    seq = 0
    err = False
    try:
        while True:
            frame = temp if temp is not None else reader.read()
            temp = None
            if frame is None:
                break
            I0 = I1
            I1 = loader.load(frame)
            ssim = compute_ssim(I0, I1)
            break_flag = False

            if ssim > 0.996:  # スタティック判定(元コードと同一)
                nxt = reader.read()
                if nxt is None:
                    break_flag = True
                    src = lastframe_np
                else:
                    temp = nxt
                    src = nxt
                I1 = loader.load(src)
                with torch.autocast('cuda', dtype=torch.float16, enabled=not args.fp32):
                    I1 = model.inference(I0, I1, scale=args.scale)
                ssim = compute_ssim(I0, I1)
                cur_img = tensor_to_img(I1, h, w)
            else:
                cur_img = frame

            base = lastframe_np
            lf = np.concatenate((base, base), 1) if args.montage else base

            if ssim < 0.2:  # シーンカット:GPUを使わず複製だけ(超高速パス)
                img0 = tensor_to_img(I0, h, w)
                mid = np.concatenate((base, img0), 1) if args.montage else img0
                slots.acquire()
                result_q.put(('r', seq, lf, [mid] * (args.multi - 1)))
            else:
                slots.acquire()
                task_q.put(('infer', seq, I0, I1, base, lf))
            seq += 1
            lastframe_np = cur_img
            if break_flag:
                break
    except Exception:
        err = True
        traceback.print_exc()
        result_q.put(('abort', 'sequencer error'))
    finally:
        for _ in workers:
            task_q.put(None)
        for t in workers:
            t.join()
        if not err:
            slots.acquire()
            lf = np.concatenate((lastframe_np, lastframe_np), 1) if args.montage else lastframe_np
            result_q.put(('final', lf))


def writer(args, result_q, out, png_pool, pbar, slots, state):
    nxt, buf, futs, written = 0, {}, [], 0

    def write_one(img):
        nonlocal written, futs
        if args.png:
            futs.append(png_pool.submit(cv2.imwrite, 'vid_out/{:0>7d}.png'.format(written), img))
            if len(futs) > 64:
                futs = [f for f in futs if not f.done()]
        elif out is not None:
            out.write(img)
        written += 1

    while True:
        m = result_q.get()
        if m[0] == 'abort':
            state['error'] = m[1]
            slots.abort()
            return
        if m[0] == 'final':
            slots.release()
            write_one(m[1])
            while nxt in buf:
                lf, mids = buf.pop(nxt)
                write_one(lf)
                for im in mids:
                    write_one(im)
                pbar.update(1)
                nxt += 1
            return
        _, seq, lf, mids = m
        slots.release()
        buf[seq] = (lf, mids)
        while nxt in buf:
            lf, mids = buf.pop(nxt)
            write_one(lf)
            for im in mids:
                write_one(im)
            pbar.update(1)
            nxt += 1


def build_model(model_dir, dev):
    from train_log.RIFE_HDv3 import Model
    m = Model()
    if not hasattr(m, 'version'):
        m.version = 0
    m.load_model(model_dir, -1)
    print("Loaded 3.x/4.x HD model.")
    m.eval()
    for name in list(vars(m)):  # 全サブネットを指定GPUへ(Model側の実装差異に対応)
        o = getattr(m, name)
        if isinstance(o, torch.nn.Module):
            o.to(dev)
    return m


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

    assert torch.cuda.is_available(), "GPUが必要です"
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    ngpu = args.ngpu if args.ngpu > 0 else min(2, torch.cuda.device_count())
    dev0 = torch.device('cuda:0')

    models = [build_model(args.modelDir, dev0)]
    for g in range(1, ngpu):
        models.append(build_model(args.modelDir, torch.device(f'cuda:{g}')))

    if args.video is not None:
        w_full, h, fps, tot_frame = ffprobe_meta(args.video)
        if args.fps is None:
            fpsNotAssigned = True
            args.fps = fps * args.multi
        else:
            fpsNotAssigned = False
        video_path_wo_ext, ext = os.path.splitext(args.video)
        print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(
            video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
        if args.png == False and fpsNotAssigned == True:
            print("The audio will be merged after interpolation process")
        else:
            print("Will not merge audio because using png or fps flag!")
        left = 0
        if args.montage:
            left = w_full // 4
        w = w_full // 2 if args.montage else w_full
        reader = FFmpegReader(args.video, h, w_full, left, w)
    else:
        files = [f for f in os.listdir(args.img) if 'png' in f]
        files.sort(key=lambda x: int(x[:-4]))
        f0 = cv2.imread(os.path.join(args.img, files[0]))
        assert f0 is not None
        tot_frame = len(files)
        fpsNotAssigned = False
        left = 0
        if args.montage:
            left = f0.shape[1] // 4
        w = f0.shape[1] // 2 if args.montage else f0.shape[1]
        h = f0.shape[0]
        reader = ImageReader(args.img, files, left, w)
        print('image sequence: {} frames, {}x{}'.format(tot_frame, w, h))

    lastframe_np = reader.read()
    tmp = max(128, int(128 / args.scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    vid_out_name = None
    out = None
    if args.png:
        os.makedirs('vid_out', exist_ok=True)
    else:
        vid_out_name = args.output if args.output is not None else \
            '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(args.fps)), args.ext)
        if args.cv_writer or shutil.which('ffmpeg') is None:
            out = CVWriter(vid_out_name, args.fps, w, h)
        else:
            out = FFmpegWriter(vid_out_name, args.fps, w, h, args.crf, args.preset)

    pbar = tqdm(total=max(1, tot_frame))
    task_q = Queue(maxsize=8)
    result_q = Queue()
    slots = Slots(2 * ngpu + 4)
    state = {'error': None}
    png_pool = ThreadPoolExecutor(max_workers=4) if args.png else None

    workers = []
    for g in range(ngpu):
        t = threading.Thread(target=worker, args=(g, torch.device(f'cuda:{g}'),
                                                  models[g], args, h, w, task_q, result_q), daemon=True)
        t.start()
        workers.append(t)

    seq_thread = threading.Thread(
        target=sequencer,
        args=(args, reader, models[0], dev0, FrameLoader(dev0, h, w, padding),
              task_q, result_q, slots, h, w, workers, lastframe_np), daemon=True)
    seq_thread.start()
    wt = threading.Thread(target=writer, args=(args, result_q, out, png_pool, pbar, slots, state), daemon=True)
    wt.start()

    seq_thread.join()
    wt.join()
    if state['error']:
        raise RuntimeError(state['error'])
    if png_pool:
        png_pool.shutdown(wait=True)
    if out is not None:
        out.close()
    reader.close()
    pbar.close()

    if args.png == False and fpsNotAssigned == True and args.video is not None:
        try:
            transferAudio(args.video, vid_out_name)
        except Exception:
            print("Audio transfer failed. Interpolated video will have no audio")
            targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
            os.rename(targetNoAudio, vid_out_name)


if __name__ == '__main__':
    main()