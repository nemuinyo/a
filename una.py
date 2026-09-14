import os
import cv2
import torch
import argparse
import numpy as np
import subprocess
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
from queue import Queue
from model.pytorch_msssim import ssim_matlab
warnings.filterwarnings("ignore")

# ---------- ★変更1: skvideo.vreader互換のRGB24リーダー ----------
class FFmpegReader:
    def __init__(self, path, h, w):
        self.h, self.w, self.fb = h, w, h * w * 3
        self.p = subprocess.Popen(
            ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '2', '-i', path,
             '-map', '0:v:0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        n = self.fb
        buf = bytearray(n)
        v = memoryview(buf)
        got = 0
        while got < n:
            r = self.p.stdout.readinto(v[got:])
            if not r:
                self._done = True
                raise StopIteration
            got += r
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 3)

    def close(self):
        try:
            self.p.stdout.close(); self.p.terminate(); self.p.wait()
        except Exception:
            pass

# ---------- ★変更2: cv2.VideoWriter互換のlibx264/NVENCライター ----------
class FFmpegOut:
    def __init__(self, path, fps, w, h, crf=18, preset='veryfast', nvenc=False):
        cmd = ['ffmpeg', '-v', 'error', '-y', '-nostdin',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (w, h),
               '-framerate', repr(float(fps)), '-i', '-',
               '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-r', repr(float(fps))]
        if nvenc:
            cmd += ['-c:v', 'h264_nvenc', '-preset', 'p4', '-rc', 'vbr',
                    '-cq', str(crf), '-b:v', '0', '-pix_fmt', 'yuv420p']
        else:
            cmd += ['-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
                    '-pix_fmt', 'yuv420p', '-threads', '2']
        cmd.append(path)
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    def write(self, img):
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img)
        self.p.stdin.write(img.data)
    def release(self):
        try:
            self.p.stdin.close(); self.p.wait()
        except Exception:
            pass

def transferAudio(sourceVideo, targetVideo):
    import shutil
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
        if (os.path.getsize(targetVideo) == 0):
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)
    shutil.rmtree("temp")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
parser.add_argument('--multi', dest='multi', type=int, default=2)
parser.add_argument('--crf', dest='crf', type=int, default=18, help='encoder quality (lower=better)')
parser.add_argument('--preset', dest='preset', type=str, default='veryfast', help='x264 preset')
parser.add_argument('--nvenc', dest='nvenc', action='store_true', help='use NVENC hardware encoder')

args = parser.parse_args()
if args.exp != 1:
    args.multi = (2 ** args.exp)
assert (not args.video is None or not args.img is None)
if args.skip:
    print("skip flag is abandoned, please refer to issue #207.")
if args.UHD and args.scale==1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if not args.img is None:
    args.png = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if(args.fp16):
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

from train_log.RIFE_HDv3 import Model
model = Model()
if not hasattr(model, 'version'):
    model.version = 0
model.load_model(args.modelDir, -1)
print("Loaded 3.x/4.x HD model.")
model.eval()
model.device()

if not args.video is None:
    videoCapture = cv2.VideoCapture(args.video)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    vw = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoCapture.release()
    if args.fps is None:
        fpsNotAssigned = True
        args.fps = fps * args.multi
    else:
        fpsNotAssigned = False
    videogen = FFmpegReader(args.video, vh, vw)   # ★変更1
    lastframe = next(videogen)
    video_path_wo_ext, ext = os.path.splitext(args.video)
    print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
    if args.png == False and fpsNotAssigned == True:
        print("The audio will be merged after interpolation process")
    else:
        print("Will not merge audio because using png or fps flag!")
else:
    files = [f for f in os.listdir(args.img) if 'png' in f]
    tot_frame = len(files)
    files.sort(key=lambda x: int(x[:-4]))
    def _img_iter():
        for f in files[1:]:
            yield cv2.imread(os.path.join(args.img, f), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = _img_iter()
    lastframe = cv2.imread(os.path.join(args.img, files[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
h, w, _ = lastframe.shape
vid_out_name = None
vid_out = None
if args.png:
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')

def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item[:, :, ::-1])   # RGB→BGRしてライターへ(元と同じ呼び方)

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
            if not args.img is None and isinstance(frame, str):
                frame = cv2.imread(os.path.join(args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            if user_args.montage:
                frame = frame[:, left: left + w]
            read_buffer.put(frame)
    except Exception:
        pass
    read_buffer.put(None)

def make_inference(I0, I1, n):    
    global model
    if model.version >= 3.9:
        res = []
        for i in range(n):
            res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), args.scale))
        return res
    else:
        middle = model.inference(I0, I1, args.scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

def pad_image(img):
    if(args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

# ---------- ★変更3: 転置/float化をGPUへ(数値は元と完全等価) ----------
def to_tensor(frame):
    x = torch.from_numpy(frame).to(device, non_blocking=True).permute(2, 0, 1).unsqueeze(0).float() / 255.
    return pad_image(x)

if args.montage:
    left = w // 4
    w = w // 2
tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)

# ★変更2: ライター生成(montage調整後のサイズで作るため元コードより後ろへ移動)
if not args.png:
    if args.output is not None:
        vid_out_name = args.output
    else:
        vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(args.fps)), args.ext)
    out_w = w * 2 if args.montage else w
    vid_out = FFmpegOut(vid_out_name, args.fps, out_w, h, args.crf, args.preset, args.nvenc)

pbar = tqdm(total=tot_frame)
if args.montage:
    lastframe = lastframe[:, left: left + w]
write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

I1 = to_tensor(lastframe)
temp = None # save lastframe when processing static frame

while True:
    if temp is not None:
        frame = temp
        temp = None
    else:
        frame = read_buffer.get()
    if frame is None:
        break
    I0 = I1
    I1 = to_tensor(frame)
    I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
    I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

    break_flag = False
    if ssim > 0.996:
        frame = read_buffer.get() # read a new frame
        if frame is None:
            break_flag = True
            frame = lastframe
        else:
            temp = frame
        I1 = to_tensor(frame)
        I1 = model.inference(I0, I1, scale=args.scale)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        
    if ssim < 0.2:
        output = []
        for i in range(args.multi - 1):
            output.append(I0)
    else:
        output = make_inference(I0, I1, args.multi - 1)

    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
    else:
        write_buffer.put(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid[:h, :w])
    pbar.update(1)
    lastframe = frame
    if break_flag:
        break

if args.montage:
    write_buffer.put(np.concatenate((lastframe, lastframe), 1))
else:
    write_buffer.put(lastframe)
write_buffer.put(None)

import time
while(not write_buffer.empty()):
    time.sleep(0.1)
time.sleep(0.2)
pbar.close()
if not vid_out is None:
    vid_out.release()
if not args.video is None and not args.png:
    videogen.close()   # ★変更1: デコーダ後始末

# move audio to new video file if appropriate
if args.png == False and fpsNotAssigned == True and not args.video is None:
    try:
        transferAudio(args.video, vid_out_name)
    except:
        print("Audio transfer failed. Interpolated video will have no audio")
        targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        os.rename(targetNoAudio, vid_out_name)
