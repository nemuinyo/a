import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab
import subprocess
import time

warnings.filterwarnings("ignore")

def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(targetVideo) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0): # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
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
    videoCapture.release()
    if args.fps is None:
        fpsNotAssigned = True
        args.fps = fps * args.multi
    else:
        fpsNotAssigned = False
    videogen = skvideo.io.vreader(args.video)
    lastframe = next(videogen)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_path_wo_ext, ext = os.path.splitext(args.video)
    print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
    if args.png == False and fpsNotAssigned == True:
        print("The audio will be merged after interpolation process")
    else:
        print("Will not merge audio because using png or fps flag!")
else:
    videogen = []
    for f in os.listdir(args.img):
        if 'png' in f:
            videogen.append(f)
    tot_frame = len(videogen)
    videogen.sort(key= lambda x:int(x[:-4]))
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = videogen[1:]
h, w, _ = lastframe.shape

# prepare output filename
if args.output is not None:
    vid_out_name = args.output
else:
    if not args.video is None:
        video_path_wo_ext = os.path.splitext(args.video)[0]
    else:
        video_path_wo_ext = "out"
    vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(args.fps)), args.ext)

# ffmpeg process (None if png mode)
ffmpeg_proc = None
if not args.png:
    # We assume frames in write_buffer are RGB (H,W,3) as in original code (transpose used)
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{w}x{h}',
        '-r', str(int(np.round(args.fps))),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        vid_out_name
    ]
    try:
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
    except Exception as e:
        print("Failed to start ffmpeg, falling back to cv2.VideoWriter. Error:", e)
        ffmpeg_proc = None

# create vid_out directory for pngs if needed
if args.png:
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')

def clear_write_buffer(user_args, write_buffer, ffmpeg_proc=None):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            # item is RGB; cv2.imwrite expects BGR
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            if ffmpeg_proc is not None and ffmpeg_proc.stdin:
                try:
                    # item is RGB (H,W,3), pix_fmt=rgb24
                    ffmpeg_proc.stdin.write(item.tobytes())
                except BrokenPipeError:
                    # ffmpeg died; stop writing
                    break
                except Exception as e:
                    print("Error writing to ffmpeg stdin:", e)
                    break
            else:
                # fallback: try to write using cv2.VideoWriter if available
                try:
                    # convert RGB to BGR for cv2
                    if 'vid_out_cv2' in globals() and vid_out_cv2 is not None:
                        vid_out_cv2.write(item[:, :, ::-1])
                except Exception:
                    break
    # close ffmpeg stdin and wait for process to finish
    if ffmpeg_proc is not None:
        try:
            if ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.close()
        except:
            pass
        try:
            ffmpeg_proc.wait()
        except:
            pass

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
            if not user_args.img is None:
                frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            if user_args.montage:
                frame = frame[:, left: left + w]
            read_buffer.put(frame)
    except:
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

if args.montage:
    left = w // 4
    w = w // 2
tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)
if args.montage:
    lastframe = lastframe[:, left: left + w]
write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)

# If ffmpeg failed to start and we are not in png mode, fallback to cv2.VideoWriter
vid_out_cv2 = None
if not args.png and ffmpeg_proc is None:
    try:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vid_out_cv2 = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))
    except Exception as e:
        print("Failed to create cv2.VideoWriter fallback:", e)
        vid_out_cv2 = None

_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer, ffmpeg_proc))

I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I1 = pad_image(I1)
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
    I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
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
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)
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

# wait until writer finishes consuming buffer
while(not write_buffer.empty()):
    time.sleep(0.1)
pbar.close()

# release cv2 fallback if used
if 'vid_out_cv2' in globals() and vid_out_cv2 is not None:
    try:
        vid_out_cv2.release()
    except:
        pass

# move audio to new video file if appropriate
if args.png == False and fpsNotAssigned == True and not args.video is None:
    try:
        transferAudio(args.video, vid_out_name)
    except Exception as e:
        print("Audio transfer failed. Interpolated video will have no audio. Error:", e)
        targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        try:
            os.rename(targetNoAudio, vid_out_name)
        except:
            pass
