import os
import sys
import subprocess

def _create_vsrife_script(
    vapoursynth_file: str,
    hentai_name: str,
    video_resolution: str = "1080p",
    video_aspect: str = "16:9"
):
    script = []

    if video_resolution == '1080p':
        video_width = "1440" if video_aspect == '4:3' else "1920"
        script = [
            'from vsrife import rife',
            'import vapoursynth as vs',
            f'clip = vs.core.ffms2.Source(source="./{hentai_name} [4k][HEVC].mkv")',
            f'clip = vs.core.resize.Bicubic(clip, width={video_width}, height=1080, format=vs.RGBS, matrix_in_s="709")',
            'clip = rife(clip=clip, model="4.25", factor_num=2, factor_den=1)',
            'clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")',
            'clip.set_output()'
        ]
    elif video_resolution == '2160p':
        video_width = "2880" if video_aspect == '4:3' else "3840"
        script = [
            'import vapoursynth as vs',
            'from vsrife import rife',
            f'clip = vs.core.ffms2.Source(source="./{hentai_name} [4k][HEVC].mkv")',
            f'clip = vs.core.resize.Bicubic(clip, width={video_width}, height=2160, format=vs.RGBS, matrix_in_s="709")',
            'clip = rife(clip=clip, model="4.25.lite", factor_num=2, factor_den=1)',
            'clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")',
            'clip.set_output()'
        ]

    with open(vapoursynth_file, 'w') as fs:
        fs.writelines([i + '\n' for i in script])

def _interpolate(
    vapoursynth_file: str,
    interpolate_output: str,
):
    cmd = [
        "vspipe", 
        "-c", "y4m",
        vapoursynth_file,
        "-", "|",
        "ffmpeg", "-v", "quiet", "-stats",
        "-i", "-",
        "-c:v", "hevc_nvenc",
        "-qp", "5",
        interpolate_output
    ]

    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nffmpeg failed with error code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

def interpolate(
    upscale_video_input: str,
    interpolate_output: str,
    hentai_name: str,
    video_resolution: str = "1080p",
    video_aspect: str = "16:9",
):
    if os.path.isfile(interpolate_output):
        print('Already interpolated')
        return
    
    vapoursynth_file = f"{upscale_video_input}.vpy"
    ffindex_file = f"{upscale_video_input}.ffindex"

    _create_vsrife_script(vapoursynth_file, hentai_name, video_resolution, video_aspect)
    _interpolate(vapoursynth_file, interpolate_output)

    # Cleanup
    os.remove(ffindex_file)
    os.remove(vapoursynth_file)
    