import os
import re

from utils.interpolate import interpolate
from utils.mediainfo import get_aspect_ratio

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

for filename in os.listdir('0-Source'):
    if filename == '.gitignore':
        continue

    input_file = os.path.join('0-Source', filename)
    if not os.path.isfile(input_file):
        continue

    # -------- ファイル名解析 --------
    try:
        hentai_name = re.sub(r'\[.*?\]|\(.*?\)', "", filename).rsplit('.', 1)[0].strip()
        folder_name = re.sub(r'[^A-Za-z ]+', '', hentai_name).strip()
        episode_number = re.findall(r'\d+', hentai_name)[-1]
    except:
        print(f"Error Parsing name for: {filename}")
        continue

    print("Parsed Name:", hentai_name)

    # -------- 出力フォルダ --------
    output_folder = os.path.join('2-Out', folder_name)
    create_folder(output_folder)

    # -------- 出力ファイル名 --------
    interpolate_output = os.path.join(
        output_folder,
        f"{hentai_name} [48fps][HEVC].mkv"
    )

    # -------- アスペクト比取得 --------
    aspect_ratio = get_aspect_ratio(input_file)

    # -------- 元動画 → 直接フレーム補間 --------
    interpolate(
        input_file,            # 入力（元動画）
        interpolate_output,    # 出力
        hentai_name,
        "1080p",               # ここは元コードに合わせるのが安全
        aspect_ratio
    )

print("All files processed.")