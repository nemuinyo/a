import os
import re
import sys

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

    # ---------- 名前解析 ----------
    try:
        hentai_name = re.sub(r'\[.*?\]|\(.*?\)', "", filename).rsplit('.', 1)[0].strip()
        folder_name = re.sub(r'[^A-Za-z0-9 ]+', '', hentai_name).strip()
    except:
        print(f"Error Parsing name for: {filename}")
        continue

    print("Parsed Name:", hentai_name)

    # ---------- 出力フォルダ ----------
    output_folder = os.path.join('2-Out', folder_name)
    create_folder(output_folder)

    # ---------- 出力ファイル ----------
    interpolate_output = os.path.join(
        output_folder,
        f"{hentai_name} [48fps][HEVC].mkv"
    )

    # ---------- アスペクト比取得 ----------
    aspect_ratio = get_aspect_ratio(input_file)

    # ---------- フレーム補間のみ ----------
    interpolate(
        input_file,            # 元動画
        interpolate_output,    # 出力
        hentai_name,
        "1080p",
        aspect_ratio
    )

print("All files processed.")
