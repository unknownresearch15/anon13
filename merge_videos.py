import math
import os
import subprocess

video_folder = "asset/good_videos"
output_file = "output_grid.mp4"

rows, cols = 6, 8

videos = [
    os.path.join(video_folder, f)
    for f in os.listdir(video_folder)
    if f.endswith((".mp4", ".mov", ".avi"))
]
videos.sort()

videos = videos[: rows * cols]

filter_complex = ""
for i in range(len(videos)):
    filter_complex += f"[{i}:v]setpts=PTS-STARTPTS,scale=256x256[v{i}];"  # specify the size of the video

inputs = " ".join(f"-i '{v}'" for v in videos)
grid = "".join(f"[v{i}]" for i in range(len(videos)))
filter_complex += f"{grid}xstack=inputs={len(videos)}:layout="
layout = []
for r in range(rows):
    for c in range(cols):
        layout.append(f"{c * 256}_{r * 256}")
filter_complex += "|".join(layout) + "[out]"

# final ffmpeg command
cmd = f'ffmpeg {inputs} -filter_complex "{filter_complex}" -map "[out]" -c:v libx264 -crf 23 -preset fast {output_file}'

print("Running command:\n", cmd)
subprocess.run(cmd, shell=True)
