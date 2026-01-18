import json
import os
import subprocess
import sys

# Import centralized configuration for consistent paths across all scripts
from environment import resolve_project_dir

# Get project number from command line or use default
project_num = input("Enter project number: ") or "001"

# === SETTINGS ===
# Use centralized path configuration (handles /app/projects in cloud, projects locally)
project_dir = resolve_project_dir(project_num)
frames_dir = project_dir / "frames_filtered_active_only"  # Input: frames from PyVista

output_video = project_dir / "fea_animation.mp4"  # Output MP4 file at project root
fps = 5  # Frames per second

# === CREATE MP4 VIDEO ===
# Check frames directory exists
if not frames_dir.exists():
    print(f"❌ Frames directory not found: {frames_dir}")
    sys.exit(1)

# Count frames for reporting
frame_files = [f for f in os.listdir(frames_dir) if f.endswith(".png")]
if not frame_files:
    print(f"❌ No frames found in {frames_dir}")
    sys.exit(1)

num_frames = len(frame_files)
print(f"🎬 Creating {output_video} with {num_frames} frames using ffmpeg...")

# Use ffmpeg for multi-threaded video encoding
# -framerate: input framerate
# -i: input pattern (frame_00000.png, frame_00001.png, etc.)
# -c:v libx264: use H.264 codec
# -threads 0: use all available CPU cores
# -preset fast: encoding speed/compression tradeoff
# -crf 23: constant rate factor (18-28 is good range, lower = better quality)
# -pix_fmt yuv420p: pixel format for compatibility
# -y: overwrite output file without asking

ffmpeg_cmd = [
    "ffmpeg",
    "-framerate", str(fps),
    "-i", str(frames_dir / "frame_%05d.png"),
    "-c:v", "libx264",
    "-threads", "0",  # Use all available cores
    "-preset", "fast",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
    "-y",  # Overwrite output
    str(output_video)
]

try:
    result = subprocess.run(
        ffmpeg_cmd,
        check=True,
        capture_output=True,
        text=True
    )
    print(f"✅ Saved animation as {output_video}")
except subprocess.CalledProcessError as e:
    print(f"❌ FFmpeg failed with error:")
    print(e.stderr)
    sys.exit(1)

# Save animation file size to progress.json
animation_file_size = output_video.stat().st_size
progress_file = project_dir / "progress.json"

# Read existing progress data
progress_data = {}
if progress_file.exists():
    try:
        with open(progress_file, "r") as f:
            progress_data = json.load(f)
    except Exception:
        pass

# Format file size in human-readable format
if animation_file_size < 1024:
    file_size_formatted = f"{animation_file_size} B"
elif animation_file_size < 1024 * 1024:
    file_size_formatted = f"{animation_file_size / 1024:.1f} KB"
elif animation_file_size < 1024 * 1024 * 1024:
    file_size_formatted = f"{animation_file_size / (1024 * 1024):.1f} MB"
else:
    file_size_formatted = f"{animation_file_size / (1024 * 1024 * 1024):.1f} GB"

# Add file size to progress data
progress_data["animation_file_size"] = animation_file_size
progress_data["animation_file_size_formatted"] = file_size_formatted

# Write updated progress data
try:
    with open(progress_file, "w") as f:
        json.dump(progress_data, f, indent=2)
    print(f"📊 Animation file size: {file_size_formatted}")
except Exception as e:
    print(f"⚠️ Could not save file size to progress.json: {e}")
