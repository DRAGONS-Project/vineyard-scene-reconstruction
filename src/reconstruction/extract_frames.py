"""Extract frames from a video at a fixed sampling rate."""

import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: Path, output_dir: Path, fps: float) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(1, round(source_fps / fps))

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            out_path = output_dir / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("output_dir", type=Path, help="Directory to write extracted frames")
    parser.add_argument(
        "--fps", type=float, default=2.0, help="Target sampling rate in frames per second"
    )
    args = parser.parse_args()

    n = extract_frames(args.video, args.output_dir, args.fps)
    print(f"Extracted {n} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
