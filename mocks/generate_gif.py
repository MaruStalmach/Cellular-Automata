###############################
# Example usage
# python -m mocks.generate_gif -h
# python -m mocks.generate_gif -f multispecies_experiment -k bact_red bact_green bact_blue --fps 12
# python -m mocks.generate_gif -f demo -k alive --duration 8
##############################################################

import argparse
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

from libs.rendering.Renderer import CARenderer
from libs.tracking.StateTracker import StateTracker, TrackedStatePlayback


def _parse_keys(raw_keys: list[str] | None, available_keys: tuple[str, ...]) -> list[str]:
    if not raw_keys:
        return list(available_keys)

    parsed_keys: list[str] = []
    for item in raw_keys:
        parsed_keys.extend([key for key in item.split(",") if key])

    return parsed_keys


def _frame_for_keys(state, keys: list[str], spatial_rank: int) -> np.ndarray:
    frames = []
    for key in keys:
        values = state[key]
        if values.ndim == spatial_rank + 1:
            values = values.max(axis=-1)
        frames.append(values)

    if len(frames) == 1:
        return frames[0]

    return np.stack(frames, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generate_gif",
        description="Replay a tracked simulation in the renderer",
    )
    parser.add_argument("-f", "--file", required=True, help="tracked .npz file name")
    parser.add_argument(
        "-k",
        "--keys",
        nargs="+",
        help="one or more keys to render, separated by spaces or commas",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="GIF frame rate when duration is not set",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="total GIF duration in seconds; overrides fps when set",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="GIF loop count; 0 means loop forever",
    )

    args = parser.parse_args()

    tracker = StateTracker.load(args.file)
    selected_keys = _parse_keys(args.keys, tracker.history.keys())

    if not selected_keys:
        raise ValueError("no keys selected for replay")

    playback = TrackedStatePlayback(tracker, step_no=0)
    spatial_rank = len(playback.state.geometry.size)
    captured_frames: list[np.ndarray] = []
    should_capture_frame = False

    def update_callback(step: bool = True):
        nonlocal should_capture_frame
        if step:
            should_capture_frame = True
            playback(step=True)
        return _frame_for_keys(playback.state, selected_keys, spatial_rank)

    def frame_callback(frame: np.ndarray) -> None:
        nonlocal should_capture_frame
        if should_capture_frame:
            captured_frames.append(frame)
            should_capture_frame = False

    def save_gif() -> None:
        if not captured_frames:
            print("No frames captured yet; press P to record frames before saving a GIF.")
            return

        output_dir = Path("gifs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(args.file).stem}.gif"

        if args.duration is not None:
            frame_duration = args.duration / len(captured_frames)
        else:
            frame_duration = 1.0 / args.fps

        imageio.mimsave(
            output_path,
            captured_frames,
            duration=frame_duration,
            loop=args.loop,
        )
        print(f"Saved GIF to {output_path}")

    first_frame = _frame_for_keys(playback.state, selected_keys, spatial_rank)

    renderer = CARenderer(
        init_state=first_frame,
        width=1200,
        height=800,
        update_callback=update_callback,
        frame_callback=frame_callback,
        gif_callback=save_gif,
        keys_to_render=len(selected_keys),
    )
    renderer.run()


if __name__ == "__main__":
    main()