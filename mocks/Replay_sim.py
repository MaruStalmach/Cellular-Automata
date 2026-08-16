import argparse

import numpy as np

from libs.rendering.Renderer import CARenderer
from libs.tracking.StateTracker import StateTracker, TrackedStatePlayback


##########################
# EXAMPLE USAGE
# python -m mocks.Replay_sim -f demo -k bacteria substrate
# python -m mocks.Replay_sim -f output_files/tracked_simulations/demo.npz -k bacteria,substrate
#############################


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
        prog="Replay_sim",
        description="Replay a tracked cellular automaton simulation",
    )
    parser.add_argument("-f", "--file", required=True, help="tracked .npz file name")
    parser.add_argument(
        "-k",
        "--keys",
        nargs="+",
        help="one or more keys to render, separated by spaces or commas",
    )
    parser.add_argument("-s", "--stride", default=1)

    args = parser.parse_args()

    tracker = StateTracker.load(args.file)
    selected_keys = _parse_keys(args.keys, tracker.history.keys())
    stride = int(args.stride)

    if not selected_keys:
        raise ValueError("no keys selected for replay")

    playback = TrackedStatePlayback(tracker, step_no=0, stride=stride)
    spatial_rank = len(playback.state.geometry.size)

    def update_callback(step: bool = True):
        if step:
            playback(step=True)
        return _frame_for_keys(playback.state, selected_keys, spatial_rank)

    first_frame = _frame_for_keys(playback.state, selected_keys, spatial_rank)
    renderer = CARenderer(
        init_state=first_frame,
        update_callback=update_callback,
        keys_to_render=len(selected_keys),
    )
    renderer.run()


if __name__ == "__main__":
    main()