import copy
import random
import time
from dataclasses import dataclass, field

from src.tracker import cell_from_normalized, CELL_NAMES

TARGET_TIMEOUT_S = 5.0
MIN_DWELL_FRAMES = 3

# Weighted cell selection: corners > edges > center
_CELL_WEIGHTS = [1.5, 1.0, 1.5, 1.0, 0.5, 1.0, 1.5, 1.0, 1.5]


@dataclass
class GameState:
    phase: str = "idle"
    duration_s: int = 60
    start_time: float = 0.0
    current_target: int = -1
    target_start_time: float = 0.0
    hand_side: str = "right"
    events: list = field(default_factory=list)
    trajectory_buffer: list = field(default_factory=list)
    participant_name: str = ""
    participant_age: int | None = None
    last_target: int = -1
    dwell_count: int = 0
    flash_frames: int = 0
    flash_color: str = "none"
    _event_counter: int = 0


@dataclass
class FrameResult:
    event: str = "none"   # "none" | "hit" | "timeout" | "new_target" | "game_over"
    hit_cell: int = -1
    new_target: int = -1
    remaining_s: float = 0.0
    elapsed_s: float = 0.0
    dwell_count: int = 0
    hit_count: int = 0


class GameEngine:
    def start(self, duration_s: int, hand_side: str) -> GameState:
        state = GameState(
            phase="countdown",
            duration_s=duration_s,
            start_time=time.perf_counter(),
            hand_side=hand_side,
        )
        state.current_target = _select_next_target(-1)
        state.target_start_time = state.start_time
        return state

    def process_frame(
        self,
        state: GameState,
        right_pos: tuple | None,
        left_pos: tuple | None,
        grid_bounds: tuple,
        frame_time: float,
    ) -> tuple:
        if state is None or state.phase != "running":
            return state, FrameResult()

        state = copy.copy(state)
        state.events = state.events
        state.trajectory_buffer = state.trajectory_buffer

        elapsed = frame_time - state.start_time
        remaining = max(0.0, state.duration_s - elapsed)
        result = FrameResult(remaining_s=remaining, elapsed_s=elapsed)

        if state.flash_frames > 0:
            state.flash_frames -= 1
        else:
            state.flash_color = "none"

        # Determine active hand position
        active_pos = _get_active_pos(state.hand_side, right_pos, left_pos)
        active_hand = _get_active_hand(state.hand_side, right_pos, left_pos)

        # Record trajectory
        if active_pos is not None:
            state.trajectory_buffer = list(state.trajectory_buffer)
            state.trajectory_buffer.append({
                "t": frame_time,
                "x": active_pos[0],
                "y": active_pos[1],
            })
            if len(state.trajectory_buffer) > 100:
                state.trajectory_buffer = state.trajectory_buffer[-100:]

        # Check game over
        if elapsed >= state.duration_s:
            state.phase = "done"
            _record_event(state, success=False, hit_time=None, hand=active_hand,
                          frame_time=frame_time)
            result.event = "game_over"
            result.hit_count = _count_hits(state.events)
            return state, result

        # Check hand in target cell
        if state.hand_side == "both":
            r_cell = cell_from_normalized(right_pos, grid_bounds) if right_pos else -1
            l_cell = cell_from_normalized(left_pos, grid_bounds) if left_pos else -1
            r_hit = r_cell == state.current_target
            l_hit = l_cell == state.current_target
            in_target = r_hit or l_hit
            active_hand = "right" if r_hit else ("left" if l_hit else active_hand)
        else:
            current_cell = cell_from_normalized(active_pos, grid_bounds) if active_pos else -1
            in_target = current_cell == state.current_target

        if in_target:
            state.dwell_count += 1
        else:
            state.dwell_count = 0

        result.dwell_count = state.dwell_count

        # Hit registered
        if state.dwell_count >= MIN_DWELL_FRAMES:
            _record_event(state, success=True, hit_time=frame_time, hand=active_hand,
                          frame_time=frame_time)
            state.last_target = state.current_target
            state.current_target = _select_next_target(state.last_target)
            state.target_start_time = frame_time
            state.dwell_count = 0
            state.flash_frames = 5
            state.flash_color = "green"
            state.trajectory_buffer = []
            result.event = "hit"
            result.hit_cell = state.last_target
            result.new_target = state.current_target

        # Timeout
        elif frame_time - state.target_start_time > TARGET_TIMEOUT_S:
            _record_event(state, success=False, hit_time=None, hand=active_hand,
                          frame_time=frame_time)
            state.last_target = state.current_target
            state.current_target = _select_next_target(state.last_target)
            state.target_start_time = frame_time
            state.dwell_count = 0
            state.flash_frames = 3
            state.flash_color = "red"
            state.trajectory_buffer = []
            result.event = "timeout"
            result.new_target = state.current_target

        result.hit_count = _count_hits(state.events)
        return state, result


def _select_next_target(last: int) -> int:
    weights = list(_CELL_WEIGHTS)
    if last >= 0:
        weights[last] = 0.0
    cells = list(range(9))
    return random.choices(cells, weights=weights, k=1)[0]


def _get_active_pos(hand_side: str, right_pos, left_pos):
    if hand_side == "right":
        return right_pos
    if hand_side == "left":
        return left_pos
    # "both": pick whichever is not None, prefer right
    return right_pos if right_pos is not None else left_pos


def _get_active_hand(hand_side: str, right_pos, left_pos) -> str:
    if hand_side == "right":
        return "right"
    if hand_side == "left":
        return "left"
    return "right" if right_pos is not None else "left"


def _record_event(state: GameState, success: bool, hit_time, hand: str, frame_time: float):
    traj = list(state.trajectory_buffer)
    path_length = _compute_path_length(traj)
    direct_dist = _compute_direct_distance(traj)
    rt = (hit_time - state.target_start_time) * 1000 if (success and hit_time) else None

    event = {
        "event_id": state._event_counter,
        "target_cell": state.current_target,
        "hand": hand,
        "target_shown_at": state.target_start_time,
        "hit_at": hit_time,
        "reaction_time_ms": rt,
        "trajectory": traj,
        "success": success,
        "path_length": path_length,
        "direct_distance": direct_dist,
    }
    state.events = list(state.events)
    state.events.append(event)
    state._event_counter += 1


def _compute_path_length(traj: list) -> float:
    if len(traj) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(traj)):
        dx = traj[i]["x"] - traj[i - 1]["x"]
        dy = traj[i]["y"] - traj[i - 1]["y"]
        total += (dx * dx + dy * dy) ** 0.5
    return total


def _compute_direct_distance(traj: list) -> float:
    if len(traj) < 2:
        return 0.0
    dx = traj[-1]["x"] - traj[0]["x"]
    dy = traj[-1]["y"] - traj[0]["y"]
    return (dx * dx + dy * dy) ** 0.5


def _count_hits(events: list) -> int:
    return sum(1 for e in events if e["success"])
