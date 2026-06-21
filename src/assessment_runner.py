"""Launches the cv2 assessment window as a subprocess.

cv2.imshow requires the main thread on Windows.  Running in a subprocess
gives the window its own OS process with a fresh main thread, making it
fully reliable without any platform-specific workarounds.
"""
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

from src.game_engine import GameState


class AssessmentRunner:
    """Manages the lifecycle of the cv2 assessment subprocess."""

    def __init__(self, tracker=None, engine=None) -> None:
        # tracker / engine are accepted for API compatibility but unused here —
        # the subprocess creates its own instances.
        self._proc: subprocess.Popen | None = None
        self._output_file: str | None = None
        self._stop_file: str | None = None
        self.final_state: GameState | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def is_done(self) -> bool:
        """True once the subprocess has written its result file."""
        if self.final_state is not None:
            return True
        if self._output_file and os.path.exists(self._output_file):
            try:
                with open(self._output_file, "rb") as fh:
                    self.final_state = pickle.load(fh)
                try:
                    os.unlink(self._output_file)
                except OSError:
                    pass
                self._output_file = None
                return True
            except Exception:
                pass  # file may still be mid-write; try again next poll
        return False

    def start(self, initial_state: GameState) -> None:
        if self.is_running:
            return
        self.final_state = None

        # Temp file for the subprocess to write its final GameState into
        fd_out, self._output_file = tempfile.mkstemp(suffix=".pkl", prefix="assessment_result_")
        os.close(fd_out)
        os.unlink(self._output_file)   # subprocess creates it; we just need the path

        # Temp file whose *existence* tells the subprocess to stop
        fd_stop, self._stop_file = tempfile.mkstemp(suffix=".stop", prefix="assessment_stop_")
        os.close(fd_stop)
        os.unlink(self._stop_file)     # doesn't exist yet — stop is signalled by creating it

        script = Path(__file__).parent / "run_assessment.py"
        cmd = [
            sys.executable, str(script),
            "--duration",  str(initial_state.duration_s),
            "--hand",      initial_state.hand_side,
            "--name",      initial_state.participant_name or "",
            "--age",       str(initial_state.participant_age or 0),
            "--output",    self._output_file,
            "--stopfile",  self._stop_file,
        ]
        # stdout/stderr inherit from parent so errors appear in the Gradio console
        self._proc = subprocess.Popen(cmd)

    def stop(self) -> None:
        """Signal the subprocess to stop by creating the stop-signal file."""
        if self._stop_file:
            try:
                Path(self._stop_file).touch()
            except OSError:
                pass
        # Give it a moment to finish gracefully; terminate if it hangs
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
