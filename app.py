import ctypes
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, QUrl, Signal
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QFormLayout, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QProgressBar, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from pipeline import comfy_server, images, voice
from pipeline.segmenter import (
    DEFAULT_SYSTEM, Scene, split_sentences, stream_story,
)

PREVIEW_CHARS = 70
WORDS_PER_MINUTE = 150
THUMB_HEIGHT = 200


def resource_path(name: str) -> str:
    """Works both running from source and from a PyInstaller onefile exe,
    where bundled files are extracted to sys._MEIPASS at runtime."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    return str(base / name)


ICON_PATH = resource_path("resources/images/icons/creepty_ico.ico")


# ---------- shared pipeline helpers ----------

def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_image_prompt(scene: Scene) -> str:
    """Structured prompt: the Qwen3 text encoder reads labelled fields."""
    lines = [
        f"context: {scene.text.strip()}",
        f"image: {scene.image_prompt.strip()}",
    ]
    if scene.mood.strip():
        lines.append(f"mood: {scene.mood.strip()}")
    return "\n".join(lines)


def asset_name(run_id: str, row: int) -> str:
    # The suffix keeps every render unique, so a moved or regenerated scene
    # never points at a file that belongs to another scene.
    return f"{run_id}/scene_{row + 1:03d}_{uuid.uuid4().hex[:6]}"


def can_render_image(scene: Scene) -> bool:
    return bool(scene.text.strip() and scene.image_prompt.strip())


def can_render_voice(scene: Scene) -> bool:
    return bool(scene.text.strip())


# ---------- workers ----------

class Generator(QObject):
    """Runs the whole pipeline off the UI thread: scenes, images, voice."""

    progress = Signal(str)                      # splitting / startup status
    image_progress = Signal(str)                # image-stream status
    voice_progress = Signal(str)                # voice-stream status
    scenes_ready = Signal(int, list)             # batch index, scenes
    asset_done = Signal(int, str, str, float)    # row, kind, path, seconds
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, story: str, prompt: str, run_id: str):
        super().__init__()
        self.story = story
        self.prompt = prompt
        self.run_id = run_id
        self.scenes: list[Scene] = []
        self.cancelled = False

    def cancel(self):
        """Checked between steps; also stops the image being rendered."""
        self.cancelled = True
        images.interrupt()

    def run(self):
        try:
            self.progress.emit("Splitting scenes...")
            for index, total, scenes in stream_story(self.story, self.prompt):
                if self.cancelled:
                    self.finished.emit("Cancelled.")
                    return
                self.progress.emit(f"Splitting scenes (batch {index} of {total})...")
                self.scenes.extend(scenes)
                self.scenes_ready.emit(index, scenes)

            if self.cancelled:
                self.finished.emit("Cancelled.")
                return

            self.progress.emit("Starting ComfyUI...")
            if not comfy_server.start():
                self.failed.emit(
                    f"Couldn't start ComfyUI in {comfy_server.COMFY_DIR}."
                )
                return

            self.generate_assets()
            self.finished.emit("Cancelled." if self.cancelled else "Done.")
        except Exception as error:
            self.failed.emit(f"Failed: {error}")

    def generate_assets(self):
        """Images (GPU, via ComfyUI's HTTP API) and voice (GPU, local
        Parler-TTS) run one after the other, since both compete for the
        same limited VRAM. Each stream still processes its own scenes in
        order and reports its own status independently."""
        image_rows = [
            row for row, scene in enumerate(self.scenes)
            if can_render_image(scene)
        ]
        voice_rows = [
            row for row, scene in enumerate(self.scenes)
            if can_render_voice(scene)
        ]

        total = len(image_rows)
        for index, row in enumerate(image_rows, start=1):
            if self.cancelled:
                return
            self.image_progress.emit(f"{index} of {total} (scene {row + 1})")
            start = time.monotonic()
            path = images.generate_image(
                build_image_prompt(self.scenes[row]), asset_name(self.run_id, row)
            )
            self.asset_done.emit(row, "image", path, time.monotonic() - start)
        if total:
            self.image_progress.emit("done")

        total = len(voice_rows)
        for index, row in enumerate(voice_rows, start=1):
            if self.cancelled:
                return
            self.voice_progress.emit(f"{index} of {total} (scene {row + 1})")
            start = time.monotonic()
            path = voice.generate_voice(
                self.scenes[row].text, asset_name(self.run_id, row),
                mood=self.scenes[row].mood,
            )
            self.asset_done.emit(row, "audio", path, time.monotonic() - start)
        if total:
            self.voice_progress.emit("done")


class AssetWorker(QObject):
    """Renders images and/or voice for specific scenes, using their current
    edited state. Images and voice run one after the other, since both
    compete for the same limited VRAM. Either job list may be empty, so
    this also covers a single-scene regen of just one asset kind. Each
    stream reports its own status independently, so both are folded into
    the same status line."""

    progress = Signal(str)                       # startup status only
    image_progress = Signal(str)
    voice_progress = Signal(str)
    asset_done = Signal(int, str, str, float)     # row, kind, path, seconds
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        image_jobs: list[tuple[int, Scene]],
        voice_jobs: list[tuple[int, Scene]],
        run_id: str,
    ):
        super().__init__()
        self.image_jobs = image_jobs
        self.voice_jobs = voice_jobs
        self.run_id = run_id
        self.cancelled = False

    def cancel(self):
        """Checked between items in each stream; also stops the image
        currently being rendered."""
        self.cancelled = True
        images.interrupt()

    def _run_images(self):
        total = len(self.image_jobs)
        for index, (row, scene) in enumerate(self.image_jobs, start=1):
            if self.cancelled:
                return
            self.image_progress.emit(f"{index} of {total} (scene {row + 1})")
            start = time.monotonic()
            path = images.generate_image(
                build_image_prompt(scene), asset_name(self.run_id, row)
            )
            self.asset_done.emit(row, "image", path, time.monotonic() - start)
        if total:
            self.image_progress.emit("done")

    def _run_voices(self):
        total = len(self.voice_jobs)
        for index, (row, scene) in enumerate(self.voice_jobs, start=1):
            if self.cancelled:
                return
            self.voice_progress.emit(f"{index} of {total} (scene {row + 1})")
            start = time.monotonic()
            path = voice.generate_voice(
                scene.text, asset_name(self.run_id, row), mood=scene.mood
            )
            self.asset_done.emit(row, "audio", path, time.monotonic() - start)
        if total:
            self.voice_progress.emit("done")

    def run(self):
        try:
            if self.image_jobs:
                self.progress.emit("Starting ComfyUI...")
                if not comfy_server.start():
                    self.failed.emit(
                        f"Couldn't start ComfyUI in {comfy_server.COMFY_DIR}."
                    )
                    return
                self._run_images()

            if self.voice_jobs:
                self._run_voices()

            self.finished.emit("Cancelled." if self.cancelled else "Done.")
        except Exception as error:
            self.failed.emit(f"Failed: {error}")


# ---------- window ----------

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # Keep references alive: a local QThread gets collected mid-run.
        self.worker = None
        self.thread = None

        # The list is the source of truth; the table is only a view of it.
        self.scenes: list[Scene] = []
        self.current = -1

        self.loading = False      # guards the detail fields against feedback
        self.segmenting = False   # scene count isn't final while this is True
        self.busy = False         # any worker running
        self.run_id: str | None = None

        # Seconds per generated asset, used for the ETA. Tracked separately
        # per kind: image and voice now run one after the other (both use
        # the GPU), so the total wait is their sum, not the max of the two.
        self.image_times: list[float] = []
        self.voice_times: list[float] = []

        # Pieces of the single status line: the current stage message, and
        # the two asset streams once they start reporting. Kept separate so
        # they can be recombined without one overwriting the other.
        self._stage_text = ""
        self._image_text = ""
        self._voice_text = ""

        # One player for the whole window; the output must stay referenced.
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        self.setWindowTitle("Creepty")
        self.resize(1200, 880)
        self.setCentralWidget(self._build())
        self.set_busy(False)
        self.update_progress()

    # ---------- construction ----------

    def _build(self) -> QWidget:
        root = QVBoxLayout()
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)
        root.addWidget(QLabel("Model instructions"))
        root.addWidget(self._prompt_box())
        root.addWidget(QLabel("Story"))
        root.addWidget(self._story_box())
        root.addLayout(self._action_row())
        root.addWidget(self.status)
        root.addWidget(self._scene_area(), stretch=1)
        central = QWidget()
        central.setLayout(root)
        return central

    def _prompt_box(self) -> QWidget:
        self.prompt = QTextEdit()
        self.prompt.setPlainText(DEFAULT_SYSTEM)
        self.prompt.setMaximumHeight(90)
        return self.prompt

    def _story_box(self) -> QWidget:
        self.story = QTextEdit()
        self.story.setPlaceholderText(
            "Paste the text you want to turn into a video."
        )
        self.story.setMaximumHeight(140)
        return self.story

    def _action_row(self) -> QHBoxLayout:
        self.button = QPushButton("Generate")
        self.button.clicked.connect(self.generate)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        self.reset = QPushButton("Reset prompt")
        self.reset.clicked.connect(
            lambda: self.prompt.setPlainText(DEFAULT_SYSTEM)
        )

        # Single line for every status message: stage transitions
        # ("Splitting scenes...", "Starting ComfyUI...") followed by the
        # two asset streams once they start, e.g.
        # "Images: 2 of 5 (scene 3) | Voice: 1 of 4 (scene 2)"
        self.status = QLabel("")

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v of %m assets")
        self.progress_bar.setMinimumWidth(280)

        self.eta = QLabel("")
        self.eta.setStyleSheet("color: #888;")

        row = QHBoxLayout()
        row.addWidget(self.button)
        row.addWidget(self.cancel_button)
        row.addWidget(self.reset)
        row.addStretch()
        row.addWidget(self.progress_bar)
        row.addWidget(self.eta)
        return row

    def _scene_area(self) -> QWidget:
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._scene_list())
        split.addWidget(self._detail_panel())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 3)
        return split

    def _scene_list(self) -> QWidget:
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["#", "Scene", "Mood", "~Time", "Assets"]
        )
        self.table.verticalHeader().setVisible(False)
        # Read-only and single-line: long text never blows up a row.
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.setWordWrap(False)
        self.table.verticalHeader().setDefaultSectionSize(28)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in (2, 3, 4):
            header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )

        self.table.itemSelectionChanged.connect(self.load_detail)

        self.insert_button = QPushButton("Insert")
        self.delete_button = QPushButton("Delete")
        self.split_button = QPushButton("Split")
        self.up_button = QPushButton("Up")
        self.down_button = QPushButton("Down")
        self.missing_button = QPushButton("Generate missing")

        self.insert_button.clicked.connect(self.insert_scene)
        self.delete_button.clicked.connect(self.delete_scene)
        self.split_button.clicked.connect(self.split_scene)
        self.up_button.clicked.connect(lambda: self.move_scene(-1))
        self.down_button.clicked.connect(lambda: self.move_scene(1))
        self.missing_button.clicked.connect(self.generate_missing)

        buttons = QHBoxLayout()
        for widget in (
            self.insert_button, self.delete_button, self.split_button,
            self.up_button, self.down_button,
        ):
            buttons.addWidget(widget)
        buttons.addStretch()
        buttons.addWidget(self.missing_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        layout.addLayout(buttons)
        panel = QWidget()
        panel.setLayout(layout)
        return panel

    def _detail_panel(self) -> QWidget:
        self.detail_text = QTextEdit()
        self.detail_text.setPlaceholderText("Select a scene on the left.")
        self.detail_image = QTextEdit()
        self.detail_image.setMaximumHeight(80)
        self.detail_mood = QLineEdit()

        for widget in (self.detail_text, self.detail_image, self.detail_mood):
            widget.setEnabled(False)

        self.detail_text.textChanged.connect(self.save_detail)
        self.detail_image.textChanged.connect(self.save_detail)
        self.detail_mood.textChanged.connect(self.save_detail)

        form = QFormLayout()
        form.setContentsMargins(12, 0, 0, 0)
        form.addRow(QLabel("Narration text"))
        form.addRow(self.detail_text)
        form.addRow(QLabel("Image prompt"))
        form.addRow(self.detail_image)
        form.addRow("Mood", self.detail_mood)
        form.addRow(self._preview_row())

        panel = QWidget()
        panel.setLayout(form)
        return panel

    def _preview_row(self) -> QWidget:
        self.thumbnail = QLabel("No image yet")
        self.thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail.setFixedHeight(THUMB_HEIGHT)
        self.thumbnail.setFrameShape(QFrame.Shape.StyledPanel)
        self.thumbnail.setStyleSheet("color: #888;")
        self.image_button = QPushButton("Generate image")
        self.image_button.clicked.connect(self.generate_current_image)

        # Audio slot: plays audio_path through the shared QMediaPlayer.
        self.audio_label = QLabel("No audio yet")
        self.audio_label.setStyleSheet("color: #888;")
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_audio)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.player.stop)
        self.voice_button = QPushButton("Generate voice")
        self.voice_button.clicked.connect(self.generate_current_voice)

        audio_row = QHBoxLayout()
        audio_row.addWidget(self.play_button)
        audio_row.addWidget(self.stop_button)
        audio_row.addWidget(self.voice_button)
        audio_row.addStretch()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 8, 0, 0)
        layout.addWidget(QLabel("Image"))
        layout.addWidget(self.thumbnail)
        layout.addWidget(self.image_button)
        layout.addWidget(QLabel("Audio"))
        layout.addWidget(self.audio_label)
        layout.addLayout(audio_row)
        box = QWidget()
        box.setLayout(layout)
        return box

    # ---------- busy state ----------

    def set_busy(self, busy: bool):
        """Locks everything that would shift rows under a running worker."""
        self.busy = busy
        self.button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        for widget in (
            self.insert_button, self.delete_button, self.split_button,
            self.up_button, self.down_button, self.missing_button,
        ):
            widget.setEnabled(not busy)
        self.refresh_detail_buttons()

    def refresh_detail_buttons(self):
        selected = 0 <= self.current < len(self.scenes)
        scene = self.scenes[self.current] if selected else None
        self.image_button.setEnabled(
            selected and not self.busy and can_render_image(scene)
        )
        self.voice_button.setEnabled(
            selected and not self.busy and can_render_voice(scene)
        )

    # ---------- status line ----------

    def set_stage(self, text: str):
        """A one-off stage message (splitting, starting ComfyUI). Once the
        asset streams start reporting, their combined line takes over."""
        self._stage_text = text
        self._image_text = ""
        self._voice_text = ""
        self.status.setText(text)

    def set_image_status(self, text: str):
        self._image_text = text
        self._refresh_asset_line()

    def set_voice_status(self, text: str):
        self._voice_text = text
        self._refresh_asset_line()

    def _refresh_asset_line(self):
        parts = []
        if self._image_text:
            parts.append(f"Images: {self._image_text}")
        if self._voice_text:
            parts.append(f"Voice: {self._voice_text}")
        if parts:
            self.status.setText(" | ".join(parts))

    # ---------- progress ----------

    def update_progress(self):
        """Counts finished assets: one image and one audio per scene."""
        total = len(self.scenes) * 2
        # Nothing to track yet: hide instead of showing an empty bar.
        self.progress_bar.setVisible(bool(total) or self.segmenting)
        self.eta.setVisible(bool(total) or self.segmenting)

        if self.segmenting:
            self.progress_bar.setRange(0, 0)  # indeterminate
            self.eta.setText("Splitting...")
            return

        if not total:
            return

        images_done = sum(bool(scene.image_path) for scene in self.scenes)
        voices_done = sum(bool(scene.audio_path) for scene in self.scenes)
        done = images_done + voices_done

        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)

        images_remaining = len(self.scenes) - images_done
        voices_remaining = len(self.scenes) - voices_done

        if not images_remaining and not voices_remaining:
            self.eta.setText("Complete")
            return

        # Images and voice now run one after the other (both use the GPU),
        # so the wait is their combined total, not whichever is slower.
        image_seconds = self._stream_eta(self.image_times, images_remaining)
        voice_seconds = self._stream_eta(self.voice_times, voices_remaining)

        # A stream with items still remaining but no completed asset yet
        # has no real estimate: treating it as 0 (as `... or 0` would)
        # silently drops it from the ETA instead of flagging that the
        # total is a floor, not a real number.
        pending = []
        if images_remaining and image_seconds is None:
            pending.append("images")
        if voices_remaining and voice_seconds is None:
            pending.append("voice")

        if image_seconds is None and voice_seconds is None:
            self.eta.setText(f"{images_remaining + voices_remaining} left")
            return

        seconds = (image_seconds or 0) + (voice_seconds or 0)
        hours, rest = divmod(int(seconds), 3600)
        minutes, secs = divmod(rest, 60)
        eta = f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"
        if pending:
            eta += f"+ ({'/'.join(pending)} not started)"
        self.eta.setText(f"~{eta} left")

    @staticmethod
    def _stream_eta(times: list[float], remaining: int) -> float | None:
        """Estimated seconds left for one stream, or None if it has no
        completed assets yet to estimate from."""
        if remaining <= 0 or not times:
            return None
        # Average the last 10: early runs are slower than steady state.
        window = times[-10:]
        return sum(window) / len(window) * remaining

    # ---------- previews ----------

    def load_previews(self, scene: Scene | None):
        if scene is None:
            self.thumbnail.setPixmap(QPixmap())
            self.thumbnail.setText("No image yet")
            self.audio_label.setText("No audio yet")
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            return

        if scene.image_path:
            pixmap = QPixmap(scene.image_path)
            if pixmap.isNull():
                self.thumbnail.setPixmap(QPixmap())
                self.thumbnail.setText("Image file not found")
            else:
                self.thumbnail.setPixmap(
                    pixmap.scaledToHeight(
                        THUMB_HEIGHT,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        else:
            self.thumbnail.setPixmap(QPixmap())
            self.thumbnail.setText("No image yet")

        has_audio = bool(scene.audio_path)
        self.audio_label.setText(
            scene.audio_path if has_audio else "No audio yet"
        )
        self.play_button.setEnabled(has_audio)
        self.stop_button.setEnabled(has_audio)

    def play_audio(self):
        if self.current < 0:
            return
        path = self.scenes[self.current].audio_path
        if not path:
            return
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()

    # ---------- table rendering ----------

    def preview(self, text: str) -> str:
        flat = " ".join(text.split())
        return flat if len(flat) <= PREVIEW_CHARS else flat[:PREVIEW_CHARS] + "…"

    def duration(self, text: str) -> str:
        seconds = round(len(text.split()) / WORDS_PER_MINUTE * 60)
        return f"{seconds // 60}:{seconds % 60:02d}"

    def assets(self, scene: Scene) -> str:
        image = "IMG" if scene.image_path else "·"
        audio = "SND" if scene.audio_path else "·"
        return f"{image} {audio}"

    def refresh_row(self, row: int):
        scene = self.scenes[row]
        for column, value in enumerate(
            [
                str(row + 1),
                self.preview(scene.text) or "(empty)",
                scene.mood,
                self.duration(scene.text),
                self.assets(scene),
            ]
        ):
            item = QTableWidgetItem(value)
            item.setToolTip(scene.text)   # full text on hover, no giant cell
            self.table.setItem(row, column, item)

    def refresh_table(self, keep_row: int = -1):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.scenes))
        for row in range(len(self.scenes)):
            self.refresh_row(row)
        self.table.blockSignals(False)
        if 0 <= keep_row < len(self.scenes):
            self.table.selectRow(keep_row)
        else:
            self.load_detail()
        self.update_progress()

    # ---------- detail panel ----------

    def load_detail(self):
        rows = self.table.selectionModel().selectedRows()
        self.current = rows[0].row() if rows else -1
        enabled = 0 <= self.current < len(self.scenes)

        self.loading = True
        for widget in (self.detail_text, self.detail_image, self.detail_mood):
            widget.setEnabled(enabled)

        if enabled:
            scene = self.scenes[self.current]
            self.detail_text.setPlainText(scene.text)
            self.detail_image.setPlainText(scene.image_prompt)
            self.detail_mood.setText(scene.mood)
            self.load_previews(scene)
        else:
            self.detail_text.clear()
            self.detail_image.clear()
            self.detail_mood.clear()
            self.load_previews(None)
        self.loading = False
        self.refresh_detail_buttons()

    def save_detail(self):
        if self.loading or self.current < 0:
            return
        scene = self.scenes[self.current]
        self.scenes[self.current] = Scene(
            text=self.detail_text.toPlainText(),
            image_prompt=self.detail_image.toPlainText(),
            mood=self.detail_mood.text(),
            image_path=scene.image_path,   # editing text keeps the assets
            audio_path=scene.audio_path,
        )
        self.table.blockSignals(True)
        self.refresh_row(self.current)
        self.table.blockSignals(False)
        self.update_progress()
        self.refresh_detail_buttons()

    # ---------- scene operations ----------

    def show_scenes(self, batch_index: int, scenes: list[Scene]):
        # A capacity retry restarts from batch 1, so clear instead of appending.
        if batch_index == 1:
            self.scenes = []
        self.scenes.extend(scenes)
        self.segmenting = False   # scenes exist now, the bar can go determinate
        self.refresh_table()
        self.table.scrollToBottom()

    def asset_ready(self, row: int, kind: str, path: str, seconds: float):
        if not 0 <= row < len(self.scenes):
            return
        self.scenes[row] = self.scenes[row].model_copy(
            update={f"{kind}_path": path}
        )
        (self.image_times if kind == "image" else self.voice_times).append(seconds)
        self.table.blockSignals(True)
        self.refresh_row(row)
        self.table.blockSignals(False)
        self.update_progress()
        if row == self.current:
            self.load_previews(self.scenes[row])

    def insert_scene(self):
        """Adds an empty scene after the selection, or at the end."""
        row = self.current + 1 if self.current >= 0 else len(self.scenes)
        self.scenes.insert(row, Scene(text="", image_prompt="", mood=""))
        self.refresh_table(keep_row=row)
        self.detail_text.setFocus()

    def delete_scene(self):
        if self.current < 0:
            return
        row = self.current
        del self.scenes[row]
        self.refresh_table(keep_row=min(row, len(self.scenes) - 1))

    def split_scene(self):
        """Breaks the selected scene in two at its midpoint sentence."""
        if self.current < 0:
            self.status.setText("Select a scene to split.")
            return
        row = self.current
        scene = self.scenes[row]
        sentences = split_sentences(scene.text)
        if len(sentences) < 2:
            self.status.setText("This scene has only one sentence.")
            return
        half = len(sentences) // 2
        # Splitting invalidates the assets: they were made for the whole scene.
        self.scenes[row] = Scene(
            text=" ".join(sentences[:half]),
            image_prompt=scene.image_prompt,
            mood=scene.mood,
        )
        self.scenes.insert(
            row + 1,
            Scene(
                text=" ".join(sentences[half:]),
                image_prompt=scene.image_prompt,
                mood=scene.mood,
            ),
        )
        self.refresh_table(keep_row=row + 1)

    def move_scene(self, offset: int):
        if self.current < 0:
            return
        target = self.current + offset
        if not 0 <= target < len(self.scenes):
            return
        row = self.current
        self.scenes[row], self.scenes[target] = (
            self.scenes[target], self.scenes[row],
        )
        self.refresh_table(keep_row=target)

    # ---------- running workers ----------

    def start_worker(self, worker: QObject):
        """Wires the signals every worker shares and starts it on a thread."""
        self.set_busy(True)
        self._stage_text = ""
        self._image_text = ""
        self._voice_text = ""
        self.status.setText("")

        self.thread = QThread()
        self.worker = worker
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.set_stage)
        self.worker.image_progress.connect(self.set_image_status)
        self.worker.voice_progress.connect(self.set_voice_status)
        self.worker.asset_done.connect(self.asset_ready)
        self.worker.finished.connect(self.status.setText)
        self.worker.failed.connect(self.status.setText)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.generation_over)
        self.thread.start()

    def generate(self):
        story = self.story.toPlainText().strip()
        prompt = self.prompt.toPlainText().strip()
        if not story:
            self.status.setText("Paste a story to get started.")
            return
        if not prompt:
            self.status.setText("The prompt can't be empty.")
            return

        self.scenes = []
        self.image_times = []
        self.voice_times = []
        self.segmenting = True
        self.run_id = new_run_id()
        self.refresh_table()

        worker = Generator(story, prompt, self.run_id)
        worker.scenes_ready.connect(self.show_scenes)
        self.start_worker(worker)

    def run_assets(self, image_rows: list[int], voice_rows: list[int]):
        """Renders the given rows' images and/or voice. Both streams run
        one after the other on the one worker."""
        image_jobs = [(row, self.scenes[row]) for row in image_rows]
        voice_jobs = [(row, self.scenes[row]) for row in voice_rows]
        if not image_jobs and not voice_jobs:
            self.status.setText("No scenes to render.")
            return
        if self.run_id is None:
            self.run_id = new_run_id()
        self.start_worker(AssetWorker(image_jobs, voice_jobs, self.run_id))

    def generate_current_image(self):
        if 0 <= self.current < len(self.scenes):
            self.run_assets([self.current], [])

    def generate_current_voice(self):
        if 0 <= self.current < len(self.scenes):
            self.run_assets([], [self.current])

    def generate_missing(self):
        """Fills every scene missing an image and/or audio. Images and
        voice run one after the other."""
        rows_without_image = [
            row for row, scene in enumerate(self.scenes)
            if not scene.image_path and can_render_image(scene)
        ]
        rows_without_audio = [
            row for row, scene in enumerate(self.scenes)
            if not scene.audio_path and can_render_voice(scene)
        ]
        if not rows_without_image and not rows_without_audio:
            self.status.setText("All assets are complete.")
            return
        self.run_assets(rows_without_image, rows_without_audio)

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.status.setText("Cancelling after the current step...")
            self.cancel_button.setEnabled(False)

    def generation_over(self):
        self.segmenting = False
        self.set_busy(False)
        self.update_progress()


if __name__ == "__main__":
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "creepty.app.1"
        )

    qt = QApplication(sys.argv)
    qt.setWindowIcon(QIcon(ICON_PATH))
    qt.aboutToQuit.connect(comfy_server.stop)

    window = Window()
    window.setWindowIcon(QIcon(ICON_PATH))
    window.show()
    sys.exit(qt.exec())
