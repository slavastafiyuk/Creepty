import ctypes
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Qt, QUrl, Slot
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QFormLayout, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QProgressBar, QPushButton, QFileDialog,
    QSplitter, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from pipeline import comfy_server
from pipeline.project import Project, load_project, save_project
from pipeline.scenes import edit_scene, has_asset
from pipeline.video import can_export_video
from pipeline.workers import AssetWorker, Generator, VideoWorker, can_render_image, can_render_voice, new_run_id
from pipeline.segmenter import (
    DEFAULT_SYSTEM, Scene, split_sentences,
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


# ---------- window ----------

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # Keep references alive: a local QThread gets collected mid-run.
        self.worker = None
        self.thread = None
        self._closing = False
        self.project_path: Path | None = None

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
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)

        self.setWindowTitle("Creepty")
        self.resize(1200, 880)
        self.setCentralWidget(self._build())
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.setInterval(500)
        self.autosave_timer.timeout.connect(self.autosave)
        self.story.textChanged.connect(self.schedule_save)
        self.prompt.textChanged.connect(self.schedule_save)
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
        self.open_button = QPushButton("Open project")
        self.open_button.clicked.connect(self.open_project)
        self.save_button = QPushButton("Save project as...")
        self.save_button.clicked.connect(self.save_project_as)
        self.export_button = QPushButton("Export video")
        self.export_button.clicked.connect(self.export_video)
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
        row.addWidget(self.open_button)
        row.addWidget(self.save_button)
        row.addWidget(self.button)
        row.addWidget(self.export_button)
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
            ["#", "Scene", "Mood", "Time", "Assets"]
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
        self.detail_mood.setPlaceholderText("mysterious, fear, dread, passion, tense, calm")

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
            self.open_button, self.reset, self.story, self.prompt,
        ):
            widget.setEnabled(not busy)
        self.refresh_detail_buttons()
        self.refresh_export_button()

    def refresh_detail_buttons(self):
        selected = 0 <= self.current < len(self.scenes)
        scene = self.scenes[self.current] if selected else None
        for widget in (self.detail_text, self.detail_image, self.detail_mood):
            widget.setEnabled(selected and not self.busy)
        self.image_button.setEnabled(
            selected and not self.busy and can_render_image(scene)
        )
        self.voice_button.setEnabled(
            selected and not self.busy and can_render_voice(scene)
        )

    def refresh_export_button(self):
        self.export_button.setEnabled(
            not self.busy and not self._closing and can_export_video(self.scenes)
        )

    # ---------- status line ----------

    @Slot(str)
    def set_stage(self, text: str):
        """A one-off stage message (splitting, starting ComfyUI). Once the
        asset streams start reporting, their combined line takes over."""
        self._stage_text = text
        self._image_text = ""
        self._voice_text = ""
        self.status.setText(text)

    @Slot(str)
    def set_image_status(self, text: str):
        self._image_text = text
        self._refresh_asset_line()

    @Slot(str)
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
        self.refresh_export_button()
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

        images_done = sum(has_asset(scene, "image") for scene in self.scenes)
        voices_done = sum(has_asset(scene, "audio") for scene in self.scenes)
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

        has_audio = has_asset(scene, "audio")
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

    def duration(self, scene: Scene) -> str:
        measured = scene.audio_duration if has_asset(scene, "audio") else None
        seconds = round(measured if measured is not None else len(scene.text.split()) / WORDS_PER_MINUTE * 60)
        return f"{'' if measured is not None else '~'}{seconds // 60}:{seconds % 60:02d}"

    def assets(self, scene: Scene) -> str:
        image = "IMG" if has_asset(scene, "image") else "·"
        audio = "SND" if has_asset(scene, "audio") else "·"
        return f"{image} {audio}"

    def refresh_row(self, row: int):
        scene = self.scenes[row]
        for column, value in enumerate(
            [
                str(row + 1),
                self.preview(scene.text) or "(empty)",
                scene.mood,
                self.duration(scene),
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
        self.load_detail()
        self.update_progress()

    # ---------- detail panel ----------

    def load_detail(self):
        self.player.stop()
        rows = self.table.selectionModel().selectedRows()
        self.current = rows[0].row() if rows else -1
        enabled = 0 <= self.current < len(self.scenes)

        self.loading = True
        for widget in (self.detail_text, self.detail_image, self.detail_mood):
            widget.setEnabled(enabled and not self.busy)

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
        if self.loading or self.busy or self.current < 0:
            return
        scene = self.scenes[self.current]
        self.scenes[self.current] = edit_scene(
            scene, text=self.detail_text.toPlainText(),
            image_prompt=self.detail_image.toPlainText(),
            mood=self.detail_mood.text(),
        )
        self.player.stop()
        self.load_previews(self.scenes[self.current])
        self.schedule_save()
        self.table.blockSignals(True)
        self.refresh_row(self.current)
        self.table.blockSignals(False)
        self.update_progress()
        self.refresh_detail_buttons()

    # ---------- scene operations ----------

    @Slot(int, list)
    def show_scenes(self, batch_index: int, scenes: list[Scene]):
        # Batch 1 starts a new story; retries resume only unfinished text.
        if batch_index == 1:
            self.scenes = []
        self.scenes.extend(scenes)
        self.segmenting = False   # scenes exist now, the bar can go determinate
        self.refresh_table()
        self.table.scrollToBottom()
        self.autosave()

    @Slot(str, int, str, str, float, float)
    def asset_ready(self, scene_id: str, revision: int, kind: str, path: str,
                    seconds: float, duration: float):
        row = next((i for i, scene in enumerate(self.scenes)
                    if scene.id == scene_id and scene.revision == revision), None)
        if row is None or kind not in ("image", "audio"):
            return  # A late result must never attach to revised or replaced text.
        updates = {f"{kind}_path": path}
        if kind == "audio":
            updates["audio_duration"] = duration
        self.scenes[row] = self.scenes[row].model_copy(update=updates)
        self.autosave()
        (self.image_times if kind == "image" else self.voice_times).append(seconds)
        self.table.blockSignals(True)
        self.refresh_row(row)
        self.table.blockSignals(False)
        self.update_progress()
        if row == self.current:
            self.load_previews(self.scenes[row])

    def insert_scene(self):
        """Adds an empty scene after the selection, or at the end."""
        if self.busy or self._closing:
            return
        row = self.current + 1 if self.current >= 0 else len(self.scenes)
        self.scenes.insert(row, Scene(text="", image_prompt="", mood=""))
        self.schedule_save()
        self.refresh_table(keep_row=row)
        self.detail_text.setFocus()

    def delete_scene(self):
        if self.busy or self._closing:
            return
        if self.current < 0:
            return
        row = self.current
        del self.scenes[row]
        self.schedule_save()
        self.refresh_table(keep_row=min(row, len(self.scenes) - 1))

    def split_scene(self):
        """Breaks the selected scene in two at its midpoint sentence."""
        if self.busy or self._closing:
            return
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
        self.schedule_save()
        self.refresh_table(keep_row=row + 1)

    def move_scene(self, offset: int):
        if self.busy or self._closing:
            return
        if self.current < 0:
            return
        target = self.current + offset
        if not 0 <= target < len(self.scenes):
            return
        row = self.current
        self.scenes[row], self.scenes[target] = (
            self.scenes[target], self.scenes[row],
        )
        self.schedule_save()
        self.refresh_table(keep_row=target)

    # ---------- running workers ----------

    def start_worker(self, worker: QObject):
        """Wires the signals every worker shares and starts it on a thread."""
        if self.thread is not None or self._closing:
            return
        self.set_busy(True)
        self._stage_text = ""
        self._image_text = ""
        self._voice_text = ""
        self.status.setText("")

        self.thread = QThread(self)
        self.worker = worker
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run, Qt.ConnectionType.DirectConnection)
        self.worker.progress.connect(self.set_stage)
        if hasattr(self.worker, "image_progress"):
            self.worker.image_progress.connect(self.set_image_status)
        if hasattr(self.worker, "voice_progress"):
            self.worker.voice_progress.connect(self.set_voice_status)
        if hasattr(self.worker, "asset_done"):
            self.worker.asset_done.connect(self.asset_ready)
        self.worker.finished.connect(self.status.setText)
        self.worker.failed.connect(self.status.setText)
        self.worker.finished.connect(self.thread.quit, Qt.ConnectionType.DirectConnection)
        self.worker.failed.connect(self.thread.quit, Qt.ConnectionType.DirectConnection)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.generation_over)
        self.thread.start()

    def generate(self):
        if self.busy or self._closing:
            return
        story = self.story.toPlainText().strip()
        prompt = self.prompt.toPlainText().strip()
        if not story:
            self.status.setText("Paste a story to get started.")
            return
        if not prompt:
            self.status.setText("The prompt can't be empty.")
            return

        if not self.autosave():
            return
        run_id = new_run_id()
        target = Path(__file__).parent / "output" / run_id / "project.json"
        try:
            save_project(target, Project(run_id=run_id, story=story, system_prompt=prompt))
        except (OSError, ValueError) as error:
            self.status.setText(f"New project could not be saved: {error}")
            return
        self.player.stop()
        self.scenes = []
        self.project_path = target
        self.image_times = []
        self.voice_times = []
        self.segmenting = True
        self.run_id = run_id
        self.refresh_table()

        worker = Generator(story, prompt, self.run_id)
        worker.scenes_ready.connect(self.show_scenes)
        self.start_worker(worker)

    def run_assets(self, image_rows: list[int], voice_rows: list[int]):
        """Renders the given rows' images and/or voice. Both streams run
        one after the other on the one worker."""
        if self.busy or self._closing:
            return
        image_jobs = [(row, self.scenes[row]) for row in image_rows]
        voice_jobs = [(row, self.scenes[row]) for row in voice_rows]
        if not image_jobs and not voice_jobs:
            self.status.setText("No scenes to render.")
            return
        if not self.autosave():
            return
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
            if not has_asset(scene, "image") and can_render_image(scene)
        ]
        rows_without_audio = [
            row for row, scene in enumerate(self.scenes)
            if not has_asset(scene, "audio") and can_render_voice(scene)
        ]
        if not rows_without_image and not rows_without_audio:
            complete = self.scenes and all(has_asset(scene, kind) for scene in self.scenes for kind in ("image", "audio"))
            self.status.setText("All assets are complete." if complete else "Add narration text and image prompts to render missing assets.")
            return
        self.run_assets(rows_without_image, rows_without_audio)

    def export_video(self):
        if self.busy or self._closing:
            return
        if not can_export_video(self.scenes):
            self.status.setText("Generate all images and narration before exporting.")
            return
        if not self.autosave():
            return

        self.player.stop()
        directory = self.project_path.parent if self.project_path else Path(__file__).parent / "output" / self.run_id
        path, _ = QFileDialog.getSaveFileName(
            self, "Export video", str(directory / "video.mp4"), "MP4 video (*.mp4)"
        )
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"

        self.start_worker(VideoWorker(self.scenes, path))

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.status.setText("Cancelling after the current step...")
            self.cancel_button.setEnabled(False)

    @Slot()
    def generation_over(self):
        thread = self.thread
        if thread is None:
            return
        # finished can arrive before native thread-local cleanup has completed.
        # Never join a live thread while the GUI holds Python's GIL.
        if not thread.wait(0):
            QTimer.singleShot(10, self.generation_over)
            return
        self.worker = None
        self.thread = None
        thread.deleteLater()
        self.segmenting = False
        self.set_busy(self._closing)
        self.update_progress()
        self.autosave()
        if self._closing:
            QTimer.singleShot(0, self.close)

    # ---------- project persistence and shutdown ----------

    def schedule_save(self):
        if not self.loading:
            self.autosave_timer.start()

    def project(self) -> Project:
        if self.run_id is None:
            self.run_id = new_run_id()
        return Project(run_id=self.run_id, story=self.story.toPlainText(),
                       system_prompt=self.prompt.toPlainText(), scenes=self.scenes)

    def autosave(self) -> bool:
        self.autosave_timer.stop()
        if not self.scenes and not self.story.toPlainText().strip() and self.run_id is None:
            return True
        try:
            project = self.project()
            if self.project_path is None:
                self.project_path = Path(__file__).parent / "output" / project.run_id / "project.json"
            save_project(self.project_path, project)
            return True
        except (OSError, ValueError) as error:
            self.status.setText(f"Project could not be saved: {error}")
            return False

    def save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save project", str(self.project_path or "project.json"), "Creepty project (*.json)")
        if not path:
            return
        try:
            target = Path(path)
            save_project(target, self.project())
            self.project_path = target.resolve()
            self.autosave_timer.stop()
            self.status.setText(f"Project saved: {self.project_path}")
        except (OSError, ValueError) as error:
            self.status.setText(f"Project could not be saved: {error}")

    def open_project(self):
        if self.busy or self._closing:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", str(Path(__file__).parent / "output"), "Creepty project (*.json)")
        if not path or not self.autosave():
            return
        try:
            project, warnings = load_project(Path(path))
        except (OSError, ValueError) as error:
            self.status.setText(f"Project could not be opened: {error}")
            return
        self.player.stop()
        self.loading = True
        self.story.setPlainText(project.story)
        self.prompt.setPlainText(project.system_prompt)
        self.loading = False
        self.run_id = project.run_id
        self.project_path = Path(path).resolve()
        self.scenes = project.scenes
        self.image_times, self.voice_times = [], []
        self.refresh_table(keep_row=0)
        message = f"Project opened: {self.project_path}"
        if warnings:
            message += f"; {len(warnings)} missing or invalid assets marked for regeneration."
        self.status.setText(message)
        self.status.setToolTip("\n".join(warnings))

    def closeEvent(self, event):
        self.player.stop()
        if self.thread is not None:
            event.ignore()
            self._closing = True
            self.cancel()
            self.set_busy(True)
            self.save_button.setEnabled(False)
            self.status.setText("Closing after the current step and GPU cleanup...")
            return
        if not self.autosave():
            event.ignore()
            self._closing = False
            self.set_busy(False)
            self.save_button.setEnabled(True)
            return
        event.accept()


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