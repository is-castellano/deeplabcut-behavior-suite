
import sys
import os
import json
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import seaborn as sns
from scipy.signal import savgol_filter
from shapely.geometry import Point, Polygon, LineString, box
from shapely.affinity import scale as scale_geometry
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QVBoxLayout, QWidget, QFileDialog, QLabel, 
                             QFrame, QMessageBox, QInputDialog,QLineEdit, QHBoxLayout,
                             QDialog, QSizePolicy, QScrollBar, QSlider)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import traceback
from shapely.vectorized import contains

# --- SCIENTIFIC LOGIC ---
def clean_behavior(series, fps, gap_sec, dur_sec):
    s = pd.Series(series).astype(float)
    filled = s.rolling(window=int(gap_sec * fps), min_periods=1, center=True).max()
    cleaned = filled.rolling(window=int(dur_sec * fps), min_periods=1, center=True).min()
    return cleaned.fillna(0).astype(bool).values


def fill_internal_gaps(series, max_gap_frames):
    arr = np.asarray(series, dtype=bool).copy()
    if max_gap_frames <= 0 or not np.any(arr):
        return arr

    true_idx = np.flatnonzero(arr)
    start = true_idx[0]
    end = true_idx[-1]
    i = start
    while i <= end:
        if arr[i]:
            i += 1
            continue
        gap_start = i
        while i <= end and not arr[i]:
            i += 1
        gap_end = i - 1
        if gap_start > start and gap_end < end and (gap_end - gap_start + 1) <= max_gap_frames:
            arr[gap_start:gap_end + 1] = True
    return arr


def extend_sniff_onsets(sniff_series, candidate_series, max_pre_frames):
    sniff = np.asarray(sniff_series, dtype=bool).copy()
    candidates = np.asarray(candidate_series, dtype=bool)
    if max_pre_frames <= 0 or not np.any(sniff):
        return sniff

    starts = np.flatnonzero(sniff & np.r_[True, ~sniff[:-1]])
    for start in starts:
        pre_start = max(0, start - max_pre_frames)
        i = start - 1
        while i >= pre_start and candidates[i]:
            sniff[i] = True
            i -= 1
    return sniff


def select_roi_foreground(window_name, frame):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)
    roi = cv2.selectROI(window_name, frame, False)
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass
    return roi


def video_review_dir(video_path):
    video_folder = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    review_dir = os.path.join(video_folder, f"{video_name}_DLC_review")
    os.makedirs(review_dir, exist_ok=True)
    return review_dir


def video_review_subdir(video_path, subfolder):
    path = os.path.join(video_review_dir(video_path), subfolder)
    os.makedirs(path, exist_ok=True)
    return path


class VideoTimelinePlayer(QDialog):
    def __init__(
        self,
        video_path,
        behavior_frames=None,
        fps=30,
        parent=None,
        frame_csv_path=None,
        roi_exp=None,
        roi_ctrl=None,
        roi_exp_df=None,
        roi_ctrl_df=None
    ):
        super().__init__(parent)
        self.setWindowTitle("Video Segment Player")
        self.setMinimumSize(950, 650)

        self.video_path = video_path
        self.frame_csv_path = frame_csv_path
        self.roi_exp = roi_exp
        self.roi_ctrl = roi_ctrl
        self.roi_exp_df = roi_exp_df.copy() if roi_exp_df is not None else None
        self.roi_ctrl_df = roi_ctrl_df.copy() if roi_ctrl_df is not None else None
        self.modified = False
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video.")

        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(video_fps) if video_fps and video_fps > 0 else float(fps)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.behavior_frames = self._clean_frames(behavior_frames)
        self.play_frames = self.behavior_frames.copy()
        if len(self.play_frames) == 0 and self.total_frames > 0:
            self.play_frames = np.arange(self.total_frames, dtype=int)

        self.play_index = 0
        self.selected_start = int(self.play_frames[0]) if len(self.play_frames) else 0
        self.selected_end = int(self.play_frames[-1]) if len(self.play_frames) else 0
        self.max_frame = self.total_frames - 1 if self.total_frames > 0 else max(self.selected_end, 1)
        self.visible_start = 0
        self.visible_end = self.max_frame
        self.zoom_history = []

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._show_next_frame)

        self._build_ui()
        self._draw_timeline()
        self._show_frame(self.selected_start)

    def _roi_for_frame(self, roi, roi_df, frame_number):
        if roi_df is not None and not roi_df.empty:
            required_cols = ["frame", "x", "y", "width", "height"]
            if all(col in roi_df.columns for col in required_cols):
                rows = roi_df.loc[roi_df["frame"].astype(int) == int(frame_number)]
                if not rows.empty:
                    row = rows.iloc[-1]
                    values = [row["x"], row["y"], row["width"], row["height"]]
                    if not pd.isna(values).any():
                        return tuple(int(round(float(v))) for v in values)

        if roi:
            return tuple(int(round(float(v))) for v in roi)
        return None

    def _draw_roi_overlay(self, frame, frame_number):
        overlays = [
            (self._roi_for_frame(self.roi_exp, self.roi_exp_df, frame_number), (255, 0, 0), "EXP"),
            (self._roi_for_frame(self.roi_ctrl, self.roi_ctrl_df, frame_number), (0, 255, 255), "CTRL"),
        ]
        for roi, color, label in overlays:
            if not roi:
                continue
            x, y, w, h = roi
            if w <= 0 or h <= 0:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                label,
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    def _clean_frames(self, frames):
        if frames is None:
            return np.array([], dtype=int)
        arr = np.asarray(frames, dtype=int)
        arr = arr[arr >= 0]
        if self.total_frames > 0:
            arr = arr[arr < self.total_frames]
        return np.unique(arr)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.fig = Figure(figsize=(9, 2.0), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.timeline_canvas = FigureCanvas(self.fig)
        self.timeline_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.timeline_canvas.setFixedHeight(170)
        layout.addWidget(self.timeline_canvas)

        timeline_controls = QHBoxLayout()
        self.btn_zoom_in = QPushButton("Zoom in selection")
        self.btn_zoom_in.clicked.connect(self.zoom_in_selection)

        self.btn_zoom_out = QPushButton("Zoom out")
        self.btn_zoom_out.clicked.connect(self.zoom_out)

        self.timeline_scroll = QScrollBar(Qt.Orientation.Horizontal)
        self.timeline_scroll.valueChanged.connect(self._on_scroll_changed)

        timeline_controls.addWidget(self.btn_zoom_in)
        timeline_controls.addWidget(self.btn_zoom_out)
        timeline_controls.addWidget(QLabel("Scroll"))
        timeline_controls.addWidget(self.timeline_scroll, stretch=1)
        layout.addLayout(timeline_controls)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.video_label, stretch=1)

        controls = QHBoxLayout()
        self.lbl_selection = QLabel()
        self.lbl_selection.setStyleSheet("color: #2c3e50; font-weight: bold;")

        self.btn_play = QPushButton("Play selected frames")
        self.btn_play.clicked.connect(self.play_selected_frames)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.pause)

        self.btn_toggle_behavior = QPushButton("Toggle behavior in selection")
        self.btn_toggle_behavior.clicked.connect(self.toggle_behavior_in_selection)

        self.btn_save_csv = QPushButton("Save CSV")
        self.btn_save_csv.clicked.connect(self.save_behavior_csv)

        controls.addWidget(self.lbl_selection, stretch=1)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_pause)
        controls.addWidget(self.btn_toggle_behavior)
        controls.addWidget(self.btn_save_csv)
        layout.addLayout(controls)

    def _draw_timeline(self):
        self.ax.clear()
        self.ax.set_xlim(self.visible_start, self.visible_end)
        self.ax.set_ylim(0, 1)
        self.ax.set_yticks([])
        self.ax.set_xlabel("Frame")
        self.ax.set_title(f"Drag on the timeline to select frames")

        if len(self.behavior_frames):
            self.ax.vlines(
                self.behavior_frames,
                0.18,
                0.82,
                color="#c0392b",
                linewidth=0.8,
                alpha=0.65,
                )
            

        self.selection_patch = self.ax.axvspan(
            self.selected_start,
            self.selected_end,
            color="#2980b9",
            alpha=0.20
        )
        self.current_line = self.ax.axvline(
            self.selected_start,
            color="#27ae60",
            linewidth=2
        )
        self.span_selector = SpanSelector(
            self.ax,
            self._on_span_selected,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.25, facecolor="#2980b9"),
            interactive=True
        )
        self._update_selection_label()
        self._update_scrollbar()
        self.timeline_canvas.draw_idle()

    def _on_span_selected(self, xmin, xmax):
        start, end = sorted((int(round(xmin)), int(round(xmax))))
        if self.total_frames > 0:
            start = max(0, min(start, self.total_frames - 1))
            end = max(0, min(end, self.total_frames - 1))
        if end < start:
            return

        self.selected_start = start
        self.selected_end = end
        self.play_frames = np.arange(start, end + 1, dtype=int)
        self.play_index = 0
        self._update_selection_patch()
        self._show_frame(start)
        self.play_selected_frames()

    def _visible_frame_count(self):
        return max(1, self.visible_end - self.visible_start + 1)

    def _set_timeline_view(self, start):
        visible_count = self._visible_frame_count()
        max_start = max(0, self.max_frame - visible_count + 1)
        self.visible_start = int(max(0, min(start, max_start)))
        self.visible_end = int(min(self.max_frame, self.visible_start + visible_count - 1))
        self.ax.set_xlim(self.visible_start, self.visible_end)
        self._update_scrollbar()
        self.timeline_canvas.draw_idle()

    def _update_scrollbar(self):
        if not hasattr(self, "timeline_scroll"):
            return

        visible_count = self._visible_frame_count()
        max_start = max(0, self.max_frame - visible_count + 1)
        self.timeline_scroll.blockSignals(True)
        self.timeline_scroll.setEnabled(max_start > 0)
        self.timeline_scroll.setMinimum(0)
        self.timeline_scroll.setMaximum(max_start)
        self.timeline_scroll.setPageStep(visible_count)
        self.timeline_scroll.setSingleStep(max(1, visible_count // 20))
        self.timeline_scroll.setValue(min(self.visible_start, max_start))
        self.timeline_scroll.blockSignals(False)

    def _on_scroll_changed(self, value):
        self._set_timeline_view(value)

    def zoom_in_selection(self):
        if self.selected_end <= self.selected_start:
            QMessageBox.warning(self, "Zoom", "Select at least two frames with the timeline first.")
            return

        self.zoom_history.append((self.visible_start, self.visible_end))
        self.visible_start = max(0, self.selected_start)
        self.visible_end = min(self.max_frame, self.selected_end)
        self.ax.set_xlim(self.visible_start, self.visible_end)
        self._update_scrollbar()
        self.timeline_canvas.draw_idle()

    def zoom_out(self):
        if self.zoom_history:
            self.visible_start, self.visible_end = self.zoom_history.pop()
        else:
            self.visible_start = 0
            self.visible_end = self.max_frame

        self.ax.set_xlim(self.visible_start, self.visible_end)
        self._update_scrollbar()
        self.timeline_canvas.draw_idle()

    def _ensure_frame_visible(self, frame_number):
        if self.visible_start <= frame_number <= self.visible_end:
            return

        visible_count = self._visible_frame_count()
        half_window = max(0, visible_count // 2)
        self._set_timeline_view(frame_number - half_window)

    def _update_selection_patch(self):
        self.selection_patch.set_xy([
            [self.selected_start, 0],
            [self.selected_start, 1],
            [self.selected_end, 1],
            [self.selected_end, 0],
            [self.selected_start, 0],
        ])
        self._update_selection_label()
        self.timeline_canvas.draw_idle()

    def _update_selection_label(self):
        count = max(0, self.selected_end - self.selected_start + 1)
        behavior_count = len(self.behavior_frames)
        modified_marker = " *modified*" if self.modified else ""
        self.lbl_selection.setText(
            f"Selection: {self.selected_start} - {self.selected_end} ({count} frames) | "
            f"Behavior frames: {behavior_count}{modified_marker}"
        )

    def toggle_behavior_in_selection(self):
        if self.selected_end < self.selected_start:
            return

        selected_frames = np.arange(self.selected_start, self.selected_end + 1, dtype=int)
        selected_set = set(selected_frames.tolist())
        behavior_set = set(self.behavior_frames.tolist())

        if behavior_set.intersection(selected_set):
            updated_frames = [frame for frame in self.behavior_frames if frame not in selected_set]
            action = "removed"
        else:
            updated_frames = sorted(behavior_set.union(selected_set))
            action = "added"

        self.behavior_frames = self._clean_frames(updated_frames)
        self.modified = True
        self._draw_timeline()
        self._show_frame(self.selected_start)
        QMessageBox.information(
            self,
            "Behavior updated",
            f"Behavior frames {action} for selection {self.selected_start}-{self.selected_end}."
        )

    def save_behavior_csv(self):
        default_path = self.frame_csv_path or os.path.join(
            video_review_subdir(self.video_path, "Behavior"),
            "behavior_frames_modified.csv"
        )
        file, _ = QFileDialog.getSaveFileName(
            self,
            "Save behavior CSV",
            default_path,
            "CSV Files (*.csv)"
        )
        if not file:
            return

        try:
            out_df = pd.DataFrame({"frame": self.behavior_frames.astype(int)})

            if self.frame_csv_path and os.path.exists(self.frame_csv_path):
                original_df = pd.read_csv(self.frame_csv_path, sep=";")
                unnamed_cols = [col for col in original_df.columns if str(col).startswith("Unnamed")]
                if unnamed_cols:
                    original_df = original_df.drop(columns=unnamed_cols)

                if "frame" in original_df.columns:
                    original_df["frame"] = pd.to_numeric(original_df["frame"], errors="coerce")
                    original_df = original_df.dropna(subset=["frame"])
                    original_df["frame"] = original_df["frame"].astype(int)
                    original_df = original_df.drop_duplicates(subset=["frame"], keep="first")
                    out_df = out_df.merge(original_df, on="frame", how="left")

            out_df.to_csv(file, sep=";", index=False)
            self.frame_csv_path = file
            self.modified = False
            self._update_selection_label()
            QMessageBox.information(self, "Saved", f"Saved {len(out_df)} behavior frames.")
        except Exception:
            QMessageBox.critical(self, "Error", traceback.format_exc())

    def play_selected_frames(self):
        if len(self.play_frames) == 0:
            return
        self.play_index = 0
        interval_ms = max(1, int(1000 / self.fps))
        self.timer.start(interval_ms)

    def pause(self):
        self.timer.stop()

    def _show_next_frame(self):
        if self.play_index >= len(self.play_frames):
            self.timer.stop()
            return
        frame_number = int(self.play_frames[self.play_index])
        self._show_frame(frame_number)
        self.play_index += 1

    def _show_frame(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ret, frame = self.cap.read()
        if not ret:
            return

        self._draw_roi_overlay(frame, frame_number)

        cv2.putText(
            frame,
            f"Frame: {int(frame_number)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy()).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)
        self.current_line.set_xdata([frame_number, frame_number])
        self._ensure_frame_visible(frame_number)
        self.timeline_canvas.draw_idle()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        current = int(self.current_line.get_xdata()[0]) if hasattr(self, "current_line") else self.selected_start
        self._show_frame(current)

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        super().closeEvent(event)


class DynamicROIEditor(QDialog):
    def __init__(self, video_path, roi_name="exp", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Dynamic {roi_name.upper()} ROI Editor")
        self.setMinimumSize(950, 700)

        self.video_path = video_path
        self.roi_name = roi_name
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video.")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.total_frames <= 0:
            raise ValueError("Unable to read video frame count.")

        self.current_frame = 0
        self.keyframes = {}
        self.pending_roi = None
        self.pending_roi_frame = None
        self.roi_df = self._empty_roi_dataframe()
        self.saved_csv_path = os.path.join(
            video_review_subdir(self.video_path, "ROI"),
            f"{self.roi_name}_roi_by_frame.csv"
        )

        self._build_ui()
        self._show_frame(0)

    def _empty_roi_dataframe(self):
        return pd.DataFrame({
            "frame": np.arange(self.total_frames, dtype=int),
            "x": np.nan,
            "y": np.nan,
            "width": np.nan,
            "height": np.nan,
        })

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: #2c3e50; font-weight: bold;")
        layout.addWidget(self.info_label)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.video_label, stretch=1)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.total_frames - 1)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        frame_row = QHBoxLayout()
        self.frame_input = QLineEdit()
        self.frame_input.setPlaceholderText("Frame")
        self.frame_input.returnPressed.connect(self.go_to_frame_from_input)

        self.btn_prev = QPushButton("Previous")
        self.btn_prev.clicked.connect(lambda: self.go_to_frame(self.current_frame - 1))

        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(lambda: self.go_to_frame(self.current_frame + 1))

        self.btn_select_roi = QPushButton("Select ROI at current frame")
        self.btn_select_roi.clicked.connect(self.select_roi_current_frame)

        self.btn_delete_roi = QPushButton("Delete keyframe")
        self.btn_delete_roi.clicked.connect(self.delete_current_keyframe)

        frame_row.addWidget(QLabel("Frame"))
        frame_row.addWidget(self.frame_input)
        frame_row.addWidget(self.btn_prev)
        frame_row.addWidget(self.btn_next)
        frame_row.addWidget(self.btn_select_roi)
        frame_row.addWidget(self.btn_delete_roi)
        layout.addLayout(frame_row)

        save_row = QHBoxLayout()
        self.btn_open = QPushButton("Open CSV")
        self.btn_open.clicked.connect(self.open_csv)

        self.btn_accept = QPushButton("Use this ROI")
        self.btn_accept.clicked.connect(self.use_pending_roi_interval)

        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self.save_csv_default)

        self.btn_finish = QPushButton("Close")
        self.btn_finish.clicked.connect(self.accept_roi)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        save_row.addWidget(self.btn_open)
        save_row.addStretch(1)
        save_row.addWidget(self.btn_accept)
        save_row.addWidget(self.btn_save)
        save_row.addWidget(self.btn_finish)
        save_row.addWidget(self.btn_cancel)
        layout.addLayout(save_row)

    def _on_slider_changed(self, value):
        self._show_frame(int(value))

    def go_to_frame(self, frame_number):
        frame_number = int(max(0, min(frame_number, self.total_frames - 1)))
        if self.slider.value() != frame_number:
            self.slider.setValue(frame_number)
        else:
            self._show_frame(frame_number)

    def go_to_frame_from_input(self):
        try:
            self.go_to_frame(int(self.frame_input.text()))
        except ValueError:
            QMessageBox.warning(self, "Frame", "Enter a valid frame number.")

    def _read_frame(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def _roi_for_frame(self, frame_number):
        if self.roi_df.empty:
            return None
        row = self.roi_df.loc[self.roi_df["frame"] == int(frame_number)]
        if row.empty:
            return None
        row = row.iloc[0]
        values = [row["x"], row["y"], row["width"], row["height"]]
        if pd.isna(values).any():
            return None
        return tuple(int(round(v)) for v in values)

    def _show_frame(self, frame_number):
        frame_number = int(max(0, min(frame_number, self.total_frames - 1)))
        frame = self._read_frame(frame_number)
        if frame is None:
            return

        self.current_frame = frame_number
        self.frame_input.setText(str(frame_number))

        roi = self._roi_for_frame(frame_number)
        is_pending_frame = self.pending_roi_frame == frame_number and self.pending_roi is not None
        if roi is None and is_pending_frame:
            roi = self.pending_roi
        if roi is not None:
            x, y, w, h = roi
            color = (0, 255, 0) if frame_number in self.keyframes else (255, 180, 0)
            if is_pending_frame:
                color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy()).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)
        self._update_info_label()

    def _update_info_label(self):
        exact = "assigned" if self.current_frame in self.keyframes else "empty/interpolated on save"
        if self.pending_roi_frame == self.current_frame and self.pending_roi is not None:
            exact = "selected, not applied"
        if not self.keyframes and self.pending_roi is None:
            exact = "no ROI"
        pending = ""
        if self.pending_roi_frame is not None:
            pending = f" | Pending start: {self.pending_roi_frame}"
        self.info_label.setText(
            f"{self.roi_name.upper()} ROI | Frame {self.current_frame}/{self.total_frames - 1} | "
            f"Assigned frames: {len(self.keyframes)} | Current ROI: {exact}{pending}"
        )

    def select_roi_current_frame(self):
        frame = self._read_frame(self.current_frame)
        if frame is None:
            QMessageBox.warning(self, "ROI", "Unable to read current frame.")
            return

        roi = select_roi_foreground(f"Select {self.roi_name.upper()} ROI", frame)
        cv2.destroyAllWindows()
        if roi == (0, 0, 0, 0):
            return

        self.pending_roi = tuple(int(v) for v in roi)
        self.pending_roi_frame = int(self.current_frame)
        self._show_frame(self.current_frame)

    def delete_current_keyframe(self):
        if self.current_frame not in self.keyframes:
            QMessageBox.information(self, "ROI", "No ROI is assigned at the current frame.")
            return
        del self.keyframes[self.current_frame]
        self.roi_df.loc[
            self.roi_df["frame"] == self.current_frame,
            ["x", "y", "width", "height"]
        ] = np.nan
        self._show_frame(self.current_frame)

    def use_pending_roi_interval(self):
        if self.pending_roi is None or self.pending_roi_frame is None:
            QMessageBox.warning(self, "ROI", "Select a ROI before applying it to an interval.")
            return

        start, end = sorted((int(self.pending_roi_frame), int(self.current_frame)))
        x, y, w, h = self.pending_roi

        assigned_before = sorted(frame for frame in self.keyframes if frame < start)
        if assigned_before:
            prev_frame = assigned_before[-1]
            prev_roi = self.keyframes[prev_frame]
            gap_frames = np.arange(prev_frame + 1, start, dtype=int)
            if len(gap_frames):
                interp_values = np.column_stack([
                    np.linspace(prev_roi[col_idx], self.pending_roi[col_idx], len(gap_frames) + 2)[1:-1]
                    for col_idx in range(4)
                ])
                for frame_number, values in zip(gap_frames, interp_values):
                    roi_tuple = tuple(float(v) for v in values)
                    self.keyframes[int(frame_number)] = roi_tuple
                    self.roi_df.loc[
                        self.roi_df["frame"] == int(frame_number),
                        ["x", "y", "width", "height"]
                    ] = list(roi_tuple)
        elif start > 0:
            leading_mask = self.roi_df["frame"].between(0, start - 1)
            self.roi_df.loc[leading_mask, ["x", "y", "width", "height"]] = [x, y, w, h]
            for frame_number in range(0, start):
                self.keyframes[frame_number] = self.pending_roi

        mask = self.roi_df["frame"].between(start, end)
        self.roi_df.loc[mask, ["x", "y", "width", "height"]] = [x, y, w, h]

        for frame_number in range(start, end + 1):
            self.keyframes[frame_number] = self.pending_roi

        self.pending_roi = None
        self.pending_roi_frame = None
        self._show_frame(self.current_frame)
        QMessageBox.information(
            self,
            "ROI interval",
            f"Applied ROI to frames {start}-{end}."
        )

    def _final_roi_dataframe(self):
        df = self.roi_df.copy()
        for frame_number, roi in self.keyframes.items():
            x, y, w, h = roi
            df.loc[df["frame"] == frame_number, ["x", "y", "width", "height"]] = [x, y, w, h]

        if self.keyframes:
            cols = ["x", "y", "width", "height"]
            df[cols] = df[cols].interpolate(method="linear").ffill().bfill()

        return df

    def _save_csv(self, path):
        if not self.keyframes:
            raise ValueError("Apply at least one ROI interval before saving.")
        self.roi_df = self._final_roi_dataframe()
        self.roi_df.to_csv(path, sep=";", index=False)
        self.saved_csv_path = path

    def open_csv(self):
        start_dir = os.path.dirname(self.saved_csv_path)
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Open ROI CSV",
            start_dir,
            "CSV Files (*.csv)"
        )
        if not file:
            return
        try:
            loaded = pd.read_csv(file, sep=";")
            required_cols = ["frame", "x", "y", "width", "height"]
            missing = [col for col in required_cols if col not in loaded.columns]
            if missing:
                raise ValueError(f"Missing columns: {', '.join(missing)}")

            loaded = loaded[required_cols].copy()
            for col in required_cols:
                loaded[col] = pd.to_numeric(loaded[col], errors="coerce")

            loaded = loaded.dropna(subset=["frame"])
            loaded["frame"] = loaded["frame"].astype(int)
            loaded = loaded[
                (loaded["frame"] >= 0) &
                (loaded["frame"] < self.total_frames)
            ]
            loaded = loaded.drop_duplicates(subset=["frame"], keep="last")

            df = self._empty_roi_dataframe()
            df = df.set_index("frame")
            loaded = loaded.set_index("frame")
            df.loc[loaded.index, ["x", "y", "width", "height"]] = loaded[["x", "y", "width", "height"]]
            df = df.reset_index()

            cols = ["x", "y", "width", "height"]
            if df[cols].notna().any(axis=None):
                df[cols] = df[cols].interpolate(method="linear").ffill().bfill()

            self.roi_df = df
            self.keyframes = {}
            assigned = self.roi_df.dropna(subset=cols)
            for _, row in assigned.iterrows():
                self.keyframes[int(row["frame"])] = (
                    float(row["x"]),
                    float(row["y"]),
                    float(row["width"]),
                    float(row["height"])
                )
            self.pending_roi = None
            self.pending_roi_frame = None
            self.saved_csv_path = file
            self._show_frame(self.current_frame)
            QMessageBox.information(self, "Loaded", f"Loaded ROI data for {len(self.roi_df)} frames.")
        except Exception as exc:
            QMessageBox.warning(self, "ROI", str(exc))

    def save_csv_default(self):
        file, _ = QFileDialog.getSaveFileName(
            self,
            "Save ROI CSV",
            self.saved_csv_path,
            "CSV Files (*.csv)"
        )
        if not file:
            return
        try:
            self._save_csv(file)
            QMessageBox.information(self, "Saved", f"Saved ROI data for {len(self.roi_df)} frames.")
        except Exception as exc:
            QMessageBox.warning(self, "ROI", str(exc))

    def accept_roi(self):
        try:
            if not self.keyframes:
                raise ValueError("Apply or load at least one ROI interval before closing.")
            self.roi_df = self._final_roi_dataframe()
            self.roi_df.to_csv(self.saved_csv_path, sep=";", index=False)
            self.accept()
        except Exception as exc:
            QMessageBox.warning(self, "ROI", str(exc))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "current_frame"):
            self._show_frame(self.current_frame)

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)


class DLCLabSuiteFinal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLC Behavior Analytics - Bovetti's Lab Edition")
        self.setMinimumSize(600, 700)
        self.setStyleSheet("background-color: #fcfcfc;")

        self.video_path, self.csv_path, self.frame_csv_path = None, None, None
        self.roi_exp, self.roi_ctrl, self.arena_poly = None, None, None
        self.roi_exp_df, self.roi_ctrl_df = None, None
        self.roi_exp_csv_path, self.roi_ctrl_csv_path = None, None
        self.video_players = []
        self.start_frame = 0 
        self.FPS, self.THRESHOLD_THIGMO_CM = 30, 5.0
        self.CAGE_LONG_CM, self.CAGE_SHORT_CM = 53.0, 24.0
        self.CM_PER_PIXEL = 0.05
        self.CM_PER_PIXEL_X, self.CM_PER_PIXEL_Y = self.CM_PER_PIXEL, self.CM_PER_PIXEL
        self.SNIFF_DISTANCE_CM = 3.0
        self.SNIFF_MAX_ANGLE_DEG = 70.0
        self.SNIFF_DIRECTION_FREE_CM = 1.0
        self.SNIFF_BODY_OVERLAP_MARGIN_CM = 0.5
        self.SNIFF_BODYPOINT_MARGIN_CM = 2.0
        self.SNIFF_BODYPOINT_MIN_POINTS = 2
        self.SNIFF_NOSE_CONTACT_MARGIN_CM = 0.8
        self.SNIFF_NOSE_CONTACT_LIKELIHOOD = 0.35
        self.SNIFF_NEAR_DISTANCE_CM = 3 
        self.SNIFF_NEAR_MAX_ANGLE_DEG = 100.0
        self.SNIFF_GAP_SEC = 0.05
        self.SNIFF_GAP_FILL_FRAMES = 10
        self.SNIFF_PRE_CONTACT_FRAMES = 0 
        self.SNIFF_PRE_CONTACT_DISTANCE_CM = 0
        self.SNIFF_PRE_CONTACT_MAX_ANGLE_DEG = 115.0
        self.SNIFF_CANDIDATE_DISTANCE_CM = 7.0
        self.SNIFF_CANDIDATE_MAX_ANGLE_DEG = 140.0
        self.SNIFF_CANDIDATE_MIN_LIKELIHOOD = 0.25
        self.SNIFF_MIN_DURATION_SEC = 0.2

        self.init_ui()
        self.update_cage_label()

    def calibrate_cm_per_pixel_from_arena(self):
        if self.arena_poly is None:
            return

        minx, miny, maxx, maxy = self.arena_poly.bounds
        arena_width_px = maxx - minx
        arena_height_px = maxy - miny
        if arena_width_px <= 0 or arena_height_px <= 0:
            return

        if arena_width_px >= arena_height_px:
            real_width_cm, real_height_cm = self.CAGE_LONG_CM, self.CAGE_SHORT_CM
        else:
            real_width_cm, real_height_cm = self.CAGE_SHORT_CM, self.CAGE_LONG_CM

        self.CM_PER_PIXEL_X = real_width_cm / arena_width_px
        self.CM_PER_PIXEL_Y = real_height_cm / arena_height_px
        self.CM_PER_PIXEL = (self.CM_PER_PIXEL_X + self.CM_PER_PIXEL_Y) / 2.0

    def update_cage_label(self):
        if hasattr(self, "lbl_cage"):
            self.lbl_cage.setText(
                f"CAGE FORMAT: {self.CAGE_LONG_CM:.1f} x {self.CAGE_SHORT_CM:.1f} cm | "
                f"scale X={self.CM_PER_PIXEL_X:.5f}, Y={self.CM_PER_PIXEL_Y:.5f} cm/px"
            )

    def configure_cage_format(self):
        long_cm, ok = QInputDialog.getDouble(
            self,
            "Cage format",
            "Long side (cm):",
            self.CAGE_LONG_CM,
            1.0,
            500.0,
            2
        )
        if not ok:
            return

        short_cm, ok = QInputDialog.getDouble(
            self,
            "Cage format",
            "Short side (cm):",
            self.CAGE_SHORT_CM,
            1.0,
            500.0,
            2
        )
        if not ok:
            return

        self.CAGE_LONG_CM = max(long_cm, short_cm)
        self.CAGE_SHORT_CM = min(long_cm, short_cm)
        self.calibrate_cm_per_pixel_from_arena()
        self.update_cage_label()

    def distance_cm(self, x1, y1, x2, y2):
        return np.sqrt(
            ((x1 - x2) * self.CM_PER_PIXEL_X) ** 2
            + ((y1 - y2) * self.CM_PER_PIXEL_Y) ** 2
        )

    def diff_distance_cm(self, dx, dy):
        return np.sqrt((dx * self.CM_PER_PIXEL_X) ** 2 + (dy * self.CM_PER_PIXEL_Y) ** 2)

    def create_btn(self, text):
        btn = QPushButton(text)
        btn.setStyleSheet("color: black; background-color: white; border: 1px solid #ccc; padding: 12px; font-weight: bold; border-radius: 5px;")
        return btn

    def init_ui(self):
        central = QWidget(); layout = QVBoxLayout(central)
        layout.setContentsMargins(50, 40, 50, 40); layout.setSpacing(15)
        
        title = QLabel("DLC SCIENTIFIC ANALYTICS"); title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)

        self.btn_video = self.create_btn("1. SELECT VIDEO SOURCE")
        self.btn_video.clicked.connect(self.scegli_video); layout.addWidget(self.btn_video)

        self.btn_csv = self.create_btn("2. SELECT DLC DATA (.CSV)")
        self.btn_csv.clicked.connect(self.scegli_csv); layout.addWidget(self.btn_csv)

        self.lbl_info = QLabel("STATUS: Awaiting Files"); self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_info.setStyleSheet("color: #555; font-weight: bold; font-family: monospace;")
        layout.addWidget(self.lbl_info)

        line = QFrame(); line.setFrameShape(QFrame.Shape.HLine); layout.addWidget(line)

        self.btn_roi_exp = self.create_btn("3. DEFINE EXPERIMENTAL ROI"); self.btn_roi_exp.setEnabled(False)
        self.btn_roi_exp.clicked.connect(lambda: self.disegna_roi('exp')); layout.addWidget(self.btn_roi_exp)

        self.btn_roi_ctrl = self.create_btn("4. DEFINE CONTROL ROI"); self.btn_roi_ctrl.setEnabled(False)
        self.btn_roi_ctrl.clicked.connect(lambda: self.disegna_roi('ctrl')); layout.addWidget(self.btn_roi_ctrl)

        self.btn_arena = self.create_btn("5. DEFINE ARENA (POLYGON)"); self.btn_arena.setEnabled(False)
        self.btn_arena.clicked.connect(self.disegna_arena); layout.addWidget(self.btn_arena)

        self.lbl_cage = QLabel()
        self.lbl_cage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_cage.setStyleSheet("color: #555; font-weight: bold; font-family: monospace;")
        layout.addWidget(self.lbl_cage)

        self.btn_cage = self.create_btn("SET CAGE FORMAT (CM)")
        self.btn_cage.clicked.connect(self.configure_cage_format)
        layout.addWidget(self.btn_cage)

        # --- VIDEO SEGMENT PLAYER ---
        player_label = QLabel("VIDEO SEGMENT PLAYER")
        player_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        player_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        layout.addWidget(player_label)

        row = QHBoxLayout()

        self.start_input = QLineEdit()
        self.start_input.setPlaceholderText("Start frame")

        self.end_input = QLineEdit()
        self.end_input.setPlaceholderText("End frame")

        self.btn_csv_frames = self.create_btn("LOAD CSV FRAMES")
        self.btn_csv_frames.clicked.connect(self.load_csv_frames)
        
        row.addWidget(self.start_input)
        row.addWidget(self.end_input)
        row.addWidget(self.btn_csv_frames)

        layout.addLayout(row)

        self.btn_play_segment = self.create_btn("PLAY VIDEO SEGMENT")
        self.btn_play_segment.clicked.connect(self.play_segment_from_ui)
        layout.addWidget(self.btn_play_segment)
        self.btn_run = QPushButton("EXECUTE COMPLETE ANALYTICS")
        self.btn_run.setEnabled(False); self.btn_run.clicked.connect(self.avvia_motore_analisi)
        self.btn_run.setMinimumHeight(64)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: 900;
                font-size: 15px;
                padding: 18px;
                border: 2px solid #1e874b;
                border-radius: 5px;
            }
            QPushButton:disabled {
                background-color: #8fd1aa;
                color: #1b4d32;
                border: 2px solid #6fb98f;
            }
        """)
        
        layout.addWidget(self.btn_run)
        self.setCentralWidget(central)

    def scegli_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if file:
            self.video_path = file
            sec, ok = QInputDialog.getDouble(self, "Setup", "Start time (seconds):", 0.0, 0, 3600, 1)
            if ok: self.start_frame = int(sec * self.FPS)
            self.aggiorna_stato()
    
    def load_csv_frames(self):
        if not self.video_path:
            QMessageBox.warning(self, "Error", "Select a video before loading frame CSV.")
            return

        start_dir = video_review_dir(self.video_path)
        file, _ = QFileDialog.getOpenFileName(self, "Open CSV", start_dir, "CSV Files (*.csv)")
        
        if not file:
            return

        try:
            df = pd.read_csv(file, sep = ";")

            if "frame" not in df.columns:
                QMessageBox.warning(self, "Error", "Column 'frame' not found in CSV.")
                return

            # salva come lista di interi
            frame_list = df["frame"].dropna().astype(int).tolist()

            # salva separatamente il CSV dei frame, senza sostituire il CSV DLC principale
            self.frame_csv_path = file

            QMessageBox.information(
                self,
                "CSV Loaded",
                f"Loaded {len(frame_list)} frames."
            )
            self.open_timeline_player(frame_list, file)

        except Exception:
            import traceback
            QMessageBox.critical(self, "Error", traceback.format_exc())

    def play_segment_from_ui(self):
        try:
            if not self.video_path:
                QMessageBox.warning(self, "Error", "Select a video before playing a segment.")
                return

            if self.start_input.text() == "" or self.end_input.text() == "":
                QMessageBox.warning(self, "Error", "Enter both start and end frame.")
                return

            start = int(self.start_input.text())
            end = int(self.end_input.text())

            if start < 0 or end < 0:
                QMessageBox.warning(self, "Error", "Start and end frames must be >= 0.")
                return

            if end < start:
                QMessageBox.warning(self, "Error", "End frame must be greater than or equal to start frame.")
                return

            frame_list = np.arange(start, end + 1, dtype=int)

            self.open_timeline_player(frame_list)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid input: {str(e)}")

    def open_timeline_player(self, frame_list=None, frame_csv_path=None):
        try:
            player = VideoTimelinePlayer(
                self.video_path,
                frame_list,
                self.FPS,
                self,
                frame_csv_path,
                self.roi_exp,
                self.roi_ctrl,
                self.roi_exp_df,
                self.roi_ctrl_df
            )
            player.finished.connect(lambda _=None, p=player: self._remove_video_player(p))
            self.video_players.append(player)
            player.show()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _remove_video_player(self, player):
        if player in self.video_players:
            self.video_players.remove(player)

    def play_video_segment(self, video_path, frame_list, fps=30):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Unable to open video.")
            return

        for frame_number in frame_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ret, frame = cap.read()

            if not ret:
                continue

            cv2.putText(
                frame,
                f"Frame: {int(frame_number)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Video Segment", frame)

            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    
    
    def scegli_csv(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open CSV")
        if file: self.csv_path = file; self.aggiorna_stato()

    def aggiorna_stato(self):
        if self.video_path and self.csv_path:
            for b in [self.btn_roi_exp, self.btn_roi_ctrl, self.btn_arena, self.btn_run]: b.setEnabled(True)
            self.lbl_info.setText("STATUS: READY"); self.lbl_info.setStyleSheet("color: #27ae60;")

    def disegna_roi(self, mode):
        if mode in ('exp', 'ctrl'):
            try:
                editor = DynamicROIEditor(self.video_path, mode, self)
                if editor.exec() == QDialog.DialogCode.Accepted:
                    roi_df = editor.roi_df.copy()
                    csv_path = editor.saved_csv_path

                    first_roi = roi_df.dropna(subset=["x", "y", "width", "height"]).head(1)
                    roi_tuple = None
                    if not first_roi.empty:
                        row = first_roi.iloc[0]
                        roi_tuple = (
                            int(round(row["x"])),
                            int(round(row["y"])),
                            int(round(row["width"])),
                            int(round(row["height"]))
                        )

                    if mode == 'exp':
                        self.roi_exp_df = roi_df
                        self.roi_exp_csv_path = csv_path
                        self.roi_exp = roi_tuple
                        label = "Experimental"
                    else:
                        self.roi_ctrl_df = roi_df
                        self.roi_ctrl_csv_path = csv_path
                        self.roi_ctrl = roi_tuple
                        label = "Control"

                    QMessageBox.information(
                        self,
                        f"{label} ROI",
                        f"Dynamic ROI saved:\n{csv_path}"
                    )
                return
            except Exception:
                QMessageBox.critical(self, "Error", traceback.format_exc())
                return

        cap = cv2.VideoCapture(self.video_path); cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        ret, frame = cap.read(); cap.release()
        if ret:
            roi = select_roi_foreground(f"Select {mode.upper()} ROI", frame)
            cv2.destroyAllWindows()
            if roi != (0,0,0,0):
                if mode == 'exp': self.roi_exp = roi
                else: self.roi_ctrl = roi

    def disegna_arena(self):
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return

            arena_path = os.path.join(video_review_subdir(self.video_path, "ROI"), "arena_config.json")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(frame_rgb)
            ax.set_title("Click: Add | Drag: Move | Z: Undo | C: Clear | ENTER: Confirm")

            points_plot, = ax.plot([], [], 'ro-', markersize=8, mfc='yellow', lw=2)

            pts = []
            selected_node = None

            if os.path.exists(arena_path):
                try:
                    with open(arena_path, "r") as f:
                        pts = json.load(f)
                        pts = [(p[0], p[1]) for p in pts]
                except:
                    pts = []

            def update_plot():
                if len(pts) > 1:
                    x_vals, y_vals = zip(*(pts + [pts[0]]))
                    points_plot.set_data(x_vals, y_vals)
                elif len(pts) == 1:
                    points_plot.set_data([pts[0][0]], [pts[0][1]])
                else:
                    points_plot.set_data([], [])
                fig.canvas.draw_idle()

            update_plot()

            def on_click(event):
                nonlocal selected_node
                if event.inaxes != ax: return

                for i, p in enumerate(pts):
                    if np.hypot(p[0] - event.xdata, p[1] - event.ydata) < 15:
                        selected_node = i
                        return

                pts.append((event.xdata, event.ydata))
                update_plot()

            def on_motion(event):
                nonlocal selected_node
                if selected_node is not None and event.inaxes == ax:
                    pts[selected_node] = (event.xdata, event.ydata)
                    update_plot()

            def on_release(event):
                nonlocal selected_node
                selected_node = None

            def on_key(event):
                nonlocal pts
                if event.key == 'z':
                    if pts:
                        pts.pop()
                        update_plot()
                elif event.key == 'c':
                    pts = []
                    update_plot()
                elif event.key == 'enter':
                    if len(pts) > 2:
                        self.arena_poly = Polygon(pts)
                        self.calibrate_cm_per_pixel_from_arena()
                        self.update_cage_label()
                        try:
                            with open(arena_path, "w") as f:
                                json.dump(pts, f)
                        except:
                            pass
                        plt.close(fig)

            fig.canvas.mpl_connect('button_press_event', on_click)
            fig.canvas.mpl_connect('motion_notify_event', on_motion)
            fig.canvas.mpl_connect('button_release_event', on_release)
            fig.canvas.mpl_connect('key_press_event', on_key)

            plt.show()

    def avvia_motore_analisi(self):
        try:
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            results_root = os.path.join(desktop, "DLC_Scientific_Output")
            os.makedirs(results_root, exist_ok=True)
            
            df_raw = pd.read_csv(self.csv_path, header=[0, 1, 2, 3], index_col=0)
            scorer = df_raw.columns.get_level_values(0)[0]
            
            # --- MODIFICA 1: ID DA INDIVIDUALS ---
            animal = df_raw.columns.get_level_values(1)[0]
            
            db_path = os.path.join(results_root, "Consolidated_Behavioral_Data.xlsx")
            print(f"DOVE Excel? {db_path}")
            if os.path.exists(db_path):
                existing_db = pd.read_excel(db_path)
                if "Animal_ID" in existing_db.columns and animal in existing_db["Animal_ID"].astype(str).values:
                    reply = QMessageBox.question(
                        self,
                        "Duplicate",
                        f"Subject '{animal}' already analyzed.\n\nRe-run analysis and replace the existing Excel row?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return

            animal_dir = os.path.join(results_root, animal)
            for sub in ["Occupancy_Heatmaps", "Spatial_Navigation", "Behavioral_Sequencing"]: os.makedirs(os.path.join(animal_dir, sub), exist_ok=True)

            bodyparts = list(dict.fromkeys(df_raw.columns.get_level_values(2)))

            def pick_nose_tail_pair():
                if "nose" in bodyparts and "tail_base" in bodyparts:
                    return "nose", "tail_base"

                bodypart_set = {str(bp) for bp in bodyparts}
                for bp in bodyparts:
                    bp_name = str(bp)
                    if not bp_name.startswith("nose"):
                        continue
                    suffix = bp_name[len("nose"):]
                    tail_name = f"tail_base{suffix}"
                    if tail_name in bodypart_set:
                        return bp_name, tail_name

                nose_candidates = [str(bp) for bp in bodyparts if str(bp).startswith("nose")]
                tail_candidates = [str(bp) for bp in bodyparts if str(bp).startswith("tail_base")]
                if nose_candidates and tail_candidates:
                    return nose_candidates[0], tail_candidates[0]

                raise KeyError(
                    "Cannot find a nose/tail_base bodypart pair. "
                    f"Available bodyparts: {', '.join(map(str, bodyparts))}"
                )

            nose_bp, tail_bp = pick_nose_tail_pair()

            nx = df_raw.loc[:, (scorer, animal, nose_bp, 'x')].to_numpy()[self.start_frame:]
            ny = df_raw.loc[:, (scorer, animal, nose_bp, 'y')].to_numpy()[self.start_frame:]
            tx = df_raw.loc[:, (scorer, animal, tail_bp, 'x')].to_numpy()[self.start_frame:]
            ty = df_raw.loc[:, (scorer, animal, tail_bp, 'y')].to_numpy()[self.start_frame:]
            nlk = df_raw.loc[:, (scorer, animal, nose_bp, 'likelihood')].to_numpy()[self.start_frame:]
            tlk = df_raw.loc[:, (scorer, animal, tail_bp, 'likelihood')].to_numpy()[self.start_frame:]
            frame_idx = np.arange(self.start_frame, self.start_frame + len(nx))
            bodypart_tracks = {}
            for bodypart in df_raw.columns.get_level_values(2).unique():
                bp_name = str(bodypart)
                bp_lower = bp_name.lower()
                if "tail" in bp_lower:
                    continue
                try:
                    bx = df_raw.loc[:, (scorer, animal, bodypart, 'x')].to_numpy()[self.start_frame:]
                    by = df_raw.loc[:, (scorer, animal, bodypart, 'y')].to_numpy()[self.start_frame:]
                    blk = df_raw.loc[:, (scorer, animal, bodypart, 'likelihood')].to_numpy()[self.start_frame:]
                except KeyError:
                    continue
                bodypart_tracks[bp_name] = (bx, by, blk)

                # Controllo: arena obbligatoria
            if self.arena_poly is None:
                QMessageBox.warning(self, "Arena missing", "Define the arena polygon before running the analysis.")
                return
            self.calibrate_cm_per_pixel_from_arena()

            if self.roi_exp is None and (self.roi_exp_df is None or self.roi_exp_df.empty):
                QMessageBox.warning(self, "EXP ROI missing", "Define the experimental ROI before running the analysis.")
                return
            if self.roi_ctrl is None and (self.roi_ctrl_df is None or self.roi_ctrl_df.empty):
                QMessageBox.warning(self, "CTRL ROI missing", "Define the control ROI before running the analysis.")
                return

            # Maschera: tengo solo i frame in cui nose e tail_base sono entrambi dentro il poligono
            mask = np.array([
                self.arena_poly.contains(Point(x_n, y_n)) and self.arena_poly.contains(Point(x_t, y_t))
                for x_n, y_n, x_t, y_t in zip(nx, ny, tx, ty)
            ], dtype=bool)


            # Applico la maschera a tutti i vettori
            nx = nx[mask]
            ny = ny[mask]
            tx = tx[mask]
            ty = ty[mask]
            nlk = nlk[mask]
            tlk = tlk[mask]
            frame_idx = frame_idx[mask]
            bodypart_tracks = {
                bp: (bx[mask], by[mask], blk[mask])
                for bp, (bx, by, blk) in bodypart_tracks.items()
            }
            self.df_coord = pd.DataFrame({
                                    "x": nx,
                                    "y": ny,
                                    "frame": frame_idx
                                    })

            # Controllo: dopo il filtraggio deve restare almeno qualche frame
            if len(nx) < 2:
                QMessageBox.warning(
                    self,
                    "Not enough data",
                    "After filtering by arena polygon, too few valid frames remain for analysis."
                )
                return


            # correggo per velocità negative
            smooth_window = min(11, len(nx) if len(nx) % 2 == 1 else len(nx) - 1)
            if smooth_window >= 5:
                smooth_poly = min(3, smooth_window - 1)
                nx_smooth = savgol_filter(nx, smooth_window, smooth_poly)
                ny_smooth = savgol_filter(ny, smooth_window, smooth_poly)
            else:
                nx_smooth = nx
                ny_smooth = ny
            dx, dy = np.diff(nx_smooth), np.diff(ny_smooth)
            dist_inst = self.diff_distance_cm(dx, dy)
            vel_nose = np.append(dist_inst * self.FPS, 0)
            
            def dynamic_roi_centers(roi_df):
                if roi_df is None or roi_df.empty:
                    return None
                required_cols = ["frame", "x", "y", "width", "height"]
                if any(col not in roi_df.columns for col in required_cols):
                    return None

                aligned = roi_df.copy()
                aligned["frame"] = pd.to_numeric(aligned["frame"], errors="coerce")
                aligned = aligned.dropna(subset=["frame"])
                aligned["frame"] = aligned["frame"].astype(int)
                aligned = aligned.drop_duplicates(subset=["frame"], keep="last")
                aligned = aligned.set_index("frame").reindex(frame_idx)

                for col in ["x", "y", "width", "height"]:
                    aligned[col] = pd.to_numeric(aligned[col], errors="coerce")

                aligned[["x", "y", "width", "height"]] = (
                    aligned[["x", "y", "width", "height"]]
                    .interpolate(method="linear")
                    .ffill()
                    .bfill()
                )
                valid = aligned[["x", "y", "width", "height"]].notna().all(axis=1).to_numpy()
                cx = (aligned["x"] + aligned["width"] / 2).to_numpy(dtype=float)
                cy = (aligned["y"] + aligned["height"] / 2).to_numpy(dtype=float)
                return cx, cy, valid

            def static_roi_centers(roi):
                if not roi:
                    return None
                cx = np.full(len(nx), roi[0] + roi[2] / 2, dtype=float)
                cy = np.full(len(nx), roi[1] + roi[3] / 2, dtype=float)
                valid = np.ones(len(nx), dtype=bool)
                return cx, cy, valid

            def roi_centers(roi, roi_df=None):
                centers = dynamic_roi_centers(roi_df)
                if centers is not None and np.any(centers[2]):
                    return centers
                return static_roi_centers(roi)

            def nose_inside_roi(roi, roi_df=None):
                if roi_df is not None and not roi_df.empty:
                    required_cols = ["frame", "x", "y", "width", "height"]
                    if all(col in roi_df.columns for col in required_cols):
                        aligned = roi_df.copy()
                        aligned["frame"] = pd.to_numeric(aligned["frame"], errors="coerce")
                        aligned = aligned.dropna(subset=["frame"])
                        aligned["frame"] = aligned["frame"].astype(int)
                        aligned = aligned.drop_duplicates(subset=["frame"], keep="last")
                        aligned = aligned.set_index("frame").reindex(frame_idx)

                        for col in ["x", "y", "width", "height"]:
                            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")

                        aligned[["x", "y", "width", "height"]] = (
                            aligned[["x", "y", "width", "height"]]
                            .interpolate(method="linear")
                            .ffill()
                            .bfill()
                        )
                        x0 = aligned["x"].to_numpy(dtype=float)
                        y0 = aligned["y"].to_numpy(dtype=float)
                        w = aligned["width"].to_numpy(dtype=float)
                        h = aligned["height"].to_numpy(dtype=float)
                        valid_box = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(w) & np.isfinite(h)
                        return (
                            valid_box
                            & (nx >= x0)
                            & (nx <= x0 + w)
                            & (ny >= y0)
                            & (ny <= y0 + h)
                        )

                if not roi:
                    return np.zeros(len(nx), dtype=bool)

                x0, y0, w, h = roi
                return (nx >= x0) & (nx <= x0 + w) & (ny >= y0) & (ny <= y0 + h)

            def nose_inside_expanded_roi(roi, roi_df=None, margin_cm=0.0):
                margin_x = margin_cm / max(self.CM_PER_PIXEL_X, 1e-9)
                margin_y = margin_cm / max(self.CM_PER_PIXEL_Y, 1e-9)
                if roi_df is not None and not roi_df.empty:
                    required_cols = ["frame", "x", "y", "width", "height"]
                    if all(col in roi_df.columns for col in required_cols):
                        aligned = roi_df.copy()
                        aligned["frame"] = pd.to_numeric(aligned["frame"], errors="coerce")
                        aligned = aligned.dropna(subset=["frame"])
                        aligned["frame"] = aligned["frame"].astype(int)
                        aligned = aligned.drop_duplicates(subset=["frame"], keep="last")
                        aligned = aligned.set_index("frame").reindex(frame_idx)

                        for col in ["x", "y", "width", "height"]:
                            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")

                        aligned[["x", "y", "width", "height"]] = (
                            aligned[["x", "y", "width", "height"]]
                            .interpolate(method="linear")
                            .ffill()
                            .bfill()
                        )
                        x0 = aligned["x"].to_numpy(dtype=float)
                        y0 = aligned["y"].to_numpy(dtype=float)
                        w = aligned["width"].to_numpy(dtype=float)
                        h = aligned["height"].to_numpy(dtype=float)
                        valid_box = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(w) & np.isfinite(h)
                        return (
                            valid_box
                            & (nx >= x0 - margin_x)
                            & (nx <= x0 + w + margin_x)
                            & (ny >= y0 - margin_y)
                            & (ny <= y0 + h + margin_y)
                        )

                if not roi:
                    return np.zeros(len(nx), dtype=bool)

                x0, y0, w, h = roi
                return (
                    (nx >= x0 - margin_x)
                    & (nx <= x0 + w + margin_x)
                    & (ny >= y0 - margin_y)
                    & (ny <= y0 + h + margin_y)
                )

            def mean_dist_obj(roi, roi_df=None):
                centers = roi_centers(roi, roi_df)
                if centers is None:
                    return 0
                cx, cy, valid = centers
                if not np.any(valid):
                    return 0
                d = self.distance_cm(nx[valid], ny[valid], cx[valid], cy[valid])
                return np.mean(d)

            def roi_distance_targets(roi, roi_df=None):
                if roi_df is not None and not roi_df.empty:
                    required_cols = ["frame", "x", "y", "width", "height"]
                    if all(col in roi_df.columns for col in required_cols):
                        aligned = roi_df.copy()
                        aligned["frame"] = pd.to_numeric(aligned["frame"], errors="coerce")
                        aligned = aligned.dropna(subset=["frame"])
                        aligned["frame"] = aligned["frame"].astype(int)
                        aligned = aligned.drop_duplicates(subset=["frame"], keep="last")
                        aligned = aligned.set_index("frame").reindex(frame_idx)

                        for col in ["x", "y", "width", "height"]:
                            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")

                        aligned[["x", "y", "width", "height"]] = (
                            aligned[["x", "y", "width", "height"]]
                            .interpolate(method="linear")
                            .ffill()
                            .bfill()
                        )
                        x0 = aligned["x"].to_numpy(dtype=float)
                        y0 = aligned["y"].to_numpy(dtype=float)
                        w = aligned["width"].to_numpy(dtype=float)
                        h = aligned["height"].to_numpy(dtype=float)
                        valid_box = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(w) & np.isfinite(h) & (w > 0) & (h > 0)
                        if not np.any(valid_box):
                            return roi_distance_targets(roi, None)
                        target_x = np.clip(nx, x0, x0 + w)
                        target_y = np.clip(ny, y0, y0 + h)
                        d = np.full(len(nx), np.inf, dtype=float)
                        d[valid_box] = self.distance_cm(
                            nx[valid_box],
                            ny[valid_box],
                            target_x[valid_box],
                            target_y[valid_box]
                        )
                        return target_x, target_y, d, valid_box

                if not roi:
                    return None

                x0, y0, w, h = roi
                target_x = np.clip(nx, x0, x0 + w)
                target_y = np.clip(ny, y0, y0 + h)
                d = self.distance_cm(nx, ny, target_x, target_y)
                valid_box = np.ones(len(nx), dtype=bool)
                return target_x, target_y, d, valid_box

            def body_over_roi(roi, roi_df=None):
                if roi_df is not None and not roi_df.empty:
                    required_cols = ["frame", "x", "y", "width", "height"]
                    if all(col in roi_df.columns for col in required_cols):
                        aligned = roi_df.copy()
                        aligned["frame"] = pd.to_numeric(aligned["frame"], errors="coerce")
                        aligned = aligned.dropna(subset=["frame"])
                        aligned["frame"] = aligned["frame"].astype(int)
                        aligned = aligned.drop_duplicates(subset=["frame"], keep="last")
                        aligned = aligned.set_index("frame").reindex(frame_idx)

                        for col in ["x", "y", "width", "height"]:
                            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")

                        aligned[["x", "y", "width", "height"]] = (
                            aligned[["x", "y", "width", "height"]]
                            .interpolate(method="linear")
                            .ffill()
                            .bfill()
                        )
                        x0 = aligned["x"].to_numpy(dtype=float)
                        y0 = aligned["y"].to_numpy(dtype=float)
                        w = aligned["width"].to_numpy(dtype=float)
                        h = aligned["height"].to_numpy(dtype=float)
                        valid_box = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(w) & np.isfinite(h) & (w > 0) & (h > 0)
                    else:
                        valid_box = np.zeros(len(nx), dtype=bool)
                        x0 = y0 = w = h = np.full(len(nx), np.nan, dtype=float)
                elif roi:
                    x0, y0, w, h = roi
                    x0 = np.full(len(nx), x0, dtype=float)
                    y0 = np.full(len(nx), y0, dtype=float)
                    w = np.full(len(nx), w, dtype=float)
                    h = np.full(len(nx), h, dtype=float)
                    valid_box = np.ones(len(nx), dtype=bool)
                else:
                    return np.zeros(len(nx), dtype=bool)

                margin_x = self.SNIFF_BODY_OVERLAP_MARGIN_CM / max(self.CM_PER_PIXEL_X, 1e-9)
                margin_y = self.SNIFF_BODY_OVERLAP_MARGIN_CM / max(self.CM_PER_PIXEL_Y, 1e-9)
                overlap = np.zeros(len(nx), dtype=bool)
                for i in np.flatnonzero(valid_box):
                    roi_box = box(
                        x0[i] - margin_x,
                        y0[i] - margin_y,
                        x0[i] + w[i] + margin_x,
                        y0[i] + h[i] + margin_y
                    )
                    body_axis = LineString([(tx[i], ty[i]), (nx[i], ny[i])])
                    overlap[i] = body_axis.intersects(roi_box)
                return overlap

            def bodypoint_near_roi(roi, roi_df=None):
                if not bodypart_tracks:
                    return np.zeros(len(nx), dtype=bool)

                if roi_df is not None and not roi_df.empty:
                    required_cols = ["frame", "x", "y", "width", "height"]
                    if all(col in roi_df.columns for col in required_cols):
                        aligned = roi_df.copy()
                        aligned["frame"] = pd.to_numeric(aligned["frame"], errors="coerce")
                        aligned = aligned.dropna(subset=["frame"])
                        aligned["frame"] = aligned["frame"].astype(int)
                        aligned = aligned.drop_duplicates(subset=["frame"], keep="last")
                        aligned = aligned.set_index("frame").reindex(frame_idx)

                        for col in ["x", "y", "width", "height"]:
                            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")

                        aligned[["x", "y", "width", "height"]] = (
                            aligned[["x", "y", "width", "height"]]
                            .interpolate(method="linear")
                            .ffill()
                            .bfill()
                        )
                        x0 = aligned["x"].to_numpy(dtype=float)
                        y0 = aligned["y"].to_numpy(dtype=float)
                        w = aligned["width"].to_numpy(dtype=float)
                        h = aligned["height"].to_numpy(dtype=float)
                        valid_box = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(w) & np.isfinite(h) & (w > 0) & (h > 0)
                    else:
                        valid_box = np.zeros(len(nx), dtype=bool)
                        x0 = y0 = w = h = np.full(len(nx), np.nan, dtype=float)
                elif roi:
                    x0, y0, w, h = roi
                    x0 = np.full(len(nx), x0, dtype=float)
                    y0 = np.full(len(nx), y0, dtype=float)
                    w = np.full(len(nx), w, dtype=float)
                    h = np.full(len(nx), h, dtype=float)
                    valid_box = np.ones(len(nx), dtype=bool)
                else:
                    return np.zeros(len(nx), dtype=bool)

                margin_x = self.SNIFF_BODYPOINT_MARGIN_CM / max(self.CM_PER_PIXEL_X, 1e-9)
                margin_y = self.SNIFF_BODYPOINT_MARGIN_CM / max(self.CM_PER_PIXEL_Y, 1e-9)
                contact_count = np.zeros(len(nx), dtype=int)
                for _, (bx, by, blk) in bodypart_tracks.items():
                    valid_point = np.isfinite(bx) & np.isfinite(by) & np.isfinite(blk) & (blk > 0.6)
                    contact_count += (
                        valid_box
                        & valid_point
                        & (bx >= x0 - margin_x)
                        & (bx <= x0 + w + margin_x)
                        & (by >= y0 - margin_y)
                        & (by <= y0 + h + margin_y)
                    ).astype(int)
                return contact_count >= self.SNIFF_BODYPOINT_MIN_POINTS

            def calc_sniff(roi, roi_df=None):
                targets = roi_distance_targets(roi, roi_df)
                if targets is None:
                    empty_debug = pd.DataFrame({"frame": frame_idx})
                    return np.zeros(len(nx), dtype=bool), empty_debug, np.zeros(len(nx), dtype=bool)
                target_x, target_y, d, valid = targets

                body_vec_x = (nx - tx) * self.CM_PER_PIXEL_X
                body_vec_y = (ny - ty) * self.CM_PER_PIXEL_Y
                obj_vec_x = (target_x - nx) * self.CM_PER_PIXEL_X
                obj_vec_y = (target_y - ny) * self.CM_PER_PIXEL_Y
                denom = (
                    np.sqrt(body_vec_x ** 2 + body_vec_y ** 2)
                    * np.sqrt(obj_vec_x ** 2 + obj_vec_y ** 2)
                )
                cos_angle = np.full(len(nx), -1.0, dtype=float)
                good_denom = denom > 1e-9
                cos_angle[good_denom] = (
                    body_vec_x[good_denom] * obj_vec_x[good_denom]
                    + body_vec_y[good_denom] * obj_vec_y[good_denom]
                ) / denom[good_denom]
                facing_roi = cos_angle >= np.cos(np.deg2rad(self.SNIFF_MAX_ANGLE_DEG))
                near_facing_roi = cos_angle >= np.cos(np.deg2rad(self.SNIFF_NEAR_MAX_ANGLE_DEG))
                pre_contact_facing_roi = cos_angle >= np.cos(np.deg2rad(self.SNIFF_PRE_CONTACT_MAX_ANGLE_DEG))
                candidate_facing_roi = cos_angle >= np.cos(np.deg2rad(self.SNIFF_CANDIDATE_MAX_ANGLE_DEG))
                inside_roi = nose_inside_roi(roi, roi_df)
                very_close_to_roi = d <= self.SNIFF_DIRECTION_FREE_CM
                over_roi = body_over_roi(roi, roi_df)
                bodypoint_contact = bodypoint_near_roi(roi, roi_df)
                nose_contact = nose_inside_expanded_roi(
                    roi,
                    roi_df,
                    self.SNIFF_NOSE_CONTACT_MARGIN_CM
                )

                directional_sniff = (
                    (d < self.SNIFF_DISTANCE_CM)
                    & (facing_roi | very_close_to_roi)
                    & valid
                    & (nlk > 0.6)
                    & (tlk > 0.6)
                )
                nose_inside_sniff = inside_roi & valid & (nlk > 0.6)
                nose_contact_sniff = (
                    ((nose_contact | (d <= self.SNIFF_NEAR_DISTANCE_CM)) & near_facing_roi)
                    & valid
                    & (nlk > self.SNIFF_NOSE_CONTACT_LIKELIHOOD)
                )

                raw_sniff = (
                    directional_sniff
                    | nose_inside_sniff
                )
                raw_sniff = raw_sniff | nose_contact_sniff
                pre_contact_candidate = (
                    (d <= self.SNIFF_PRE_CONTACT_DISTANCE_CM)
                    & pre_contact_facing_roi
                    & valid
                    & (nlk > self.SNIFF_NOSE_CONTACT_LIKELIHOOD)
                )
                cleaned_sniff = clean_behavior(raw_sniff, self.FPS, self.SNIFF_GAP_SEC, self.SNIFF_MIN_DURATION_SEC)
                cleaned_sniff = fill_internal_gaps(cleaned_sniff, self.SNIFF_GAP_FILL_FRAMES)
                cleaned_sniff = extend_sniff_onsets(
                    cleaned_sniff,
                    pre_contact_candidate,
                    self.SNIFF_PRE_CONTACT_FRAMES
                )
                review_candidate = (
                    (d <= self.SNIFF_CANDIDATE_DISTANCE_CM)
                    & candidate_facing_roi
                    & valid
                    & (nlk > self.SNIFF_CANDIDATE_MIN_LIKELIHOOD)
                    & ~cleaned_sniff
                )
                debug = pd.DataFrame({
                    "frame": frame_idx,
                    "nose_x": nx,
                    "nose_y": ny,
                    "tail_x": tx,
                    "tail_y": ty,
                    "distance_to_roi_cm": d,
                    "cos_angle": cos_angle,
                    "facing_roi": facing_roi,
                    "near_facing_roi": near_facing_roi,
                    "pre_contact_facing_roi": pre_contact_facing_roi,
                    "candidate_facing_roi": candidate_facing_roi,
                    "inside_roi": inside_roi,
                    "nose_contact_zone": nose_contact,
                    "body_axis_over_roi": over_roi,
                    "bodypoint_contact_unused": bodypoint_contact,
                    "directional_sniff_raw": directional_sniff,
                    "nose_inside_sniff_raw": nose_inside_sniff,
                    "nose_contact_sniff_raw": nose_contact_sniff,
                    "pre_contact_candidate": pre_contact_candidate,
                    "review_candidate": review_candidate,
                    "raw_sniff": raw_sniff,
                    "cleaned_sniff": cleaned_sniff,
                    "nose_likelihood": nlk,
                    "tail_likelihood": tlk,
                })
                return cleaned_sniff, debug, review_candidate

            f_exp, debug_exp, cand_exp = calc_sniff(self.roi_exp, self.roi_exp_df) 
            f_ctrl, debug_ctrl, cand_ctrl = calc_sniff(self.roi_ctrl, self.roi_ctrl_df)
            
            exp_sniff_frames = pd.DataFrame({
                                     "x": nx[f_exp],
                                     "y": ny[f_exp],
                                     "frame": frame_idx[f_exp]
                                    })            
            sniff_dir = video_review_subdir(self.video_path, "Sniff")
            candidates_dir = video_review_subdir(self.video_path, "Candidates")
            behavior_dir = video_review_subdir(self.video_path, "Behavior")

            exp_sniffing_path = os.path.join(sniff_dir, "exp_sniff_frames.csv")
            exp_sniff_frames.to_csv(exp_sniffing_path, sep=";")
            exp_candidates = debug_exp.loc[cand_exp, [
                "frame",
                "nose_x",
                "nose_y",
                "distance_to_roi_cm",
                "cos_angle",
                "nose_likelihood",
                "tail_likelihood"
            ]].copy()
            exp_candidates_path = os.path.join(candidates_dir, "exp_sniff_candidates.csv")
            exp_candidates.to_csv(exp_candidates_path, sep=";", index=False)

            ctrl_sniff_frames = pd.DataFrame({
                                     "x": nx[f_ctrl],
                                     "y": ny[f_ctrl],
                                     "frame": frame_idx[f_ctrl]
                                    })            
            ctrl_sniffing_path = os.path.join(sniff_dir, "ctrl_sniff_frames.csv")
            ctrl_sniff_frames.to_csv(ctrl_sniffing_path, sep=";")
            ctrl_candidates = debug_ctrl.loc[cand_ctrl, [
                "frame",
                "nose_x",
                "nose_y",
                "distance_to_roi_cm",
                "cos_angle",
                "nose_likelihood",
                "tail_likelihood"
            ]].copy()
            ctrl_candidates_path = os.path.join(candidates_dir, "ctrl_sniff_candidates.csv")
            ctrl_candidates.to_csv(ctrl_candidates_path, sep=";", index=False)


            body_len = self.distance_cm(nx, ny, tx, ty)
            mediana = np.nanmedian(pd.Series(body_len).rolling(7, center=True).median())
            f_sap = clean_behavior((body_len > mediana*1.3) & (vel_nose < 2.5) & (tlk > 0.6), self.FPS, 0.8, 0.5)

            # --- MODIFICA 2: FREEZING 1 SECONDO ---
            is_freezing = clean_behavior(vel_nose < 0.5, self.FPS, 0.5, 1.0)
            
            is_freezing_frames = pd.DataFrame({
                                     "x": nx[is_freezing],
                                     "y": ny[is_freezing],
                                     "frame": frame_idx[is_freezing],
                                     "velocita": vel_nose[is_freezing]
                                    })
            freezing_path = os.path.join(behavior_dir, "is_freezing.csv")
            is_freezing_frames.to_csv(freezing_path, sep=";")

            is_thigmo = np.zeros(len(nx), dtype=bool)
            if self.arena_poly:
                arena_cm = scale_geometry(
                    self.arena_poly,
                    xfact=self.CM_PER_PIXEL_X,
                    yfact=self.CM_PER_PIXEL_Y,
                    origin=(0, 0)
                )
                bound = arena_cm.boundary
                is_thigmo = np.array([
                    Point(px * self.CM_PER_PIXEL_X, py * self.CM_PER_PIXEL_Y).distance(bound) <= self.THRESHOLD_THIGMO_CM
                    for px, py in zip(nx, ny)
                ])
            
            res = {
                "Animal_ID": animal,
                "Exp_Sniffing_s": round(np.sum(f_exp)/self.FPS, 2),
                "Ctrl_Sniffing_s": round(np.sum(f_ctrl)/self.FPS, 2),
                "Preference_Index_PI": round((np.sum(f_exp)-np.sum(f_ctrl))/(np.sum(f_exp)+np.sum(f_ctrl)+0.001), 3),
                "Mean_Distance_to_Exp_cm": round(mean_dist_obj(self.roi_exp, self.roi_exp_df), 2),
                "Freezing_Total_Duration_s": round(np.sum(is_freezing)/self.FPS, 2),
                "Risk_Assessment_SAP_s": round(np.sum(f_sap)/self.FPS, 2),
                "Thigmotaxis_perc": round((np.sum(is_thigmo)/len(is_thigmo))*100, 2),
                "Locomotion_Total_Dist_cm": round(np.sum(dist_inst), 2),
                "Cm_per_pixel_X": round(self.CM_PER_PIXEL_X, 6),
                "Cm_per_pixel_Y": round(self.CM_PER_PIXEL_Y, 6),
                "Sniff_Max_Angle_deg": self.SNIFF_MAX_ANGLE_DEG,
                "Sniff_Direction_Free_cm": self.SNIFF_DIRECTION_FREE_CM,
                "Sniff_Nose_Contact_Margin_cm": self.SNIFF_NOSE_CONTACT_MARGIN_CM,
                "Sniff_Nose_Contact_Likelihood": self.SNIFF_NOSE_CONTACT_LIKELIHOOD,
                "Sniff_Near_Distance_cm": self.SNIFF_NEAR_DISTANCE_CM,
                "Sniff_Near_Max_Angle_deg": self.SNIFF_NEAR_MAX_ANGLE_DEG,
                "Sniff_Gap_Fill_Frames": self.SNIFF_GAP_FILL_FRAMES,
                "Sniff_Pre_Contact_Frames": self.SNIFF_PRE_CONTACT_FRAMES,
                "Sniff_Pre_Contact_Distance_cm": self.SNIFF_PRE_CONTACT_DISTANCE_CM,
                "Sniff_Pre_Contact_Max_Angle_deg": self.SNIFF_PRE_CONTACT_MAX_ANGLE_DEG,
                "Sniff_Candidate_Distance_cm": self.SNIFF_CANDIDATE_DISTANCE_CM,
                "Sniff_Candidate_Max_Angle_deg": self.SNIFF_CANDIDATE_MAX_ANGLE_DEG,
                "Sniff_Candidate_Min_Likelihood": self.SNIFF_CANDIDATE_MIN_LIKELIHOOD
            }

            self.finalizza(nx, ny, f_exp, f_ctrl, f_sap, is_thigmo, res, animal_dir, db_path,is_freezing_frames)

        except Exception as e:
            error_msg = traceback.format_exc()
            QMessageBox.critical(self, "Error", error_msg)

    def finalizza(self, x, y, f_exp, f_ctrl, f_sap, f_thigmo, res, folder, db_path, is_freezing_frames):
        new_df = pd.DataFrame([res])
        if os.path.exists(db_path):
            old_df = pd.read_excel(db_path)
            if "Animal_ID" in old_df.columns:
                old_df = old_df[old_df["Animal_ID"].astype(str) != str(res["Animal_ID"])]
            pd.concat([old_df, new_df], ignore_index=True).to_excel(db_path, index=False)
        else:
            new_df.to_excel(db_path, index=False)

        plt.style.use('seaborn-v0_8-white')
        
        # 1. BEHAVIORAL SEQUENCING
        fig1 = plt.figure(figsize=(16, 7))
        t = np.arange(len(x))/self.FPS
        plt.fill_between(t, 2.2, 3.0, where=f_exp, color='#2980b9', alpha=0.9, label='Exp Sniffing')
        plt.fill_between(t, 1.1, 1.9, where=f_ctrl, color='#95a5a6', alpha=0.9, label='Ctrl Sniffing')
        plt.fill_between(t, 0.0, 0.8, where=f_sap, color='#d35400', alpha=0.9, label='SAP')
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(2))
        plt.grid(which='major', axis='x', linestyle='-', color='gray', alpha=0.3); plt.grid(which='minor', axis='x', linestyle=':', color='lightgray', alpha=0.4)
        plt.yticks([0.4, 1.5, 2.6], ['SAP', 'Control', 'Experimental'], fontweight='bold')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1)); plt.tight_layout()
        fig1.savefig(os.path.join(folder, "Behavioral_Sequencing", "temporal_sequencing.png"), dpi=300)

        # 2. SPATIAL NAVIGATION
        fig2 = plt.figure(figsize=(8, 8))
        plt.scatter(x[~f_thigmo], y[~f_thigmo], c='#2980b9', s=1, alpha=0.3, label='Center')
        plt.scatter(x[f_thigmo], y[f_thigmo], c='#c0392b', s=3, alpha=0.8, label='Periphery (Thigmo)')
        if self.arena_poly:
            ax_x, ax_y = self.arena_poly.exterior.xy
            plt.plot(ax_x, ax_y, color='black', lw=2)
        plt.gca().invert_yaxis(); plt.legend(markerscale=5); plt.title("Spatial Navigation Pattern")
        fig2.savefig(os.path.join(folder, "Spatial_Navigation", "trajectory_pattern.png"), dpi=300)


        # 3. OCCUPANCY HEATMAPS
        fig3 = plt.figure(figsize=(8, 8))
        ax3 = fig3.add_subplot(111)

        # Heatmap numerica
        bins = 40

        frames1 = self.df_coord["frame"].astype(int)
        frames2 = is_freezing_frames["frame"].astype(int)
        df_filtered = self.df_coord[~frames1.isin(frames2)]
        
        xf = df_filtered["x"]
        yf = df_filtered["y"]
        print(f"len(xf) ={len(xf)} -- len(nx) = {len(x)}")
        heatmap, xedges, yedges = np.histogram2d(xf, yf, bins=bins)
        heatmap_seconds = heatmap / self.FPS

        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)


        mask_arena = contains(self.arena_poly, X, Y)

        heatmap_masked = np.where(mask_arena, heatmap_seconds.T, np.nan)

        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color='white')

        img = ax3.imshow(
            heatmap_masked,
            origin='lower',
            cmap=cmap,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect='equal'
        )
        # ROI
        for r, c, l in [(self.roi_exp, 'blue', 'EXP'), (self.roi_ctrl, 'black', 'CTRL')]:
            if r:
                rect = plt.Rectangle(
                    (r[0], r[1]), r[2], r[3],
                    fill=False, edgecolor=c, linewidth=2, label=l
                )
                ax3.add_patch(rect)

        # Perimetro arena
        if self.arena_poly:
            ax_x, ax_y = self.arena_poly.exterior.xy
            ax3.plot(ax_x, ax_y, color='cyan', linewidth=2.5, label='Arena')
        
        ax3.invert_yaxis()
        ax3.set_title("Occupancy Heatmap (freezing excluded)")
        ax3.legend()
        fig3.colorbar(img, ax=ax3, label="Occupancy (s)")
        fig3.savefig(os.path.join(folder, "Occupancy_Heatmaps", "density_analysis.png"), dpi=300)
        QMessageBox.information(self, "Success", f"Analysis Consolidated for {res['Animal_ID']}.")
        plt.close('all')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = DLCLabSuiteFinal()

    # Inserisci qui i file
    #win.video_path = r"C:\Users\Sergio\Dropbox\dlc\GUI-dlc\GUI-dlc\TMA-black.mp4"
    #win.csv_path = r"C:\Users\Sergio\Dropbox\dlc\GUI-dlc\GUI-dlc\risultati_analisi_topo.csv"

    # Aggiorna stato GUI
    #win.aggiorna_stato()

    win.show()
    sys.exit(app.exec())
