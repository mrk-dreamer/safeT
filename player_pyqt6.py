# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import subprocess
import sys
import os
import csv
from typing import Literal
import numpy as np
from matplotlib import colormaps
import pandas as pd
import vlc
from parselmouth import Sound
from praatio import textgrid
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QFileDialog,
    QLineEdit,
    QScrollArea,
    QLabel,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QMenu,
    QMessageBox,
)
from PyQt6 import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.exporters as pg_exporters
from asr import WhisperX_ASR
from formants import FormantData
from helper import (
    SentenceTimestamp,
    WordTimestamp,
    PhonemeTimestamp,
    get_phoneme_hash,
    ensure_utf8_display,
)
from mfa_align import DEFAULT_EXPERIMENT_NAME as MFA_DEFAULT_EXPERIMENT_NAME
from mfa_runner import run_mfa_alignment
import locale
from scipy.spatial import ConvexHull
import difflib
import shutil

from subsequence_dtw import SubsequenceDTW


class CustomLinearRegionItem(pg.LinearRegionItem):
    sigHoverEvent = QtCore.pyqtSignal(object)

    def hoverEvent(self, ev):
        if (
            self.movable
            and (not ev.isExit())
            and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)
        ):
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)
        self.sigHoverEvent.emit(self)


@dataclass
class SentenceRegion:
    region_items: list[CustomLinearRegionItem] = field(default_factory=list)
    text_item: pg.TextItem = field(
        default_factory=lambda: pg.TextItem(
            "Text here", anchor=(0.5, 0.5), color=(0, 0, 0), fill=(255, 255, 255, 100)
        )
    )


@dataclass
class PhonemeRegion:
    region_items: list[CustomLinearRegionItem] = field(default_factory=list)
    text_item: pg.TextItem = field(
        default_factory=lambda: pg.TextItem(
            "Text here", anchor=(0.5, 0.5), color=(0, 0, 0), fill=(255, 255, 255, 100)
        )
    )


class CompactLegendItem(pg.GraphicsWidget, pg.GraphicsWidgetAnchor):
    """Compact bottom-right legend displaying only each vowel's colored shape and label."""

    def __init__(self, entries: list[tuple[str, tuple[int, int, int], tuple[int, int, int], str]], col_count: int = 3):
        pg.GraphicsWidget.__init__(self)
        pg.GraphicsWidgetAnchor.__init__(self)
        self.entries = entries
        self.col_count = max(1, min(col_count, len(entries))) if entries else 1
        self.row_count = (
            (len(entries) + self.col_count - 1) // self.col_count if entries else 0
        )
        self.row_h = 15
        self.col_w = 34
        self.pad_x = 5
        self.pad_y = 4
        w = self.pad_x * 2 + self.col_count * self.col_w if entries else 0
        h = self.pad_y * 2 + self.row_count * self.row_h if entries else 0
        self.setGeometry(0, 0, w, h)
        self.setZValue(100)

    def paint(self, p, *args):
        if not self.entries:
            return
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.boundingRect()
        p.setPen(pg.mkPen((180, 180, 180, 200), width=1))
        p.setBrush(pg.mkBrush(255, 255, 255, 220))
        p.drawRoundedRect(rect, 3, 3)
        font = QtGui.QFont("Segoe UI", 8)
        p.setFont(font)
        for idx, (label, color, border_color, symbol) in enumerate(self.entries):
            r = idx // self.col_count
            c = idx % self.col_count
            x0 = self.pad_x + c * self.col_w
            y0 = self.pad_y + r * self.row_h
            cx = x0 + 6
            cy = y0 + self.row_h / 2.0
            p.save()
            p.translate(cx, cy)
            sym_size = 9.5 if symbol.startswith("t") else 8.5
            pg.graphicsItems.ScatterPlotItem.drawSymbol(
                p,
                symbol,
                sym_size,
                pg.mkPen(border_color, width=1.1),
                pg.mkBrush(*color, 195),
            )
            p.restore()
            p.setPen(pg.mkPen(0, 0, 0))
            text_rect = QtCore.QRectF(x0 + 14, y0, self.col_w - 14, self.row_h)
            p.drawText(
                text_rect,
                int(
                    QtCore.Qt.AlignmentFlag.AlignLeft
                    | QtCore.Qt.AlignmentFlag.AlignVCenter
                ),
                label,
            )


class AutoSpeechAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        # # Initialize pygame for audio playback
        # pygame.mixer.init()

        # GUI Setup
        self.setWindowTitle("Audio Player & Visualizer")
        self.setGeometry(100, 100, 900, 600)

        # central_widget = QWidget()
        # layout = QVBoxLayout()
        main_scroll_area = QScrollArea()
        main_scroll_area.setWidgetResizable(True)
        main_scroll_content = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)

        # Menu Bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Open Audio")
        file_menu.addAction("Exit")
        file_menu.triggered.connect(self.menu_actions)

        # PyQtGraph Waveform Plot
        self.waveform_plot = pg.PlotWidget(title="Audio Waveform", background="w")
        self.waveform_plot.setMinimumHeight(200)
        self.waveform_plot.showGrid(x=True, y=True)
        self.waveform_plot.setLabel("bottom", "Time", units="s")
        self.waveform_plot.setLabel("left", "Amplitude")
        layout.addWidget(self.waveform_plot, stretch=1)

        # VLC player
        self.instance = vlc.Instance()
        self.media_player = self.instance.media_player_new()

        # Playback Controls
        self.audio_buttons = QWidget()
        self.audio_buttons_layout = QHBoxLayout()

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        self.audio_buttons_layout.addWidget(self.play_button)

        self.play_viewbox_button = QPushButton("Play window")
        self.play_viewbox_button.clicked.connect(self.play_audio_in_viewbox)
        self.audio_buttons_layout.addWidget(self.play_viewbox_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_audio)
        self.audio_buttons_layout.addWidget(self.stop_button)

        self.load_stimulus_button = QPushButton("Load Stimulus")
        self.load_stimulus_button.clicked.connect(self.load_stimulus)
        self.audio_buttons_layout.addWidget(self.load_stimulus_button)

        self.run_asr_button = QPushButton("Run ASR")
        self.run_asr_button.clicked.connect(self.show_asr_menu)
        self.audio_buttons_layout.addWidget(self.run_asr_button)

        self.run_mfa_button = QPushButton("Run MFA")
        self.run_mfa_button.clicked.connect(self.run_MFA)
        self.audio_buttons_layout.addWidget(self.run_mfa_button)

        self.load_landmark_info_button = QPushButton("Load Landmark Info")
        self.load_landmark_info_button.clicked.connect(self.show_landmark_info_menu)
        self.audio_buttons_layout.addWidget(self.load_landmark_info_button)

        self.export_formant_targets_button = QPushButton("Export Formant Targets")
        self.export_formant_targets_button.clicked.connect(
            self.export_phoneme_formant_target_data
        )
        self.audio_buttons_layout.addWidget(self.export_formant_targets_button)

        self.audio_buttons.setLayout(self.audio_buttons_layout)
        layout.addWidget(self.audio_buttons)

        self.sentence_descriptor = QScrollArea()
        self.sentence_descriptor.setWidgetResizable(True)

        self.sentence_descriptor_content = QWidget()
        self.sentence_descriptor_layout = QVBoxLayout()

        self.audio_file_path_label = QLabel("Audio Path: ")
        self.sentence_descriptor_layout.addWidget(self.audio_file_path_label)

        self.stimulus_file_path_label = QLabel("Stimulus Path: ")
        self.sentence_descriptor_layout.addWidget(self.stimulus_file_path_label)

        self.sentences_textgrid_file_path_label = QLabel("Sentences TextGrid Path: ")
        self.sentence_descriptor_layout.addWidget(
            self.sentences_textgrid_file_path_label
        )

        self.detected_sentences_path_label = QLabel("Sentences Path: ")
        self.sentence_descriptor_layout.addWidget(self.detected_sentences_path_label)

        self.landmark_info_file_path_label = QLabel("Landmark Info Path: ")
        self.sentence_descriptor_layout.addWidget(self.landmark_info_file_path_label)

        self.sentence_descriptor_content.setLayout(self.sentence_descriptor_layout)

        self.sentence_descriptor.setWidget(self.sentence_descriptor_content)
        self.sentence_descriptor.setMinimumHeight(100)

        layout.addWidget(self.sentence_descriptor)

        self.sentence_adder = QWidget()
        self.sentence_adder_layout = QHBoxLayout()

        self.sentence_text_input = QLineEdit()
        self.sentence_text_input.setPlaceholderText("Enter sentence")
        self.sentence_adder_layout.addWidget(self.sentence_text_input)

        self.sentence_start_input = QLineEdit()
        self.sentence_start_input.setPlaceholderText("Start time (s)")
        self.sentence_adder_layout.addWidget(self.sentence_start_input)

        self.sentence_end_input = QLineEdit()
        self.sentence_end_input.setPlaceholderText("End time (s)")
        self.sentence_adder_layout.addWidget(self.sentence_end_input)

        self.add_sentence_button = QPushButton("Add sentence")
        self.add_sentence_button.clicked.connect(self.load_sentence_from_ui)
        self.sentence_adder_layout.addWidget(self.add_sentence_button)

        self.sentence_adder.setLayout(self.sentence_adder_layout)
        layout.addWidget(self.sentence_adder)

        # PyQtGraph Sentence Plot
        self.sentence_label_plot = pg.PlotWidget(
            title="Sentence Labels", background="w"
        )
        self.sentence_label_plot.showGrid(x=True, y=False)
        # self.sentence_label_plot.hideAxis('left')
        self.sentence_label_plot.setLabel("bottom", "Time", units="s")
        self.sentence_label_plot.setFixedHeight(80)
        layout.addWidget(self.sentence_label_plot, stretch=0)

        # PyQtGraph Spectrogram Plot
        self.spectrogram_plot = pg.PlotWidget(title="Spectrogram", background="w")
        self.spectrogram_plot.setFixedHeight(400)
        self.spectrogram_plot.showGrid(x=True, y=True)
        self.spectrogram_plot.setLabel("bottom", "Time", units="s")
        self.spectrogram_plot.setLabel("left", "Frequency", units="Hz")
        self.spectogram_img = pg.ImageItem(axisOrder="row-major")
        self.spectrogram_plot.addItem(self.spectogram_img)

        layout.addWidget(self.spectrogram_plot, stretch=1)

        # PyQtGraph Phoneme Plot
        self.phoneme_label_plot = pg.PlotWidget(title="Phoneme Labels", background="w")
        self.phoneme_label_plot.showGrid(x=True, y=False)
        self.phoneme_label_plot.setLabel("bottom", "Time", units="s")
        self.phoneme_label_plot.setFixedHeight(100)

        layout.addWidget(self.phoneme_label_plot, stretch=0)

        # Phoneme adder UI
        self.phoneme_adder = QWidget()
        self.phoneme_adder_layout = QHBoxLayout()

        self.phoneme_text_input = QLineEdit()
        self.phoneme_text_input.setPlaceholderText("Enter phoneme")
        self.phoneme_adder_layout.addWidget(self.phoneme_text_input)

        self.phoneme_start_input = QLineEdit()
        self.phoneme_start_input.setPlaceholderText("Start time (s)")
        self.phoneme_adder_layout.addWidget(self.phoneme_start_input)

        self.phoneme_end_input = QLineEdit()
        self.phoneme_end_input.setPlaceholderText("End time (s)")
        self.phoneme_adder_layout.addWidget(self.phoneme_end_input)

        self.phoneme_parent_sentence_input = QLineEdit()
        self.phoneme_parent_sentence_input.setPlaceholderText("Parent sentence")
        self.phoneme_adder_layout.addWidget(self.phoneme_parent_sentence_input)

        self.phoneme_parent_word_input = QLineEdit()
        self.phoneme_parent_word_input.setPlaceholderText("Parent word")
        self.phoneme_adder_layout.addWidget(self.phoneme_parent_word_input)

        self.add_phoneme_button = QPushButton("Add phoneme")
        self.add_phoneme_button.clicked.connect(self.load_phoneme_from_ui)
        self.phoneme_adder_layout.addWidget(self.add_phoneme_button)

        self.phoneme_adder.setLayout(self.phoneme_adder_layout)
        layout.addWidget(self.phoneme_adder)

        self.phoneme_descriptor = QScrollArea()
        self.phoneme_descriptor.setWidgetResizable(True)

        self.phoneme_descriptor_content = QWidget()
        self.phoneme_descriptor_layout = QVBoxLayout()

        self.phoneme_descriptor_content.setLayout(self.phoneme_descriptor_layout)

        self.phoneme_descriptor.setWidget(self.phoneme_descriptor_content)
        self.phoneme_descriptor.setMinimumHeight(100)

        layout.addWidget(self.phoneme_descriptor)

        # Formant scatter plot widget
        self.current_gender = "F"
        self.formant_scatter_widget = QWidget()
        self.formant_scatter_layout = QVBoxLayout()
        self.formant_scatter_widget.setLayout(self.formant_scatter_layout)
        # Axis selectors
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("X Axis (Formant):"))
        self.x_axis_combo = QComboBox()
        axis_layout.addWidget(self.x_axis_combo)
        axis_layout.addWidget(QLabel("Y Axis (Formant):"))
        self.y_axis_combo = QComboBox()
        axis_layout.addWidget(self.y_axis_combo)
        for i in range(5):
            self.x_axis_combo.addItem(f"Formant {i+1}", i)
            self.y_axis_combo.addItem(f"Formant {i+1}", i)
        self.x_axis_combo.setCurrentIndex(1)  # Default X: Formant 2
        self.y_axis_combo.setCurrentIndex(0)  # Default Y: Formant 1
        self.formant_scatter_layout.addLayout(axis_layout)
        # Phoneme selector
        self.formant_scatter_layout.addWidget(QLabel("Select phonemes to plot:"))
        self.phoneme_picker_list_widget = QListWidget()
        self.phoneme_picker_list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self.formant_scatter_layout.addWidget(self.phoneme_picker_list_widget)

        # Create a horizontal layout for the scatter plots
        scatter_plots_row = QHBoxLayout()
        self.formant_scatter_layout.addLayout(scatter_plots_row)

        # Phoneme scatter plots in one row
        # 1. Monophthong plot with its export button
        monophthong_container = QWidget()
        monophthong_layout = QVBoxLayout(monophthong_container)
        monophthong_layout.setContentsMargins(0, 0, 0, 0)

        self.monophthong_scatter_plot = pg.PlotWidget(
            title="Monophthongs", background="w"
        )
        self.monophthong_scatter_plot.setFixedHeight(400)  # larger vertical spread
        self.monophthong_scatter_plot.setFixedWidth(
            400
        )  # not filling entire horizontal space
        self.monophthong_scatter_plot.setLimits(xMin=0, yMin=0)  # first quadrant only
        self.monophthong_scatter_plot.invertY(True)  # Invert Y-axis for vowel plot
        self.monophthong_scatter_plot.invertX(True)  # Invert X-axis for vowel plot
        self.monophthong_scatter_plot.showGrid(x=True, y=True)
        self.monophthong_scatter_plot.getAxis("bottom").enableAutoSIPrefix(True)
        self.monophthong_scatter_plot.getAxis("left").enableAutoSIPrefix(True)
        self.monophthong_scatter_plot.setLabel(
            "bottom", self.x_axis_combo.currentText(), units="Hz"
        )
        self.monophthong_scatter_plot.setLabel(
            "left", self.y_axis_combo.currentText(), units="Hz"
        )
        self.monophthong_scatter_plot.getViewBox().sigRangeChanged.connect(
            lambda vb, ranges: self.update_scatter_label_visibility(
                self.monophthong_scatter_plot
            )
        )
        monophthong_layout.addWidget(
            self.monophthong_scatter_plot, alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )

        # Add export button for monophthongs directly under the plot
        self.monophthong_export_btn = QPushButton("Export Monophthongs CSV")
        self.monophthong_export_btn.setFixedWidth(400)  # Same width as the plot
        self.monophthong_export_btn.clicked.connect(
            lambda: self.export_plot_data_to_csv(self.monophthong_scatter_plot)
        )
        monophthong_layout.addWidget(
            self.monophthong_export_btn, alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )

        scatter_plots_row.addWidget(monophthong_container)

        # 2. Centring diphthong plot with its export button
        centring_container = QWidget()
        centring_layout = QVBoxLayout(centring_container)
        centring_layout.setContentsMargins(0, 0, 0, 0)

        self.centring_diphthong_scatter_plot = pg.PlotWidget(
            title="Centring Diphthongs", background="w"
        )
        self.centring_diphthong_scatter_plot.setFixedHeight(
            400
        )  # larger vertical spread
        self.centring_diphthong_scatter_plot.setFixedWidth(
            400
        )  # not filling entire horizontal space
        self.centring_diphthong_scatter_plot.setLimits(
            xMin=0, yMin=0
        )  # first quadrant only
        self.centring_diphthong_scatter_plot.invertY(
            True
        )  # Invert Y-axis for vowel plot
        self.centring_diphthong_scatter_plot.invertX(
            True
        )  # Invert X-axis for vowel plot
        self.centring_diphthong_scatter_plot.showGrid(x=True, y=True)
        self.centring_diphthong_scatter_plot.getAxis("bottom").enableAutoSIPrefix(True)
        self.centring_diphthong_scatter_plot.getAxis("left").enableAutoSIPrefix(True)
        self.centring_diphthong_scatter_plot.setLabel(
            "bottom", self.x_axis_combo.currentText(), units="Hz"
        )
        self.centring_diphthong_scatter_plot.setLabel(
            "left", self.y_axis_combo.currentText(), units="Hz"
        )
        self.centring_diphthong_scatter_plot.getViewBox().sigRangeChanged.connect(
            lambda vb, ranges: self.update_scatter_label_visibility(
                self.centring_diphthong_scatter_plot
            )
        )
        centring_layout.addWidget(
            self.centring_diphthong_scatter_plot,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )

        # Add export button for centring diphthongs directly under the plot
        self.centring_diphthong_export_btn = QPushButton(
            "Export Centring Diphthongs CSV"
        )
        self.centring_diphthong_export_btn.setFixedWidth(400)  # Same width as the plot
        self.centring_diphthong_export_btn.clicked.connect(
            lambda: self.export_plot_data_to_csv(self.centring_diphthong_scatter_plot)
        )
        centring_layout.addWidget(
            self.centring_diphthong_export_btn,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )

        scatter_plots_row.addWidget(centring_container)

        # 3. Rising diphthong plot with its export button
        rising_container = QWidget()
        rising_layout = QVBoxLayout(rising_container)
        rising_layout.setContentsMargins(0, 0, 0, 0)

        self.rising_diphthong_scatter_plot = pg.PlotWidget(
            title="Rising Diphthongs", background="w"
        )
        self.rising_diphthong_scatter_plot.setFixedHeight(400)  # larger vertical spread
        self.rising_diphthong_scatter_plot.setFixedWidth(
            400
        )  # not filling entire horizontal space
        self.rising_diphthong_scatter_plot.setLimits(
            xMin=0, yMin=0
        )  # first quadrant only
        self.rising_diphthong_scatter_plot.invertY(True)  # Invert Y-axis for vowel plot
        self.rising_diphthong_scatter_plot.invertX(True)  # Invert X-axis for vowel plot
        self.rising_diphthong_scatter_plot.showGrid(x=True, y=True)
        self.rising_diphthong_scatter_plot.getAxis("bottom").enableAutoSIPrefix(True)
        self.rising_diphthong_scatter_plot.getAxis("left").enableAutoSIPrefix(True)
        self.rising_diphthong_scatter_plot.setLabel(
            "bottom", self.x_axis_combo.currentText(), units="Hz"
        )
        self.rising_diphthong_scatter_plot.setLabel(
            "left", self.y_axis_combo.currentText(), units="Hz"
        )
        self.rising_diphthong_scatter_plot.getViewBox().sigRangeChanged.connect(
            lambda vb, ranges: self.update_scatter_label_visibility(
                self.rising_diphthong_scatter_plot
            )
        )
        rising_layout.addWidget(
            self.rising_diphthong_scatter_plot,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )

        # Add export button for rising diphthongs directly under the plot
        self.rising_diphthong_export_btn = QPushButton("Export Rising Diphthongs CSV")
        self.rising_diphthong_export_btn.setFixedWidth(400)  # Same width as the plot
        self.rising_diphthong_export_btn.clicked.connect(
            lambda: self.export_plot_data_to_csv(self.rising_diphthong_scatter_plot)
        )
        rising_layout.addWidget(
            self.rising_diphthong_export_btn,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )

        scatter_plots_row.addWidget(rising_container)

        # Add gender toggle button for ground truth quadrilateral
        self.gender_toggle_button = QPushButton("Gender: F")
        self.gender_toggle_button.setCheckable(True)
        self.gender_toggle_button.setChecked(self.current_gender == "F")
        self.gender_toggle_button.clicked.connect(self.toggle_gender)
        self.formant_scatter_layout.addWidget(self.gender_toggle_button)

        layout.addWidget(self.formant_scatter_widget)

        # central_widget.setLayout(layout)
        # self.setCentralWidget(central_widget)
        main_scroll_content.setLayout(layout)
        main_scroll_area.setWidget(main_scroll_content)
        self.setCentralWidget(main_scroll_area)

        # Data Variables
        self.AUDIO_FILE = None
        self.STIMULUS_FILE = None
        self.LANDMARK_INFO_FILE = None
        self.SENTENCES_TEXTGRID_FILE = None
        self.COMBINED_PHONEMES_TEXTGRID_FILE = None
        self.SENTENCES_DIR = None
        self.SENTENCES_PHONEMES_DIR = None
        self.PHONEME_GROUND_TRUTH_FILEPATH = "landmark_identification_ground_truth.csv"
        self.phoneme_ground_truth: list[dict] = []
        self.PHONEME_MAPPING_FILEPATH = "landmark_identification_phoneme_mapping.csv"
        self.british_to_australian_mapping: dict = {}
        self.timestamp_tolerance = 0.05
        self.whisper_ASR = WhisperX_ASR(
            "tiny", timestamp_tolerance=self.timestamp_tolerance
        )
        self.subseq_dtw_ASR = SubsequenceDTW(
            hop_length=512, sr=16000, timestamp_tolerance=self.timestamp_tolerance
        )
        self.ASR: WhisperX_ASR | SubsequenceDTW = self.whisper_ASR
        self.formant_cleaning_algo: Literal["viterbi"] | None = None # "median" not used
        self.MFA_ENVIRONMENT_NAME = "mfa_env1"
        self.MFA_ALIGNER_SCRIPT_PATH = r"mfa_align.py"
        self.CONDA_EXE_PATH = r"conda"  # or full path to conda.exe if not in PATH e.g. r"C:\Users\<YourUsername>\anaconda3\Scripts\conda.exe"
        self.SPECTROGRAM_WINDOW_LENGTH = 0.005
        self.USE_GROUND_TRUTH: bool = False
        self.playing = False
        self.sentence_timestamps: list[SentenceTimestamp] = []
        self.sentence_regions: dict[str, SentenceRegion] = {}
        self.sentence_descriptor_items: dict[str, tuple[QWidget, QLabel]] = {}
        self.formant_data: FormantData | None = None
        self.phoneme_timestamps: list[PhonemeTimestamp] = []
        self.phoneme_regions: dict[str, PhonemeRegion] = {}
        self.phoneme_descriptor_items: dict[str, tuple[QWidget, QLabel]] = {}
        # self.monophthongs=['ɪ','e','eː','æ','ɐ','ɐː','ʉː','ɜː','ɔ','oː','ʊ']
        # self.centring_diphthongs=['æɪ','ɑe','oɪ','ɪə']
        # self.rising_diphthongs=['æɔ','əʉ','iː']

        self.progress_line = None
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.sound = None

        # Load phoneme mapping on initialization
        self.load_phoneme_mapping()

    def load_phoneme_mapping(self):
        """Load the British English to Australian English phoneme mapping from CSV."""
        try:
            if os.path.exists(self.PHONEME_MAPPING_FILEPATH):
                with open(
                    self.PHONEME_MAPPING_FILEPATH, "r", encoding="utf-8"
                ) as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        british_phoneme = row["British_English"].strip()
                        australian_phoneme = row["Australian_English"].strip()
                        # Create mapping from British to Australian
                        self.british_to_australian_mapping[british_phoneme] = (
                            australian_phoneme
                        )
                print(
                    f"Loaded {len(self.british_to_australian_mapping)} phoneme mappings"
                )
            else:
                print(
                    f"Warning: Phoneme mapping file not found at {self.PHONEME_MAPPING_FILEPATH}"
                )
        except Exception as e:
            print(f"Error loading phoneme mapping: {e}")

    def map_british_to_australian(self, phoneme):
        """Convert British English phoneme to Australian English format."""
        return self.british_to_australian_mapping.get(phoneme, phoneme)

    def menu_actions(self, action):
        if action.text() == "Open Audio":
            self.load_audio()
        elif action.text() == "Exit":
            sys.exit()

    def load_audio(self):
        audio_file, _ = QFileDialog.getOpenFileName(
            self, "Load Audio", "", "WAV Files (*.wav)"
        )

        if audio_file:
            self.stop_audio()
            self.AUDIO_FILE = audio_file
            self.audio_file_path_label.setText(f"Audio Path: {self.AUDIO_FILE}")
            self.STIMULUS_FILE = None
            self.stimulus_file_path_label.setText("Stimulus Path: ")
            # pygame.mixer.music.load(self.AUDIO_FILE)
            media = self.instance.media_new(self.AUDIO_FILE)
            media.add_option(":no-video")
            self.media_player.set_media(media)

            self.clear_sentences()
            self.clear_phonemes()
            self.clear_formant_data()
            self.progress_line = None

            # Load and preprocess audio
            self.sound = Sound(self.AUDIO_FILE)
            self.sound = self.sound.extract_channel(1)
            self.formant_data = FormantData(
                sound=self.sound,
                n_formants=5,
                sampling_rate=self.sound.sampling_frequency,
                ignore_f0=True,
            )

            # Plot waveform and spectrogram
            self.plot_waveform()
            self.plot_spectrogram()
            self.plot_phoneme_labels()

            # Load TextGrid file for sentences
            self.SENTENCES_TEXTGRID_FILE = self.AUDIO_FILE.replace(".wav", ".TextGrid")
            self.SENTENCES_DIR = os.path.join(os.getcwd(), MFA_DEFAULT_EXPERIMENT_NAME)
            self.SENTENCES_PHONEMES_DIR = os.path.join(self.SENTENCES_DIR, "aligned")
            self.COMBINED_PHONEMES_TEXTGRID_FILE = os.path.join(
                self.SENTENCES_PHONEMES_DIR, "phones.TextGrid"
            )
            if os.path.exists(self.SENTENCES_DIR):
                # Use shutil to remove directory tree instead of os.removedirs
                shutil.rmtree(self.SENTENCES_DIR, ignore_errors=True)
            os.makedirs(self.SENTENCES_DIR, exist_ok=False)
            self.sentences_textgrid_file_path_label.setText(
                f"Sentences TextGrid Path: {self.SENTENCES_TEXTGRID_FILE}"
            )
            self.load_sentence_timestamps_from_textgrid()
            self.validate_and_display_all_sentences()

    def load_sentence_timestamps_from_textgrid(self):
        if os.path.exists(self.SENTENCES_TEXTGRID_FILE):
            tg = textgrid.openTextgrid(
                self.SENTENCES_TEXTGRID_FILE, includeEmptyIntervals=False
            )
            for tier in tg:
                if isinstance(tier, textgrid.IntervalTier) and tier.name == "sentences":
                    for interval in tier:
                        start, end, sentence = (
                            interval.start,
                            interval.end,
                            interval.label,
                        )
                        # Ensure proper UTF-8 encoding for sentence text
                        sentence = ensure_utf8_display(sentence)
                        self.add_sentence_timestamp(
                            sentence, start_time=start, end_time=end
                        )

    def toggle_playback(self):
        if self.playing:
            self.pause_audio()
        else:
            self.play_audio()

    def play_audio(self, set_time=None):  # set_time in ms
        if self.AUDIO_FILE:
            if not self.playing:
                current_time = self.media_player.get_time()  # in ms
                viewbox_end_time = (
                    self.waveform_plot.getViewBox().viewRange()[0][1] * 1000
                )  # Convert seconds to ms
                if current_time >= viewbox_end_time:
                    set_time = (
                        self.waveform_plot.getViewBox().viewRange()[0][0] * 1000
                    )  # Convert seconds to ms
                self.media_player.play()
                self.playing = True
                if set_time is not None:
                    duration = self.sound.get_total_duration()
                    set_time = max(
                        0, min(set_time, duration * 1000)
                    )  # Ensure set_time is within bounds
                    set_time = int(set_time)
                    self.media_player.set_time(set_time)
                    print("Setting time to:", set_time)
                self.play_button.setText("Pause")
                self.timer.start(10)  # Update every 10ms

    def play_audio_in_viewbox(self):
        if self.AUDIO_FILE:
            self.pause_audio()
            viewbox_start_time = (
                self.waveform_plot.getViewBox().viewRange()[0][0] * 1000
            )  # Convert seconds to ms
            print("Viewbox start time:", viewbox_start_time)
            viewbox_start_time = int(viewbox_start_time)
            print("Playing audio from:", viewbox_start_time)
            self.play_audio(set_time=viewbox_start_time)

    def pause_audio(self):
        if self.playing:
            # pygame.mixer.music.pause()
            self.timer.stop()
            self.media_player.pause()
            self.playing = False
            self.play_button.setText("Play")

    def stop_audio(self):
        # pygame.mixer.music.stop()
        if self.AUDIO_FILE:
            self.timer.stop()
            self.media_player.stop()
            self.playing = False
            self.play_button.setText("Play")
            self.progress_line.setValue(0)

    def load_stimulus(self):
        self.STIMULUS_FILE, _ = QFileDialog.getOpenFileName(
            self, "Load Stimulus", "", "TXT Files (*.txt)"
        )
        if self.STIMULUS_FILE:
            self.stimulus_file_path_label.setText(
                f"Stimulus Path: {self.STIMULUS_FILE}"
            )

    def show_landmark_info_menu(self):
        """Show a dropdown menu to select formant cleaning algorithm."""
        menu = QMenu(self)

        viterbi_menu = menu.addMenu("with Viterbi cleaning")
        viterbi_with_gt_action = viterbi_menu.addAction("with Ground Truth")
        viterbi_without_gt_action = viterbi_menu.addAction("without Ground Truth")
        # median_action = menu.addAction("with Median cleaning") # Not used
        no_cleaning_action = menu.addAction("with NO cleaning")

        # Show menu at button position
        button_pos = self.load_landmark_info_button.mapToGlobal(
            self.load_landmark_info_button.rect().bottomLeft()
        )
        action = menu.exec(button_pos)

        # Execute based on selection
        if action == viterbi_with_gt_action:
            self.load_landmark_info("viterbi", use_ground_truth=True)
        elif action == viterbi_without_gt_action:
            self.load_landmark_info("viterbi", use_ground_truth=False)
        # elif action == median_action:
        #     self.load_landmark_info("median") # "median" not used
        elif action == no_cleaning_action:
            self.load_landmark_info(None)

    def load_landmark_info(
        self, cleaning_algo: Literal["viterbi"] | None, use_ground_truth: bool | None = None
    ):
        """
        Load landmark info and set formant cleaning algorithm.

        Args:
            cleaning_algo: Either "viterbi" or None for no cleaning
            use_ground_truth: Whether to use ground truth targets for Viterbi emission cost
        """
        self.LANDMARK_INFO_FILE, _ = QFileDialog.getOpenFileName(
            self, "Load Landmark Info", "", "CSV Files (*.csv)"
        )
        if self.LANDMARK_INFO_FILE:
            if use_ground_truth is not None:
                self.USE_GROUND_TRUTH = use_ground_truth

            # Set the formant cleaning algorithm
            if cleaning_algo == "viterbi":
                self.formant_cleaning_algo = "viterbi"
                print(
                    f"Using Viterbi cleaning algorithm (use_ground_truth={self.USE_GROUND_TRUTH})"
                )
            # elif cleaning_algo == "median": # "median" not used
            #     self.formant_cleaning_algo = "median"
            #     print("Using Median cleaning algorithm")
            else:
                self.formant_cleaning_algo = None
                print("No formant cleaning algorithm selected")

            self.landmark_info_file_path_label.setText(
                f"Landmark Info Path: {self.LANDMARK_INFO_FILE}"
            )
            self.set_formant_targets()

    def show_asr_menu(self):
        """Show a dropdown menu to select ASR method (and voice model for Subsequence DTW)."""
        menu = QMenu(self)

        whisper_action = menu.addAction("Whisper AI")
        dtw_menu = menu.addMenu("Subsequence DTW")

        dtw_voice_actions = {
            dtw_menu.addAction("Piper TTS - Male (Alan)"): "piper_male",
            dtw_menu.addAction("Piper TTS - Female (Alba)"): "piper_female",
            dtw_menu.addAction("gTTS (Google TTS)"): "gtts",
        }
        kitten_menu = dtw_menu.addMenu("KittenTTS")
        for kitten_voice in (
            "expr-voice-2-f",
            "expr-voice-2-m",
            "expr-voice-3-f",
            "expr-voice-3-m",
            "expr-voice-4-f",
            "expr-voice-4-m",
            "expr-voice-5-f",
            "expr-voice-5-m",
        ):
            gender_label = "Female" if kitten_voice.endswith("-f") else "Male"
            dtw_voice_actions[
                kitten_menu.addAction(f"{kitten_voice} ({gender_label})")
            ] = f"kitten:{kitten_voice}"

        # Show menu at button position
        button_pos = self.run_asr_button.mapToGlobal(
            self.run_asr_button.rect().bottomLeft()
        )
        action = menu.exec(button_pos)

        # Execute based on selection
        if action == whisper_action:
            self.run_ASR("whisper")
        elif action in dtw_voice_actions:
            self.run_ASR("dtw", dtw_voice_model=dtw_voice_actions[action])

    def show_error_dialog(
        self, title: str, error: Exception | str, terminated_operation: str
    ):
        """Display a closeable error dialog describing the error and which subsequent operation was terminated."""
        message = (
            f"Error:\n{error}\n\n"
            f"Terminated operation:\n{terminated_operation}"
        )
        print(f"{title}: {error} | Terminated: {terminated_operation}")
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Close)
        msg_box.exec()

    def run_ASR(self, asr_method="whisper", dtw_voice_model: str = "piper_male"):
        """
        Run ASR with the specified method.

        Args:
            asr_method: Either "whisper" for Whisper AI or "dtw" for Subsequence DTW
            dtw_voice_model: Voice model to use when asr_method == "dtw"
        """
        if not self.AUDIO_FILE or not os.path.exists(self.AUDIO_FILE):
            print("Error: No valid audio file provided.")
            return
        if not self.STIMULUS_FILE or not os.path.exists(self.STIMULUS_FILE):
            print("Error: No valid audio or stimulus file provided.")
            return

        # Assign ASR based on selected method
        if asr_method.lower() == "whisper":
            self.ASR = self.whisper_ASR
            print("Using Whisper AI for ASR")
        elif asr_method.lower() == "dtw":
            self.subseq_dtw_ASR.set_voice_model(dtw_voice_model)
            self.ASR = self.subseq_dtw_ASR
            print(f"Using Subsequence DTW for ASR (voice model: {dtw_voice_model})")
        else:
            print(f"Unknown ASR method: {asr_method}. Defaulting to Whisper AI.")
            self.ASR = self.whisper_ASR

        self.ASR.set_asr_target(
            self.AUDIO_FILE,
            stimulus_path=self.STIMULUS_FILE,
            textgrid_path=self.SENTENCES_TEXTGRID_FILE,
        )
        if len(self.sentence_timestamps) > 0:
            print(
                "Warning: Sentence timestamps already loaded. They will NOT be replaced."
            )
        else:
            self.sentence_timestamps = self.ASR.get_valid_sentence_timestamps()
            self.validate_and_display_all_sentences()
            print("Sentence timestamps loaded.")
            try:
                self.ASR.save_sentences_as_textgrid(self.sentence_timestamps)
                print("Sentence timestamps saved to TextGrid.")
            except Exception as e:
                self.show_error_dialog(
                    "Sentence TextGrid Save Error",
                    e,
                    "Saving detected sentence timestamps to TextGrid (ASR sentence export).",
                )
                return
        print(self.sentence_timestamps)

    def validate_and_display_all_sentences(self):
        valid_timestamps: list[SentenceTimestamp] = []
        for sentence_timestamp in self.sentence_timestamps:
            sentence, start_time, end_time = (
                sentence_timestamp.sentence,
                sentence_timestamp.start,
                sentence_timestamp.end,
            )
            if (
                not sentence
                or start_time is None
                or end_time is None
                or start_time >= end_time
                or start_time < 0
                or end_time > self.sound.get_total_duration()
            ):
                continue
            valid_timestamps.append(sentence_timestamp)
            self.validate_and_display_sentence(sentence_timestamp)
        valid_timestamps.sort(key=lambda x: x.start)
        self.sentence_timestamps = valid_timestamps
        if len(self.sentence_timestamps) > 0:
            print(f"Displaying {len(self.sentence_timestamps)} valid sentences.")

    def validate_and_display_sentence(self, sentence_timestamp: SentenceTimestamp):
        if not sentence_timestamp:
            return
        sentence, start_time, end_time = (
            sentence_timestamp.sentence,
            sentence_timestamp.start,
            sentence_timestamp.end,
        )
        if (
            not sentence
            or start_time is None
            or end_time is None
            or start_time >= end_time
            or start_time < 0
            or end_time > self.sound.get_total_duration()
        ):
            return
        self.add_sentence_region(sentence, start_time=start_time, end_time=end_time)
        self.add_sentence_to_ui_list(sentence, start=start_time, end=end_time)

    def combine_phoneme_textgrids(self, textgrid_offsets):
        tg = textgrid.Textgrid()
        tg_phoneme_entries = []
        tg_word_entries = []
        tg_sentence_entries = []
        for textgrid_offset in textgrid_offsets:
            textgrid_path, sentence, start_time, end_time = textgrid_offset
            if os.path.exists(textgrid_path):
                tg_temp = textgrid.openTextgrid(
                    textgrid_path, includeEmptyIntervals=False
                )
                for tier in tg_temp:
                    if (
                        isinstance(tier, textgrid.IntervalTier)
                        and tier.name == "phones"
                    ):
                        intervals = list(tier)  # Convert tier to list for indexing
                        skip_next = False
                        for idx, interval in enumerate(intervals):
                            start, end, phoneme = (
                                interval.start,
                                interval.end,
                                interval.label,
                            )
                            # Ensure proper UTF-8 encoding for phoneme text
                            phoneme = ensure_utf8_display(phoneme)
                            if not phoneme or skip_next:
                                skip_next = False
                                continue
                            if phoneme == "ɪ":
                                next_phoneme = (
                                    intervals[idx + 1].label
                                    if idx + 1 < len(intervals)
                                    else None
                                )
                                if next_phoneme and next_phoneme == "ə":
                                    phoneme = "ɪə"
                                    end = intervals[idx + 1].end
                                    skip_next = True

                            # Apply British to Australian English phoneme mapping
                            phoneme = self.map_british_to_australian(phoneme)

                            start += start_time
                            end += start_time
                            tg_phoneme_entries.append((start, end, phoneme))
                    if isinstance(tier, textgrid.IntervalTier) and tier.name == "words":
                        for interval in tier:
                            start, end, word = (
                                interval.start,
                                interval.end,
                                interval.label,
                            )
                            # Ensure proper UTF-8 encoding for word text
                            word = ensure_utf8_display(word)
                            if not word:
                                continue
                            start += start_time
                            end += start_time
                            tg_word_entries.append((start, end, word))
                tg_sentence_entries.append((start_time, end_time, sentence))
        if not tg_phoneme_entries:
            raise ValueError("Error: No valid entries found.")

        audio_duration = Sound(self.AUDIO_FILE).get_total_duration()
        phoneme_tier = textgrid.IntervalTier(
            name="phones", entries=tg_phoneme_entries, minT=0, maxT=audio_duration
        )
        word_tier = textgrid.IntervalTier(
            name="words", entries=tg_word_entries, minT=0, maxT=audio_duration
        )
        sentence_tier = textgrid.IntervalTier(
            name="sentences", entries=tg_sentence_entries, minT=0, maxT=audio_duration
        )

        tg.addTier(sentence_tier)
        tg.addTier(word_tier)
        tg.addTier(phoneme_tier)
        # Force UTF-8 encoding when saving TextGrid files
        try:
            tg.save(
                self.COMBINED_PHONEMES_TEXTGRID_FILE,
                format="long_textgrid",
                includeBlankSpaces=True,
            )
        except UnicodeEncodeError:
            # If there's an encoding error, try to encode the text manually
            print("Warning: Unicode encoding issue detected. Attempting to fix...")
            for tier in tg.tierList:
                for interval in tier.intervalList:
                    if hasattr(interval, "label") and interval.label:
                        # Ensure the label is properly encoded as UTF-8
                        interval.label = interval.label.encode(
                            "utf-8", errors="replace"
                        ).decode("utf-8")
            tg.save(
                self.COMBINED_PHONEMES_TEXTGRID_FILE,
                format="long_textgrid",
                includeBlankSpaces=True,
            )

    def extract_formants(self):
        if not self.sound:
            raise ValueError("Sound object is not initialized.")
        if not self.formant_data:
            raise ValueError("FormantData object is not initialized.")
        self.formant_data.extract_formant_data_from_sound(self.sound)
        return (
            self.formant_data.get_formant_times(),
            self.formant_data.get_formant_values(),
        )

    def run_MFA(self):
        if (
            not self.AUDIO_FILE
            or not os.path.exists(self.AUDIO_FILE)
            or not self.STIMULUS_FILE
            or not os.path.exists(self.STIMULUS_FILE)
        ):
            print("Error: No valid audio or stimulus file provided.")
            return
        if self.COMBINED_PHONEMES_TEXTGRID_FILE and os.path.exists(
            self.COMBINED_PHONEMES_TEXTGRID_FILE
        ):
            print("Warning: Phoneme TextGrid already exists. Skipping MFA alignment.")
            self.add_phonemes_from_textgrid()
            self.detected_sentences_path_label.setText(
                f"Sentences Path: {self.SENTENCES_DIR}"
            )
            return
        valid_sentence_count = 0
        textgrid_offsets = []
        for sentence_timestamp in self.sentence_timestamps:
            sentence, start_time, end_time = (
                sentence_timestamp.sentence,
                sentence_timestamp.start,
                sentence_timestamp.end,
            )
            if (
                not sentence
                or start_time is None
                or end_time is None
                or start_time >= end_time
                or start_time < 0
                or end_time > self.sound.get_total_duration()
            ):
                continue
            valid_sentence_count += 1
            os.makedirs(self.SENTENCES_DIR, exist_ok=True)
            sound_section = self.sound.extract_part(start_time, end_time)
            sound_section.save(
                f"{self.SENTENCES_DIR}/sentence_{valid_sentence_count}.wav",
                format="WAV",
            )
            with open(
                f"{self.SENTENCES_DIR}/sentence_{valid_sentence_count}.lab",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(sentence)
            textgrid_offsets.append(
                (
                    f"{self.SENTENCES_PHONEMES_DIR}/sentence_{valid_sentence_count}.TextGrid",
                    sentence,
                    start_time,
                    end_time,
                )
            )
        print(
            f"Saved {valid_sentence_count} valid sentences to {self.SENTENCES_DIR} directory."
        )
        if valid_sentence_count > 0:
            if not self.ASR:
                self.ASR = self.whisper_ASR
                print("No ASR method selected. Defaulting to Whisper AI.")
            self.ASR.set_asr_target(
                self.AUDIO_FILE,
                stimulus_path=self.STIMULUS_FILE,
                textgrid_path=self.SENTENCES_TEXTGRID_FILE,
            )
            try:
                self.ASR.save_sentences_as_textgrid(self.sentence_timestamps)
            except Exception as e:
                self.show_error_dialog(
                    "Sentence TextGrid Save Error",
                    e,
                    "MFA phoneme alignment and loading phonemes from TextGrid.",
                )
                return
            self.detected_sentences_path_label.setText(
                f"Sentences Path: {self.SENTENCES_DIR}"
            )
            # Uses the bundled MFA environment in the standalone build,
            # otherwise `conda run -n MFA_ENVIRONMENT_NAME` as before.
            run_mfa_alignment(
                script_path=self.MFA_ALIGNER_SCRIPT_PATH,
                input_dir=f"{self.SENTENCES_DIR}",  # TODO: argument not working, output_dir needs to be added
                conda_exe=self.CONDA_EXE_PATH,
                env_name=self.MFA_ENVIRONMENT_NAME,
            )
            print("MFA alignment completed.")
            try:
                self.combine_phoneme_textgrids(textgrid_offsets=textgrid_offsets)
                print("MFA alignment saved as TextGrid.")
            except Exception as e:
                self.show_error_dialog(
                    "Phoneme TextGrid Save Error",
                    e,
                    "Saving combined MFA phoneme TextGrid and loading phonemes into the UI.",
                )
                return
        self.add_phonemes_from_textgrid()

    def update_progress(self):
        if self.playing:
            # current_time = pygame.mixer.music.get_pos() / 1000  # Convert ms to seconds
            current_time = self.media_player.get_time() / 1000  # Convert ms to seconds
            duration = self.sound.get_total_duration()

            if current_time >= duration:
                self.pause_audio()
                return

            if current_time >= self.waveform_plot.getViewBox().viewRange()[0][1]:
                self.pause_audio()

            # Move the red progress line
            self.progress_line.setValue(
                self.media_player.get_time() / 1000
            )  # Convert ms to seconds

    def jump_to_progress_line(self):
        if self.progress_line:
            self.pause_audio()
            new_time = self.progress_line.value() * 1000  # Convert seconds to ms
            new_time = int(new_time)
            # pygame.mixer.music.set_pos(new_time)
            # self.media_player.set_time(new_time)
            self.play_audio(set_time=new_time)

    def sync_x_axes(self, view, range):
        self.waveform_plot.getViewBox().setXRange(*range[0], padding=0)
        self.sentence_label_plot.getViewBox().setXRange(*range[0], padding=0)
        self.spectrogram_plot.getViewBox().setXRange(*range[0], padding=0)
        self.phoneme_label_plot.getViewBox().setXRange(*range[0], padding=0)

    def sync_sentence_regions(self, selected_region_item):
        selected_sentence = None
        selected_region_bounds = selected_region_item.getRegion()
        for sentence, sentence_region in self.sentence_regions.items():
            region_items, text_item = (
                sentence_region.region_items,
                sentence_region.text_item,
            )
            if selected_region_item in region_items:
                selected_sentence = sentence
                for region in region_items:
                    region.setRegion(selected_region_bounds)
                text_item.setPos(
                    (selected_region_bounds[0] + selected_region_bounds[1]) / 2, 0.5
                )
                break
        for sentence_timestamp in self.sentence_timestamps:
            if sentence_timestamp.sentence == selected_sentence:
                sentence_timestamp.start = selected_region_bounds[0]
                sentence_timestamp.end = selected_region_bounds[1]
                break
        self.sentence_timestamps.sort(key=lambda x: x.start)
        sentence_text = ensure_utf8_display(sentence)
        self.sentence_descriptor_items[sentence][1].setText(
            f"{selected_region_bounds[0]:.3f} - {selected_region_bounds[1]:.3f}: {sentence_text}"
        )

    def sync_hover_sentence_regions(self, hovered_region_item):
        for sentence, sentence_region in self.sentence_regions.items():
            region_items, text_item = (
                sentence_region.region_items,
                sentence_region.text_item,
            )
            if hovered_region_item in region_items:
                for region in region_items:
                    if region != hovered_region_item:
                        region.blockSignals(True)
                        region.setMouseHover(hover=hovered_region_item.mouseHovering)
                        region.blockSignals(False)

    def load_phoneme_ground_truth(self):
        """Load phoneme ground truth from CSV file."""
        if not os.path.exists(self.PHONEME_GROUND_TRUTH_FILEPATH):
            print(
                f"Warning: Phoneme ground truth file not found at {self.PHONEME_GROUND_TRUTH_FILEPATH}"
            )
            return
        try:
            with open(
                self.PHONEME_GROUND_TRUTH_FILEPATH, "r", encoding="utf-8"
            ) as csvfile:
                reader = csv.DictReader(csvfile)
                self.phoneme_ground_truth = []
                for row in reader:
                    self.phoneme_ground_truth.append(dict(row))
            print(
                f"Loaded {len(self.phoneme_ground_truth)} phoneme ground truth entries."
            )
        except Exception as e:
            print(f"Error loading phoneme ground truth: {e}")

    def get_phoneme_type(self, phoneme_timestamp: PhonemeTimestamp) -> str:
        """Determine the type of phoneme based on landmark ground truth."""
        if not os.path.exists(self.PHONEME_GROUND_TRUTH_FILEPATH):
            print(
                f"Warning: Phoneme ground truth file not found at {self.PHONEME_GROUND_TRUTH_FILEPATH}"
            )
            return ""
        if not phoneme_timestamp or not phoneme_timestamp.phoneme:
            return ""
        if not self.phoneme_ground_truth:
            self.load_phoneme_ground_truth()
        for entry in self.phoneme_ground_truth:
            if entry["Phoneme"] == phoneme_timestamp.phoneme:
                return entry.get("Type", "")
        return ""

    def get_phoneme_plot_widget(self, phoneme_type: str) -> pg.PlotWidget:
        """Get the appropriate plot widget for the phoneme type."""
        if phoneme_type == "monophthong":
            return self.monophthong_scatter_plot
        elif phoneme_type == "centring_diphthong":
            return self.centring_diphthong_scatter_plot
        elif phoneme_type == "rising_diphthong":
            return self.rising_diphthong_scatter_plot
        else:
            return None

    def sync_phoneme_regions(self, selected_region_item):
        selected_phoneme_timestamp = None
        selected_phoneme_hash = None
        selected_region_bounds = selected_region_item.getRegion()
        for phoneme_hash, phoneme_region in self.phoneme_regions.items():
            region_items, text_item = (
                phoneme_region.region_items,
                phoneme_region.text_item,
            )
            if selected_region_item in region_items:
                for phoneme_timestamp in self.phoneme_timestamps:
                    if get_phoneme_hash(phoneme_timestamp) == phoneme_hash:
                        selected_phoneme_timestamp = phoneme_timestamp
                        selected_phoneme_hash = phoneme_hash
                        break
                for region in region_items:
                    region.setRegion(selected_region_bounds)
                text_item.setPos(
                    (selected_region_bounds[0] + selected_region_bounds[1]) / 2, 0.5
                )
                break
        selected_phoneme_timestamp.start = selected_region_bounds[0]
        selected_phoneme_timestamp.end = selected_region_bounds[1]
        if len(selected_phoneme_timestamp.formant_targets) > 0:
            self.set_formant_target_for_phoneme(selected_phoneme_timestamp)
        self.phoneme_timestamps.sort(key=lambda x: x.start)

        # Update UI descriptor text
        if self.formant_data:
            descriptor_text = self.formant_data.formant_targets_to_descriptor_text(
                selected_phoneme_timestamp
            )
            if selected_phoneme_hash in self.phoneme_descriptor_items:
                self.phoneme_descriptor_items[selected_phoneme_hash][1].setText(
                    descriptor_text
                )

        if selected_phoneme_timestamp and selected_phoneme_timestamp.formant_targets:
            phoneme_type = self.get_phoneme_type(selected_phoneme_timestamp)
            if phoneme_type:
                self.update_formant_scatter_plot(
                    self.get_phoneme_plot_widget(phoneme_type)
                )

    def sync_hover_phoneme_regions(self, hovered_region_item):
        for phoneme_hash, phoneme_region in self.phoneme_regions.items():
            region_items, text_item = (
                phoneme_region.region_items,
                phoneme_region.text_item,
            )
            if hovered_region_item in region_items:
                for region in region_items:
                    region.blockSignals(True)
                    region.setMouseHover(hover=hovered_region_item.mouseHovering)
                    region.blockSignals(False)

    def add_sentence_timestamp(
        self, sentence, *, start_time, end_time
    ) -> SentenceTimestamp | None:
        if (
            not sentence
            or start_time is None
            or end_time is None
            or start_time >= end_time
            or start_time < 0
            or end_time > self.sound.get_total_duration()
        ):
            return None
        self.remove_sentence_timestamp(sentence)
        sentence_timestamp = SentenceTimestamp(sentence, start_time, end_time)
        self.sentence_timestamps.append(sentence_timestamp)
        self.sentence_timestamps.sort(key=lambda x: x.start)
        print("Added sentence timestamp:", sentence)
        return sentence_timestamp

    def find_sentence_timestamp(self, sentence: str) -> SentenceTimestamp | None:
        # First try exact match
        for sentence_timestamp in self.sentence_timestamps:
            if sentence_timestamp.sentence == sentence:
                return sentence_timestamp

        # If exact match fails, try fuzzy matching
        if sentence and self.sentence_timestamps:
            best_match = difflib.get_close_matches(
                sentence,
                [s.sentence for s in self.sentence_timestamps],
                n=1,
                cutoff=0.7,
            )
            if best_match:
                matched_sentence = best_match[0]
                for sentence_timestamp in self.sentence_timestamps:
                    if sentence_timestamp.sentence == matched_sentence:
                        print(
                            f"Found fuzzy match: '{matched_sentence}' for query: '{sentence}'"
                        )
                        return sentence_timestamp

        return None

    def find_word_in_textgrid(
        self, word_text: str, phoneme_start: float, phoneme_end: float
    ) -> WordTimestamp | None:
        """
        Search for a word in the combined phonemes TextGrid file.

        Args:
            word_text: The word to search for
            phoneme_start: Start time of the phoneme (for proximity search)
            phoneme_end: End time of the phoneme (for proximity search)

        Returns:
            WordTimestamp if found, None otherwise
        """
        if not self.COMBINED_PHONEMES_TEXTGRID_FILE or not os.path.exists(
            self.COMBINED_PHONEMES_TEXTGRID_FILE
        ):
            return None

        try:
            tg = textgrid.openTextgrid(
                self.COMBINED_PHONEMES_TEXTGRID_FILE, includeEmptyIntervals=False
            )

            # Find the word tier
            word_tier = None
            for tier in tg:
                if isinstance(tier, textgrid.IntervalTier) and tier.name == "words":
                    word_tier = tier
                    break

            if not word_tier:
                return None

            # Search for the word that contains the phoneme time range
            for word_interval in word_tier:
                # Clean up the word label
                word_label = word_interval.label
                word_label = word_label.replace("'", "'")
                word_label = ensure_utf8_display(word_label)

                # Check if this is the word we're looking for
                # First check if the word text matches (case-insensitive)
                if word_label.lower() == word_text.lower():
                    # Check if phoneme times are within word bounds (with tolerance)
                    if (
                        phoneme_start >= word_interval.start - self.timestamp_tolerance
                        and phoneme_end <= word_interval.end + self.timestamp_tolerance
                    ):
                        return WordTimestamp(
                            word=word_label,
                            start=word_interval.start,
                            end=word_interval.end,
                        )

            # If exact match with time constraint fails, try fuzzy matching
            word_labels = [
                (ensure_utf8_display(w.label.replace("'", "'")), w) for w in word_tier
            ]
            best_match = difflib.get_close_matches(
                word_text.lower(),
                [label.lower() for label, _ in word_labels],
                n=1,
                cutoff=0.7,
            )

            if best_match:
                matched_word_lower = best_match[0]
                for label, word_interval in word_labels:
                    if label.lower() == matched_word_lower:
                        # Check if phoneme times are within word bounds (with tolerance)
                        if (
                            phoneme_start
                            >= word_interval.start - self.timestamp_tolerance
                            and phoneme_end
                            <= word_interval.end + self.timestamp_tolerance
                        ):
                            print(
                                f"Found fuzzy match: '{label}' for query: '{word_text}'"
                            )
                            return WordTimestamp(
                                word=label,
                                start=word_interval.start,
                                end=word_interval.end,
                            )

        except Exception as e:
            print(f"Error searching for word in TextGrid: {e}")

        return None

    def remove_sentence_timestamp(self, sentence):
        for sentence_timestamp in self.sentence_timestamps:
            if sentence_timestamp.sentence == sentence:
                self.sentence_timestamps.remove(sentence_timestamp)
                self.sentence_timestamps.sort(key=lambda x: x.start)
                print("Removed sentence timestamp:", sentence)
                return

    def add_sentence_region(self, sentence, *, start_time, end_time):
        if (
            not sentence
            or start_time is None
            or end_time is None
            or start_time >= end_time
            or start_time < 0
            or end_time > self.sound.get_total_duration()
        ):
            return
        waveform_sentence_region = CustomLinearRegionItem(
            values=[start_time, end_time],
            brush=(0, 255, 0, 50),
            hoverBrush=(0, 255, 0, 100),
        )
        waveform_sentence_region.sigRegionChanged.connect(self.sync_sentence_regions)
        waveform_sentence_region.sigHoverEvent.connect(self.sync_hover_sentence_regions)
        self.waveform_plot.addItem(waveform_sentence_region)

        sentence_label_region = CustomLinearRegionItem(
            values=[start_time, end_time],
            brush=(0, 255, 0, 50),
            hoverBrush=(0, 255, 0, 100),
        )
        sentence_label_region.sigRegionChanged.connect(self.sync_sentence_regions)
        sentence_label_region.sigHoverEvent.connect(self.sync_hover_sentence_regions)
        self.sentence_label_plot.addItem(sentence_label_region)

        sentence_text = ensure_utf8_display(sentence)
        text_item = pg.TextItem(
            sentence_text, anchor=(0.5, 0.5), color=(0, 0, 0), fill=(255, 255, 255, 100)
        )
        text_item.setPos((start_time + end_time) / 2, 0.5)
        self.sentence_label_plot.addItem(text_item)

        spectrogram_sentence_region = CustomLinearRegionItem(
            values=[start_time, end_time],
            brush=(10, 0, 0, 50),
            hoverBrush=(10, 0, 0, 100),
        )
        spectrogram_sentence_region.sigRegionChanged.connect(self.sync_sentence_regions)
        spectrogram_sentence_region.sigHoverEvent.connect(
            self.sync_hover_sentence_regions
        )
        self.spectrogram_plot.addItem(spectrogram_sentence_region)

        if sentence in self.sentence_regions:
            self.remove_sentence_region(sentence)

        self.sentence_regions[sentence] = SentenceRegion(
            region_items=[
                waveform_sentence_region,
                sentence_label_region,
                spectrogram_sentence_region,
            ],
            text_item=text_item,
        )
        print("Added sentence regions:", sentence)
        # else:
        #     if type(self.sentence_regions[sentence]) == SentenceRegion:
        #         for region in self.sentence_regions[sentence].region_items:
        #             self.waveform_plot.removeItem(region)
        #             self.sentence_label_plot.removeItem(region)
        #             self.spectrogram_plot.removeItem(region)
        #         self.sentence_regions[sentence].region_items = []
        #         self.sentence_label_plot.removeItem(self.sentence_regions[sentence].text_item)
        #     self.sentence_regions[sentence].region_items=[waveform_sentence_region, sentence_label_region, spectrogram_sentence_region]
        #     self.sentence_regions[sentence].text_item = text_item

    def remove_sentence_region(self, sentence):
        if sentence in self.sentence_regions:
            if type(self.sentence_regions[sentence]) == SentenceRegion:
                region_items = self.sentence_regions[sentence].region_items
                self.waveform_plot.removeItem(region_items[0])
                self.sentence_label_plot.removeItem(region_items[1])
                self.spectrogram_plot.removeItem(region_items[2])
                self.sentence_label_plot.removeItem(
                    self.sentence_regions[sentence].text_item
                )
            del self.sentence_regions[sentence]
            print("Removed sentence regions:", sentence)

    def remove_sentence(self, sentence):
        self.remove_sentence_from_display(sentence)
        self.remove_sentence_timestamp(sentence)

    def remove_sentence_from_display(self, sentence):
        self.remove_sentence_region(sentence)
        self.remove_sentence_from_ui_list(sentence)

    def clear_sentences(self):
        for sentence_timestamp in self.sentence_timestamps:
            sentence = sentence_timestamp.sentence
            self.remove_sentence_from_display(sentence)
        self.sentence_timestamps = []
        self.sentence_descriptor_items = {}
        self.sentence_regions = {}
        print("Cleared sentences.")

    def add_phoneme(
        self,
        *,
        phoneme: str,
        sentence_timestamp: SentenceTimestamp | None = None,
        word_timestamp: WordTimestamp | None = None,
        start_time: float,
        end_time: float,
    ):
        if (
            not phoneme
            or start_time is None
            or end_time is None
            or start_time >= end_time
            or start_time < 0
            or end_time > self.sound.get_total_duration()
        ):
            return
        phoneme_timestamp = self.add_phoneme_timestamp(
            phoneme=phoneme,
            sentence_timestamp=sentence_timestamp,
            word_timestamp=word_timestamp,
            start_time=start_time,
            end_time=end_time,
        )
        if phoneme_timestamp:
            self.add_phoneme_region(phoneme_timestamp)

    def add_phoneme_timestamp(
        self,
        *,
        phoneme: str,
        sentence_timestamp: SentenceTimestamp | None = None,
        word_timestamp: WordTimestamp | None = None,
        start_time: float,
        end_time: float,
    ) -> PhonemeTimestamp | None:
        if (
            not phoneme
            or start_time is None
            or end_time is None
            or start_time >= end_time
            or start_time < 0
            or end_time > self.sound.get_total_duration()
        ):
            return None
        phoneme_timestamp = PhonemeTimestamp(
            phoneme,
            start_time,
            end_time,
            parent_sentence_timestamp=sentence_timestamp,
            parent_word_timestamp=word_timestamp,
        )
        self.phoneme_timestamps.append(phoneme_timestamp)
        self.phoneme_timestamps.sort(key=lambda x: x.start)
        print("Added phoneme timestamp:", phoneme)
        return phoneme_timestamp

    def add_phoneme_to_ui_list(self, phoneme_timestamp: PhonemeTimestamp):
        if not phoneme_timestamp:
            return
        phoneme = phoneme_timestamp.phoneme
        start_time = phoneme_timestamp.start
        end_time = phoneme_timestamp.end
        if (
            not phoneme
            or start_time is None
            or end_time is None
            or start_time >= end_time
            or start_time < 0
            or end_time > self.sound.get_total_duration()
        ):
            return

        # Create row widget
        phoneme_hash = get_phoneme_hash(phoneme_timestamp)
        if phoneme_hash in self.phoneme_descriptor_items:
            # self.remove_phoneme_from_ui_list(phoneme_timestamp)
            descriptor_text = self.formant_data.formant_targets_to_descriptor_text(
                phoneme_timestamp
            )
            self.phoneme_descriptor_items[phoneme_hash][1].setText(descriptor_text)
            print("Updated UI list for phoneme:", phoneme_timestamp)
            return

        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_widget.setLayout(row_layout)

        descriptor_text = self.formant_data.formant_targets_to_descriptor_text(
            phoneme_timestamp
        )
        label = QLabel(descriptor_text)
        delete_btn = QPushButton("Delete")
        delete_btn.setMaximumWidth(100)

        def remove_phoneme():
            self.remove_phoneme(phoneme_timestamp)

        delete_btn.clicked.connect(remove_phoneme)

        row_layout.addWidget(label)
        row_layout.addWidget(delete_btn)
        self.phoneme_descriptor_items[phoneme_hash] = (row_widget, label)
        self.phoneme_descriptor_layout.addWidget(row_widget)
        print("Added phoneme to UI list:", phoneme_timestamp)

    def remove_phoneme_from_ui_list(self, phoneme_timestamp: PhonemeTimestamp):
        phoneme_hash = get_phoneme_hash(phoneme_timestamp)
        if phoneme_hash in self.phoneme_descriptor_items:
            self.phoneme_descriptor_layout.removeWidget(
                self.phoneme_descriptor_items[phoneme_hash][0]
            )
            self.phoneme_descriptor_items[phoneme_hash][0].setParent(None)
            removed_phoneme_descriptor = self.phoneme_descriptor_items.pop(phoneme_hash)
            removed_phoneme_descriptor[0].deleteLater()
            removed_phoneme_descriptor[1].deleteLater()
            del removed_phoneme_descriptor
            print("Removed phoneme from UI list:", phoneme_timestamp)

    def set_formant_target_for_phoneme(self, phoneme_timestamp: PhonemeTimestamp):
        if not self.formant_data:
            raise ValueError("Formant data is not initialized.")
        if not self.formant_data.landmark_info_filepath:
            raise ValueError("Landmark info filepath is not set in formant data.")

        # First, remove old target lines to avoid conflicts with deleted objects
        self.remove_phoneme_target_lines(phoneme_timestamp)

        phoneme_formant_times, phoneme_formant_values = (
            self.formant_data.set_formant_target_for_phoneme(
                phoneme_timestamp,
                cleaning_algo=self.formant_cleaning_algo,
                use_ground_truth=self.USE_GROUND_TRUTH,
                dynamic_frequency_ceiling=False,
            )
        )
        if len(phoneme_timestamp.formant_targets) > 0:
            if (
                phoneme_formant_times.shape[0]
                and phoneme_formant_values.shape[0]
                and phoneme_formant_values.shape[1]
                and phoneme_formant_values.shape[0] == phoneme_formant_times.shape[0]
            ):
                # Define distinct colors for each formant that are easily visible on spectrogram
                # Avoiding red which is already used in the plot
                formant_colors = [
                    (255, 255, 255),  # F1: White
                    (255, 0, 255),  # F2: Magenta
                    (0, 255, 255),  # F3: Cyan
                    (255, 255, 0),  # F4: Yellow
                    (128, 128, 128),  # F5: Gray (less important)
                ]

                for i, formant_name in enumerate(["F1", "F2", "F3", "F4", "F5"]):
                    if i >= phoneme_formant_values.shape[1]:
                        break
                    self.spectrogram_plot.plot(
                        phoneme_formant_times,
                        phoneme_formant_values[:, i],
                        pen=None,  # No line
                        symbol="o",
                        symbolBrush=pg.mkBrush(formant_colors[i]),
                        symbolSize=6,
                        name=formant_name,
                    )
            self.add_phoneme_to_ui_list(phoneme_timestamp)
            self.add_phoneme_target_lines(phoneme_timestamp)

    def set_formant_targets(self):
        if not self.formant_data:
            raise ValueError("Formant data is not initialized.")
        self.formant_data.set_landmark_info_filepath(self.LANDMARK_INFO_FILE)
        for phoneme_timestamp in self.phoneme_timestamps:
            self.set_formant_target_for_phoneme(phoneme_timestamp)
        self.setup_formant_scatter_plot()

    def setup_formant_scatter_plot(self):

        # Populate axis combos and phoneme list without duplicating entries
        n_formants = self.formant_data.n_formants if self.formant_data else 5
        prev_x = self.x_axis_combo.currentIndex()
        prev_y = self.y_axis_combo.currentIndex()
        self.x_axis_combo.blockSignals(True)
        self.y_axis_combo.blockSignals(True)
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        for i in range(n_formants):
            self.x_axis_combo.addItem(f"Formant {i+1}", i)
            self.y_axis_combo.addItem(f"Formant {i+1}", i)
        self.x_axis_combo.setCurrentIndex(
            prev_x if 0 <= prev_x < n_formants else (1 if n_formants > 1 else 0)
        )
        self.y_axis_combo.setCurrentIndex(
            prev_y if 0 <= prev_y < n_formants else 0
        )
        self.x_axis_combo.blockSignals(False)
        self.y_axis_combo.blockSignals(False)

        self.update_phoneme_picker_list_widget()

        if not getattr(self, "_scatter_signals_connected", False):
            self.x_axis_combo.currentIndexChanged.connect(
                self.update_all_formant_scatter_plots
            )
            self.y_axis_combo.currentIndexChanged.connect(
                self.update_all_formant_scatter_plots
            )
            self.phoneme_picker_list_widget.itemSelectionChanged.connect(
                self.update_all_formant_scatter_plots
            )
            self._scatter_signals_connected = True

        self.update_all_formant_scatter_plots()

    def update_all_formant_scatter_plots(self):
        self.update_formant_scatter_plot(self.monophthong_scatter_plot)
        self.update_formant_scatter_plot(self.centring_diphthong_scatter_plot)
        self.update_formant_scatter_plot(self.rising_diphthong_scatter_plot)

    def update_phoneme_picker_list_widget(self):
        self.phoneme_picker_list_widget.clear()
        for pt in self.phoneme_timestamps:
            if pt.formant_targets:
                sentence_text = ensure_utf8_display(
                    pt.parent_sentence_timestamp.sentence
                    if pt.parent_sentence_timestamp
                    else ""
                )
                word_text = ensure_utf8_display(
                    pt.parent_word_timestamp.word if pt.parent_word_timestamp else ""
                )
                phoneme_text = ensure_utf8_display(pt.phoneme)
                phoneme_type = self.get_phoneme_type(pt)
                if phoneme_type:
                    # Create label with phoneme type
                    label = f"{sentence_text} | {word_text} | {phoneme_text} | {phoneme_type}"
                else:
                    label = (
                        f"{sentence_text} | {word_text} | {phoneme_text} | Unknown Type"
                    )
                item = QListWidgetItem(label)
                item.setData(256, pt)  # Qt.ItemDataRole.UserRole
                self.phoneme_picker_list_widget.addItem(item)
                item.setSelected(True)  # Default: all selected

    def plot_formant_ground_truth_quadrilateral(
        self, gender, plot_widget, plot_type="monophthong"
    ):
        """
        Plot a quadrilateral connecting all valid ground truth monophthong points for the given gender from landmark_identification_ground_truth.csv.
        The quadrilateral is shown only if x/y axes are Formant 1/Formant 2 (in any order).
        Corners are labelled by the Phoneme value in red. Draw red ellipses for each monophthong using SDs as axis lengths.
        """

        if not os.path.exists(self.PHONEME_GROUND_TRUTH_FILEPATH):
            print(f"Ground truth CSV not found: {self.PHONEME_GROUND_TRUTH_FILEPATH}")
            return
        # plot_widget = self.get_phoneme_plot_widget(plot_type)
        if not plot_widget:
            raise ValueError(f"Plot widget not given")
        df = pd.read_csv(self.PHONEME_GROUND_TRUTH_FILEPATH)
        # Filter by gender, type monophthong, and valid formant values
        df = df[
            (df["Gender"] == gender)
            & (df["Type"] == plot_type)
            & df["T1_F1_Mean"].notnull()
            & df["T1_F2_Mean"].notnull()
        ]
        if df.empty:
            return
        # Get axis mapping
        x_label = self.x_axis_combo.currentText()
        y_label = self.y_axis_combo.currentText()
        if ("Formant 1" in x_label and "Formant 2" in y_label) or (
            "Formant 2" in x_label and "Formant 1" in y_label
        ):
            x_col_T1 = "T1_F1_Mean" if "Formant 1" in x_label else "T1_F2_Mean"
            y_col_T1 = "T1_F2_Mean" if "Formant 2" in y_label else "T1_F1_Mean"
            x_col_SD = "T1_F1_SD" if "Formant 1" in x_label else "T1_F2_SD"
            y_col_SD = "T1_F2_SD" if "Formant 2" in y_label else "T1_F1_SD"
            # Get points
            points = (
                df[[x_col_T1, y_col_T1, "Phoneme", x_col_SD, y_col_SD]].dropna().values
            )
            if len(points) < 3:
                return
            # Convex hull for quadrilateral
            coords = points[:, :2].astype(float)

            # Get all phoneme labels
            all_labels = points[:, 2]
            all_coords = coords.copy()

            try:
                hull = ConvexHull(coords)
                hull_indices = hull.vertices

                # Check if 'e' is in the hull
                e_index = None
                for i, label in enumerate(all_labels):
                    if label == "e":
                        e_index = i
                        break

                # If 'e' exists but is not in the hull, add it in second position
                if e_index is not None and e_index not in hull_indices:
                    # Insert at position 1 (second element)
                    hull_indices = np.insert(hull_indices, 1, e_index)
            except Exception:
                hull_indices = np.arange(len(coords))

            hull_coords = coords[hull_indices]
            hull_labels = points[hull_indices, 2]

            # Plot polygon (red outline)
            poly = pg.PlotCurveItem(
                hull_coords[:, 0].tolist() + [hull_coords[0, 0]],
                hull_coords[:, 1].tolist() + [hull_coords[0, 1]],
                pen=pg.mkPen("r", width=2),
            )
            poly.setZValue(-10)
            plot_widget.addItem(poly)

            # Add all monophthong labels (red)
            for i, (x, y, label) in enumerate(
                zip(all_coords[:, 0], all_coords[:, 1], all_labels)
            ):
                text = pg.TextItem(str(label), anchor=(0.5, 1.2), color=(255, 0, 0))
                text.setPos(float(x), float(y))
                text.setZValue(-9)
                plot_widget.addItem(text)
            # # Draw ellipses for each monophthong
            # for pt in points:
            #     cx, cy, label, sx, sy = pt
            #     # Draw ellipse using SDs as axis lengths
            #     theta = np.linspace(0, 2*np.pi, 100)
            #     ex = cx + sx * np.cos(theta)
            #     ey = cy + sy * np.sin(theta)
            #     ellipse = pg.PlotDataItem(ex, ey, pen=pg.mkPen('r', width=1))
            #     ellipse.setZValue(-8)
            #     plot_widget.addItem(ellipse)

    # Lists of maximally distinct non-red colors and enclosed shapes (red is reserved for reference polygon)
    SCATTER_COLORS: list[tuple[int, int, int]] = [
        (66, 165, 245),   # 0: Bright Blue (pairs with 'o')
        (255, 152, 36),   # 1: Vivid Orange (pairs with 's')
        (76, 195, 105),   # 2: Emerald Green (pairs with 'd')
        (171, 110, 225),  # 3: Amethyst Purple (pairs with 'p')
        (255, 210, 30),   # 4: Gold Yellow (pairs with 'h')
        (38, 198, 218),   # 5: Cyan Turquoise (pairs with 't')
        (175, 120, 75),   # 6: Warm Bronze / Brown (pairs with 't1')
        (150, 205, 55),   # 7: Lime Chartreuse (pairs with 't2')
        (110, 125, 215),  # 8: Indigo Periwinkle (pairs with 't3')
        (190, 195, 45),   # 9: Citron Olive (pairs with 'o' - opposite hue to 0 Blue)
        (38, 166, 154),   # 10: Deep Sea Teal (pairs with 's' - opposite hue to 1 Orange)
        (218, 135, 215),  # 11: Orchid Lavender (pairs with 'd' - opposite hue to 2 Green)
    ]
    SCATTER_SHAPES: list[str] = [
        "o",   # Circle
        "s",   # Square
        "d",   # Diamond
        "p",   # Pentagon
        "h",   # Hexagon
        "t",   # Triangle down
        "t1",  # Triangle up
        "t2",  # Triangle right
        "t3",  # Triangle left
    ]

    def get_phoneme_style(
        self, phoneme: str, phoneme_type: str, sorted_vowels: list[str]
    ) -> tuple[tuple[int, int, int], str]:
        """
        Deterministically select a (color, symbol) pair from SCATTER_COLORS and SCATTER_SHAPES
        based on the vowel's index in the current dataset's sorted vowel list (no hardcoded vowel mapping).
        """
        if phoneme in sorted_vowels:
            idx = sorted_vowels.index(phoneme)
        else:
            idx = 0
        color = self.SCATTER_COLORS[idx % len(self.SCATTER_COLORS)]
        if phoneme_type == "monophthong":
            shape = self.SCATTER_SHAPES[idx % len(self.SCATTER_SHAPES)]
        else:
            shape = "o"
        return color, shape

    def update_scatter_label_visibility(self, plot_widget):
        """
        Dynamically adjust visibility of point labels based on current zoom level (ViewBox range):
        - When zoomed out, close points of the same vowel collapse to a single label per cluster,
          while distant outliers still get their own labels.
        - As we zoom in and points move further apart on screen, more labels appear.
        - Same-vowel labels have a higher closeness threshold (dropped first when too close),
          while different-vowel labels have a lower closeness threshold.
        """
        if getattr(plot_widget, "_exporting_lock", False):
            return
        label_entries = getattr(plot_widget, "point_label_items", None)
        if not label_entries:
            return

        vb = plot_widget.getViewBox()
        x_range, y_range = vb.viewRange()
        x_min, x_max = min(x_range), max(x_range)
        y_min, y_max = min(y_range), max(y_range)
        span_x = max(x_max - x_min, 1e-6)
        span_y = max(y_max - y_min, 1e-6)

        # Normalized screen-space closeness thresholds (fraction of current view span)
        # Same-vowel labels are dropped first when closer than same_vowel_thresh;
        # different-vowel labels use a much smaller closeness threshold.
        same_vowel_thresh = 0.16
        diff_vowel_thresh = 0.045

        # Separate visible vs off-screen points
        visible_entries = []
        for entry in label_entries:
            x, y = entry["x"], entry["y"]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                entry["nx"] = (x - x_min) / span_x
                entry["ny"] = (y - y_min) / span_y
                visible_entries.append(entry)
            else:
                entry["text_item"].setVisible(False)

        if not visible_entries:
            return

        # Group visible entries by vowel
        by_vowel: dict[str, list[dict]] = {}
        for entry in visible_entries:
            by_vowel.setdefault(entry["vowel"], []).append(entry)

        accepted: list[dict] = []
        accepted_ids: set[int] = set()

        def is_far_enough(cand: dict) -> bool:
            cnx, cny = cand["nx"], cand["ny"]
            cvowel = cand["vowel"]
            for acc in accepted:
                dist = float(np.hypot(cnx - acc["nx"], cny - acc["ny"]))
                thresh = (
                    same_vowel_thresh
                    if acc["vowel"] == cvowel
                    else diff_vowel_thresh
                )
                if dist < thresh:
                    return False
            return True

        # Pass 1: Give each vowel one representative label first (closest to its cluster median)
        for vowel, group in by_vowel.items():
            med_nx = float(np.median([e["nx"] for e in group]))
            med_ny = float(np.median([e["ny"] for e in group]))
            sorted_group = sorted(
                group,
                key=lambda e: float(np.hypot(e["nx"] - med_nx, e["ny"] - med_ny)),
            )
            for cand in sorted_group:
                if is_far_enough(cand):
                    accepted.append(cand)
                    accepted_ids.add(id(cand))
                    break

        # Pass 2: Consider remaining points (outliers first: largest distance from their vowel's accepted labels)
        remaining = [e for e in visible_entries if id(e) not in accepted_ids]

        def min_same_vowel_dist(cand: dict) -> float:
            same_acc = [a for a in accepted if a["vowel"] == cand["vowel"]]
            if not same_acc:
                return float("inf")
            return min(
                float(np.hypot(cand["nx"] - a["nx"], cand["ny"] - a["ny"]))
                for a in same_acc
            )

        remaining.sort(key=min_same_vowel_dist, reverse=True)
        for cand in remaining:
            if is_far_enough(cand):
                accepted.append(cand)
                accepted_ids.add(id(cand))

        for entry in visible_entries:
            entry["text_item"].setVisible(id(entry) in accepted_ids)

    def update_formant_scatter_plot(self, plot_widget):
        """
        Redraw all points for selected phonemes and axes, for the given plot type.
        """
        x_idx = self.x_axis_combo.currentData()
        y_idx = self.y_axis_combo.currentData()
        if x_idx is None:
            x_idx = 1
        if y_idx is None:
            y_idx = 0
        selected_pts = [
            self.phoneme_picker_list_widget.item(i).data(256)
            for i in range(self.phoneme_picker_list_widget.count())
            if self.phoneme_picker_list_widget.item(i).isSelected()
        ]
        max_x = 0
        max_y = 0
        plot_widget.clear()

        # Update axis labels to match selected formants and auto-scale units (Hz / kHz)
        x_label = self.x_axis_combo.currentText() or f"Formant {x_idx + 1}"
        y_label = self.y_axis_combo.currentText() or f"Formant {y_idx + 1}"
        plot_widget.getAxis("bottom").enableAutoSIPrefix(True)
        plot_widget.getAxis("left").enableAutoSIPrefix(True)
        plot_widget.setLabel("bottom", x_label, units="Hz")
        plot_widget.setLabel("left", y_label, units="Hz")

        plot_widget.scatter_data = []
        plot_widget.point_label_items = []
        plot_type = None

        # Build deterministic sorted list of vowels for this plot from current phoneme data
        all_plot_pts = [
            pt
            for pt in (self.phoneme_timestamps or selected_pts)
            if pt and pt.formant_targets and self.get_phoneme_plot_widget(self.get_phoneme_type(pt)) == plot_widget
        ]
        sorted_vowels = sorted(
            {
                 (
                    ensure_utf8_display(pt.phoneme),
                    round(float(pt.formant_targets[0].targets[0]), 1)
                    if len(pt.formant_targets[0].targets) > 0 and not np.isnan(pt.formant_targets[0].targets[0])
                    else 0.0,
                 )[0]
                for pt in all_plot_pts
            }
        )

        for pt in selected_pts:
            # Only plot matching type
            current_pt_type = self.get_phoneme_type(pt)
            current_pt_plot_widget = self.get_phoneme_plot_widget(current_pt_type)
            if not current_pt_plot_widget or not current_pt_plot_widget == plot_widget:
                continue
            plot_type = current_pt_type

            phoneme_text = ensure_utf8_display(pt.phoneme)
            color, symbol = self.get_phoneme_style(
                phoneme_text, current_pt_type, sorted_vowels
            )
            border_color = (
                max(0, color[0] - 55),
                max(0, color[1] - 55),
                max(0, color[2] - 55),
            )

            target_coords = []

            for i, ft in enumerate(pt.formant_targets):
                if len(ft.targets) > max(x_idx, y_idx):
                    x = ft.targets[x_idx]
                    y = ft.targets[y_idx]
                    if np.isnan(x) or np.isnan(y):
                        continue
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
                    if current_pt_type != "monophthong":
                        csv_label = f"{phoneme_text} T{i+1}"
                    else:
                        csv_label = phoneme_text
                    x_int = int(x)
                    y_int = int(y)

                    target_coords.append((x, y))

                    # Store data for CSV export
                    encoded_label = ensure_utf8_display(csv_label)
                    data = {"label": encoded_label, "x": x_int, "y": y_int}
                    plot_widget.scatter_data.append(data)

                    # For monophthongs, draw every target point and its vowel label inside the shape.
                    # For diphthongs, only draw circle and vowel label inside the circle (no T1/T2 suffix) at T1 (i == 0);
                    # no circle, vowel label, or T2 label at T2.
                    if current_pt_type == "monophthong" or i == 0:
                        active_symbol = symbol if current_pt_type == "monophthong" else "o"
                        # Triangles have smaller inscribed circle radius than circles/squares/polygons, so make them slightly larger
                        marker_size = 30 if active_symbol.startswith("t") else 25
                        scatter = pg.ScatterPlotItem(
                            [x],
                            [y],
                            size=marker_size,
                            pen=pg.mkPen(border_color, width=1.5),
                            brush=pg.mkBrush(*color, 185),
                            symbol=active_symbol,
                        )
                        scatter.point_data = data
                        scatter.setZValue(5)
                        plot_widget.addItem(scatter)

                        text = pg.TextItem(
                            phoneme_text, anchor=(0.5, 0.5), color=(0, 0, 0)
                        )
                        font = text.textItem.font()
                        font.setPointSizeF(8.5)
                        font.setBold(True)
                        text.setFont(font)
                        text.setPos(x, y)
                        text.setZValue(10)
                        plot_widget.addItem(text)
                        plot_widget.point_label_items.append(
                            {
                                "text_item": text,
                                "vowel": phoneme_text,
                                "x": float(x),
                                "y": float(y),
                            }
                        )

            if current_pt_type != "monophthong" and len(target_coords) >= 2:
                # Draw arrow from T1 to T2
                x1, y1 = target_coords[0]
                x2, y2 = target_coords[1]

                arrow_color = (*color, 130)

                # Draw the line
                line = pg.PlotCurveItem(
                    [x1, x2], [y1, y2], pen=pg.mkPen(arrow_color, width=2)
                )
                plot_widget.addItem(line)

                # Draw arrowhead at T2 pointing from T1
                angle = np.degrees(np.arctan2(y1 - y2, x2 - x1))
                arrow = pg.ArrowItem(
                    pen=arrow_color,
                    brush=arrow_color,
                    angle=angle,
                    headLen=15,
                    tipAngle=30,
                    baseAngle=20,
                )
                arrow.setPos(x2, y2)
                plot_widget.addItem(arrow)
        # Update compact legend at bottom right of the plot
        old_legend = getattr(plot_widget, "compact_legend", None)
        if old_legend is not None:
            try:
                if old_legend.scene() is not None:
                    old_legend.scene().removeItem(old_legend)
                old_legend.setParentItem(None)
            except RuntimeError:
                pass
            plot_widget.compact_legend = None

        plotted_vowels = sorted({entry["vowel"] for entry in plot_widget.point_label_items})
        if plotted_vowels:
            legend_entries = []
            for v in plotted_vowels:
                c, sym = self.get_phoneme_style(
                    v, plot_type or "monophthong", sorted_vowels
                )
                bc = (max(0, c[0] - 55), max(0, c[1] - 55), max(0, c[2] - 55))
                active_sym = sym if (plot_type or "monophthong") == "monophthong" else "o"
                legend_entries.append((v, c, bc, active_sym))
            col_cnt = 3 if len(legend_entries) > 6 else (2 if len(legend_entries) > 3 else 1)
            legend = CompactLegendItem(legend_entries, col_count=col_cnt)
            legend.setParentItem(plot_widget.plotItem)
            legend.anchor(itemPos=(1, 1), parentPos=(1, 1), offset=(-6, -42))
            plot_widget.compact_legend = legend

        plot_widget.setXRange(0, max_x * 1.1 if max_x > 0 else 1)
        plot_widget.setYRange(0, max_y * 1.1 if max_y > 0 else 1)
        plot_widget.setLimits(xMin=0, yMin=0)
        self.plot_formant_ground_truth_quadrilateral(
            self.current_gender, plot_widget=plot_widget
        )
        self.update_scatter_label_visibility(plot_widget)

    def clear_formant_data(self):
        if self.formant_data:
            self.formant_data.clear()
            self.formant_data = None
            print("Cleared formant data.")

    def remove_phoneme_timestamp(self, phoneme_timestamp: PhonemeTimestamp):
        self.phoneme_timestamps.remove(phoneme_timestamp)

    def add_phoneme_region(self, phoneme_timestamp: PhonemeTimestamp):
        if not phoneme_timestamp:
            return
        phoneme = phoneme_timestamp.phoneme
        start_time = phoneme_timestamp.start
        end_time = phoneme_timestamp.end
        if (
            not phoneme
            or start_time is None
            or end_time is None
            or start_time >= end_time
            or start_time < 0
            or end_time > self.sound.get_total_duration()
        ):
            return

        waveform_phoneme_region = CustomLinearRegionItem(
            values=[start_time, end_time],
            brush=(0, 255, 0, 50),
            hoverBrush=(0, 255, 0, 100),
        )
        waveform_phoneme_region.sigRegionChanged.connect(self.sync_phoneme_regions)
        waveform_phoneme_region.sigHoverEvent.connect(self.sync_hover_phoneme_regions)
        self.waveform_plot.addItem(waveform_phoneme_region)

        phoneme_region = CustomLinearRegionItem(
            values=[start_time, end_time],
            brush=(0, 255, 0, 50),
            hoverBrush=(0, 255, 0, 100),
        )
        phoneme_region.sigRegionChanged.connect(self.sync_phoneme_regions)
        phoneme_region.sigHoverEvent.connect(self.sync_hover_phoneme_regions)
        self.phoneme_label_plot.addItem(phoneme_region)

        phoneme_text = ensure_utf8_display(phoneme)
        text_item = pg.TextItem(
            phoneme_text, anchor=(0.5, 0.5), color=(0, 0, 0), fill=(255, 255, 255, 100)
        )
        text_item.setPos((start_time + end_time) / 2, 0.5)
        self.phoneme_label_plot.addItem(text_item)

        spectrogram_phoneme_region = CustomLinearRegionItem(
            values=[start_time, end_time],
            brush=(15, 0, 0, 50),
            hoverBrush=(30, 0, 0, 100),
        )
        spectrogram_phoneme_region.sigRegionChanged.connect(self.sync_phoneme_regions)
        spectrogram_phoneme_region.sigHoverEvent.connect(
            self.sync_hover_phoneme_regions
        )
        self.spectrogram_plot.addItem(spectrogram_phoneme_region)

        phoneme_hash = get_phoneme_hash(phoneme_timestamp)

        self.phoneme_regions[phoneme_hash] = PhonemeRegion(
            region_items=[
                waveform_phoneme_region,
                phoneme_region,
                spectrogram_phoneme_region,
            ],
            text_item=text_item,
        )
        print("Added phoneme regions:", phoneme)

    def remove_phoneme_region(self, phoneme_timestamp: PhonemeTimestamp):
        phoneme_hash = get_phoneme_hash(phoneme_timestamp)
        if phoneme_hash in self.phoneme_regions:
            if type(self.phoneme_regions[phoneme_hash]) == PhonemeRegion:
                region_items = self.phoneme_regions[phoneme_hash].region_items
                self.waveform_plot.removeItem(region_items[0])
                region_items[0].setParent(None)
                self.phoneme_label_plot.removeItem(region_items[1])
                region_items[1].setParent(None)
                self.spectrogram_plot.removeItem(region_items[2])
                region_items[2].setParent(None)
                self.phoneme_label_plot.removeItem(
                    self.phoneme_regions[phoneme_hash].text_item
                )
                self.phoneme_regions[phoneme_hash].text_item.setParent(None)
            removed_phoneme_region = self.phoneme_regions.pop(phoneme_hash)
            for region_item in removed_phoneme_region.region_items:
                region_item.deleteLater()
            removed_phoneme_region.text_item.deleteLater()
            del removed_phoneme_region
            print("Removed phoneme regions:", phoneme_timestamp)

    def add_phoneme_target_lines(self, phoneme_timestamp: PhonemeTimestamp):
        if not phoneme_timestamp:
            return
        phoneme = phoneme_timestamp.phoneme
        start_time = phoneme_timestamp.start
        end_time = phoneme_timestamp.end
        if (
            not phoneme
            or start_time is None
            or end_time is None
            or start_time >= end_time
            or start_time < 0
            or end_time > self.sound.get_total_duration()
        ):
            return
        # self.remove_phoneme_target_lines(phoneme_timestamp)

        if phoneme_timestamp.formant_targets:
            for formant_target in phoneme_timestamp.formant_targets:
                if not formant_target.target_line:
                    formant_target.target_line = pg.InfiniteLine(
                        pos=formant_target.timestamp,
                        angle=90,
                        pen=pg.mkPen("m", width=2),
                        hoverPen=pg.mkPen("m", width=3),
                        movable=True,
                    )
                    self.spectrogram_plot.addItem(formant_target.target_line)
                    self.make_formant_target_line_connections(phoneme_timestamp)
                else:
                    formant_target.target_line.setPos(formant_target.timestamp)
                    if formant_target.target_line not in self.spectrogram_plot.items():
                        self.spectrogram_plot.addItem(formant_target.target_line)
                        self.make_formant_target_line_connections(phoneme_timestamp)

    def remove_phoneme_target_lines(self, phoneme_timestamp: PhonemeTimestamp):
        if not phoneme_timestamp or not phoneme_timestamp.formant_targets:
            return
        for formant_target in phoneme_timestamp.formant_targets:
            if formant_target.target_line:
                try:
                    # Try to disconnect, but handle if already disconnected
                    try:
                        formant_target.target_line.disconnect()
                    except (TypeError, RuntimeError):
                        pass  # Already disconnected or deleted

                    # Try to remove from plot, handling if already deleted
                    if formant_target.target_line in self.spectrogram_plot.items():
                        self.spectrogram_plot.removeItem(formant_target.target_line)
                        formant_target.target_line.setParent(None)

                    removed_target_line = formant_target.target_line
                    formant_target.target_line = None

                    # Try to delete, but handle if already deleted
                    try:
                        removed_target_line.deleteLater()
                    except RuntimeError:
                        pass  # Object already deleted
                except RuntimeError:
                    # Handle case where the C++ object has already been deleted
                    formant_target.target_line = None

    def make_formant_target_line_connections(self, phoneme_timestamp: PhonemeTimestamp):
        if phoneme_timestamp and phoneme_timestamp.formant_targets:
            for formant_target in phoneme_timestamp.formant_targets:
                phoneme_type = self.get_phoneme_type(phoneme_timestamp)
                plot_widget = self.get_phoneme_plot_widget(phoneme_type)
                if plot_widget and formant_target and formant_target.target_line:
                    # Disconnect previous connections to avoid multiple triggers
                    try:
                        formant_target.target_line.disconnect()
                    except TypeError:
                        pass
                    formant_target.target_line.sigPositionChanged.connect(
                        lambda: self.sync_ui_data_with_phoneme_formant_target_line(
                            phoneme_timestamp
                        )
                    )

    def sync_ui_data_with_phoneme_formant_target_line(
        self, phoneme_timestamp: PhonemeTimestamp
    ):
        phoneme_type = self.get_phoneme_type(phoneme_timestamp)
        plot_widget = self.get_phoneme_plot_widget(phoneme_type)
        self.formant_data.sync_formant_targets_with_target_lines(phoneme_timestamp)
        self.add_phoneme_to_ui_list(phoneme_timestamp)
        self.update_formant_scatter_plot(plot_widget)

    def remove_phoneme(self, phoneme_timestamp: PhonemeTimestamp):
        self.remove_phoneme_from_display(phoneme_timestamp)
        self.remove_phoneme_timestamp(phoneme_timestamp)

    def remove_phoneme_from_display(self, phoneme_timestamp: PhonemeTimestamp):
        self.remove_phoneme_target_lines(phoneme_timestamp)
        self.remove_phoneme_region(phoneme_timestamp)
        self.remove_phoneme_from_ui_list(phoneme_timestamp)

    def clear_phonemes(self):
        for phoneme_timestamp in self.phoneme_timestamps:
            self.remove_phoneme_from_display(phoneme_timestamp)
        self.phoneme_timestamps = []
        self.phoneme_regions = {}
        self.update_phoneme_picker_list_widget()
        self.update_all_formant_scatter_plots()
        print("Cleared phonemes.")

    def export_phoneme_formant_target_data(self):
        self.formant_data.clear_formant_target_record_file()
        for phoneme_timestamp in self.phoneme_timestamps:
            self.formant_data.write_formant_target_record(phoneme_timestamp)
        print("Exported phoneme formant target data.")

    def load_sentence_from_ui(self):
        sentence = self.sentence_text_input.text().strip()
        try:
            start = float(self.sentence_start_input.text())
            end = float(self.sentence_end_input.text())
        except ValueError:
            return
        if (
            not sentence
            or start is None
            or end is None
            or start >= end
            or start < 0
            or end > self.sound.get_total_duration()
        ):
            return
        print("Loading sentence from ui:", sentence)
        sentence_timestamp = self.add_sentence_timestamp(
            sentence, start_time=start, end_time=end
        )
        self.validate_and_display_sentence(sentence_timestamp)

        # Clear inputs
        self.sentence_text_input.clear()
        self.sentence_start_input.clear()
        self.sentence_end_input.clear()

    def load_phoneme_from_ui(self):
        phoneme = self.phoneme_text_input.text().strip()
        parent_sentence_text = self.phoneme_parent_sentence_input.text().strip()
        parent_word_text = self.phoneme_parent_word_input.text().strip()

        try:
            start = float(self.phoneme_start_input.text())
            end = float(self.phoneme_end_input.text())
        except ValueError:
            print("Error: Invalid start or end time for phoneme.")
            return

        if (
            not phoneme
            or start is None
            or end is None
            or start >= end
            or start < 0
            or end > self.sound.get_total_duration()
        ):
            print("Error: Invalid phoneme data.")
            return

        # Find parent sentence timestamp if provided
        parent_sentence_timestamp = None
        if parent_sentence_text:
            parent_sentence_timestamp = self.find_sentence_timestamp(
                parent_sentence_text
            )
            if not parent_sentence_timestamp:
                print(
                    f"Warning: Parent sentence '{parent_sentence_text}' not found. Phoneme will be added without parent sentence."
                )

        # Create parent word timestamp if provided
        parent_word_timestamp = None
        if parent_word_text:
            # Try to find the word in the TextGrid first
            parent_word_timestamp = self.find_word_in_textgrid(
                parent_word_text, start, end
            )

            if not parent_word_timestamp:
                # If not found in TextGrid, use phoneme's time boundaries as default
                print(
                    f"Warning: Word '{parent_word_text}' not found in TextGrid. Using phoneme boundaries."
                )
                parent_word_timestamp = WordTimestamp(
                    word=parent_word_text,
                    start=start,  # Default to phoneme's start
                    end=end,  # Default to phoneme's end
                )

        print(f"Loading phoneme from ui: {phoneme}")

        # Use the existing add_phoneme method
        self.add_phoneme(
            phoneme=phoneme,
            sentence_timestamp=parent_sentence_timestamp,
            word_timestamp=parent_word_timestamp,
            start_time=start,
            end_time=end,
        )

        # Get the phoneme timestamp that was just added
        added_phoneme_timestamp = None
        for pt in reversed(self.phoneme_timestamps):
            if pt.phoneme == phoneme and pt.start == start and pt.end == end:
                added_phoneme_timestamp = pt
                break

        if added_phoneme_timestamp:
            # Set formant targets if landmark info is loaded
            if self.LANDMARK_INFO_FILE and self.formant_data:
                self.set_formant_target_for_phoneme(added_phoneme_timestamp)
                # Update scatter plots if formant targets were set
                phoneme_type = self.get_phoneme_type(added_phoneme_timestamp)
                if phoneme_type:
                    plot_widget = self.get_phoneme_plot_widget(phoneme_type)
                    if plot_widget:
                        self.update_phoneme_picker_list_widget()
                        self.update_formant_scatter_plot(plot_widget)
            else:
                # Just add to UI list without formant targets
                if self.formant_data:
                    self.add_phoneme_to_ui_list(added_phoneme_timestamp)

        # Clear inputs
        self.phoneme_text_input.clear()
        self.phoneme_start_input.clear()
        self.phoneme_end_input.clear()
        self.phoneme_parent_sentence_input.clear()
        self.phoneme_parent_word_input.clear()

    def add_sentence_to_ui_list(self, sentence, *, start, end):

        if (
            not sentence
            or start is None
            or end is None
            or start >= end
            or start < 0
            or end > self.sound.get_total_duration()
        ):
            return

        # Create row widget
        if sentence in self.sentence_descriptor_items:
            self.remove_sentence_from_ui_list(sentence)
        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_widget.setLayout(row_layout)

        sentence_text = ensure_utf8_display(sentence)
        label = QLabel(f"{start:.3f} - {end:.3f}: {sentence_text}")
        delete_btn = QPushButton("Delete")
        delete_btn.setMaximumWidth(100)

        def remove_sentence():
            self.remove_sentence(sentence)

        delete_btn.clicked.connect(remove_sentence)

        row_layout.addWidget(label)
        row_layout.addWidget(delete_btn)
        self.sentence_descriptor_items[sentence] = (row_widget, label)
        self.sentence_descriptor_layout.addWidget(row_widget)
        print("Added sentence to UI list:", sentence)

    def remove_sentence_from_ui_list(self, sentence):
        if sentence in self.sentence_descriptor_items:
            self.sentence_descriptor_layout.removeWidget(
                self.sentence_descriptor_items[sentence][0]
            )
            self.sentence_descriptor_items[sentence][0].setParent(None)
            del self.sentence_descriptor_items[sentence]
            print("Removed sentence from UI list:", sentence)

    def plot_waveform(self):
        self.waveform_plot.clear()

        time_axis = self.sound.xs()
        audio_values = self.sound.values.T.flatten()
        max_amplitude = np.max(audio_values)
        min_amplitude = np.min(audio_values)
        max_time = np.max(time_axis)
        min_time = np.min(time_axis)
        waveform_plot_vb = self.waveform_plot.getViewBox()
        sentence_label_plot_vb = self.sentence_label_plot.getViewBox()
        waveform_plot_vb.setLimits(
            xMin=min_time,
            xMax=max_time,
            yMin=min_amplitude,
            yMax=max_amplitude,
            maxXRange=max_time - min_time,  # Lock zoom out to full data duration
            maxYRange=max_amplitude - min_amplitude,
        )
        sentence_label_plot_vb.setLimits(
            xMin=min_time,
            xMax=max_time,
            yMin=0,
            yMax=1,
            maxXRange=max_time - min_time,  # Lock zoom out to full data duration
            minYRange=1,
            maxYRange=1,
        )

        # Plot waveform
        self.waveform_plot.plot(time_axis, audio_values, pen=pg.mkPen("b", width=1))
        self.progress_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen("r", width=1),
            hoverPen=pg.mkPen("r", width=2),
            movable=True,
        )
        self.progress_line.sigDragged.connect(self.jump_to_progress_line)
        # self.progress_line.sigPositionChangeFinished.connect(self.jump_to_progress_line)
        self.waveform_plot.addItem(self.progress_line)

        waveform_plot_vb.sigRangeChanged.connect(self.sync_x_axes)
        sentence_label_plot_vb.sigRangeChanged.connect(self.sync_x_axes)

    def plot_spectrogram(self):
        self.spectrogram_plot.clear()

        sound_copy = self.sound.copy()
        sound_copy.pre_emphasize()
        if self.formant_data.PRAAT_FORMANT_TIME_STEP == 0.0:
            time_step = self.formant_data.PRAAT_FORMANT_WINDOW_LENGTH / 4
        else:
            time_step = self.formant_data.PRAAT_FORMANT_TIME_STEP
        spectrogram = sound_copy.to_spectrogram(
            window_length=self.SPECTROGRAM_WINDOW_LENGTH,
            maximum_frequency=self.formant_data.PRAAT_FORMANT_MAX_FREQ,
            time_step=time_step,
        )
        extracted_formant_data = self.extract_formants()
        print("Formants extracted.")
        print(
            extracted_formant_data[0].min(),
            extracted_formant_data[0].max(),
            extracted_formant_data[0].shape,
        )
        print(extracted_formant_data[-1].shape)

        X, Y = spectrogram.x_grid(), spectrogram.y_grid()
        sg_db = 20 * np.log10(spectrogram.values)
        sg_db_normalized = (sg_db - sg_db.min()) / (sg_db.max() - sg_db.min())

        # Set the spectrogram image
        self.spectogram_img.setImage(sg_db_normalized)
        # Apply viridis colormap
        viridis_lut = (
            colormaps.get_cmap("viridis")(np.linspace(0, 1, 256))[:, :3] * 255
        ).astype(np.uint8)
        self.spectogram_img.setLookupTable(viridis_lut)
        img_width = X[-1] - X[0]
        img_height = Y[-1] - Y[0]
        # pixel_size_x = self.spectogram_img.
        self.spectogram_img.setRect(X[0], Y[0], img_width, img_height)
        self.spectrogram_plot.addItem(self.spectogram_img)

        # self.spectrogram_plot.plot(extracted_formant_data[-1], extracted_formant_data[0][:,0], pen=pg.mkPen("r", width=1), name="F1")
        # self.spectrogram_plot.plot(extracted_formant_data[-1], extracted_formant_data[0][:,1], pen=pg.mkPen("r", width=1), name="F2")
        # self.spectrogram_plot.plot(extracted_formant_data[-1], extracted_formant_data[0][:,2], pen=pg.mkPen("r", width=1), name="F3")
        # self.spectrogram_plot.plot(extracted_formant_data[-1], extracted_formant_data[0][:,3], pen=pg.mkPen("r", width=1), name="F4")
        # self.spectrogram_plot.plot(extracted_formant_data[-1], extracted_formant_data[0][:,4], pen=pg.mkPen("r", width=1), name="F5")

        # Plot only known points (scatter), no continuous interpolation
        for i, formant_name in enumerate(["F1", "F2", "F3", "F4", "F5"]):
            if i >= extracted_formant_data[-1].shape[1]:
                break
            self.spectrogram_plot.plot(
                extracted_formant_data[0],
                extracted_formant_data[-1][:, i],
                pen=None,  # No line
                symbol="o",
                symbolBrush=pg.mkBrush("r"),
                symbolSize=6,
                name=formant_name,
            )

        spectrogram_plot_vb = self.spectrogram_plot.getViewBox()
        spectrogram_plot_vb.setLimits(
            xMin=X[0],
            xMax=X[-1],
            yMin=Y[0],
            yMax=Y[-1],
            maxXRange=X[-1] - X[0],  # Lock zoom out to full data duration
            maxYRange=Y[-1] - Y[0],
        )

        spectrogram_plot_vb.sigRangeChanged.connect(self.sync_x_axes)

    def plot_specific_formant_values_on_spectrogram(self, specific_formant_values):
        pass

    def plot_phoneme_labels(self):
        self.phoneme_label_plot.clear()

        time_axis = self.sound.xs()
        max_time = np.max(time_axis)
        min_time = np.min(time_axis)
        phoneme_label_plot_vb = self.phoneme_label_plot.getViewBox()
        phoneme_label_plot_vb.setLimits(
            xMin=min_time,
            xMax=max_time,
            yMin=0,
            yMax=1,
            maxXRange=max_time - min_time,  # Lock zoom out to full data duration
            minYRange=1,
            maxYRange=1,
        )
        phoneme_label_plot_vb.sigRangeChanged.connect(self.sync_x_axes)

    def add_phonemes_from_textgrid(self):
        if not self.COMBINED_PHONEMES_TEXTGRID_FILE or not os.path.exists(
            self.COMBINED_PHONEMES_TEXTGRID_FILE
        ):
            print("Error: No valid phoneme TextGrid file provided.")
            return
        tg_phonemes = textgrid.openTextgrid(
            self.COMBINED_PHONEMES_TEXTGRID_FILE, includeEmptyIntervals=False
        )

        self.clear_phonemes()

        sentence_tier = None
        for tier in tg_phonemes:
            if isinstance(tier, textgrid.IntervalTier) and tier.name == "sentences":
                sentence_tier = tier
                break
        word_tier = None
        for tier in tg_phonemes:
            if isinstance(tier, textgrid.IntervalTier) and tier.name == "words":
                word_tier = tier
                break

        @dataclass
        class TextGridSentence:
            label: str
            start: float
            end: float

        @dataclass
        class TextGridWord:
            label: str
            start: float
            end: float

        # Clear textgrid_interval_debug_log.txt once before writing
        with open("textgrid_interval_debug_log.txt", "w", encoding="utf-8") as f:
            pass
        for tier in tg_phonemes:
            if isinstance(tier, textgrid.IntervalTier) and tier.name == "phones":
                for interval in tier:
                    start, end, phoneme = interval.start, interval.end, interval.label
                    # Ensure proper UTF-8 encoding for phoneme text
                    phoneme = ensure_utf8_display(phoneme)

                    sentence_interval = None
                    for current_sentence_interval in sentence_tier:
                        if (
                            start
                            >= current_sentence_interval.start
                            - self.timestamp_tolerance
                            and end
                            <= current_sentence_interval.end + self.timestamp_tolerance
                        ):
                            sentence_interval = TextGridSentence(
                                label=current_sentence_interval.label,
                                start=current_sentence_interval.start,
                                end=current_sentence_interval.end,
                            )
                            break
                    if sentence_interval:
                        sentence_label = ensure_utf8_display(sentence_interval.label)
                        sentence_timestamp = self.find_sentence_timestamp(
                            sentence_label
                        )
                    else:
                        sentence_timestamp = None

                    word_interval = None
                    for current_word_interval in word_tier:
                        if (
                            start
                            >= current_word_interval.start - self.timestamp_tolerance
                            and end
                            <= current_word_interval.end + self.timestamp_tolerance
                        ):
                            word_interval = TextGridWord(
                                label=current_word_interval.label,
                                start=current_word_interval.start,
                                end=current_word_interval.end,
                            )
                            word_label = word_interval.label
                            word_label = word_label.replace(
                                "'", "’"
                            )  # Replace single quotes with proper UTF-8 apostrophe
                            word_label = ensure_utf8_display(word_label)
                            word_interval.label = word_label
                            break
                    if word_interval:
                        word_label = ensure_utf8_display(word_interval.label)
                        word_timestamp = WordTimestamp(
                            word=word_label,
                            start=word_interval.start,
                            end=word_interval.end,
                        )
                    else:
                        word_timestamp = None

                    # Print interval details to file for debugging
                    with open(
                        "textgrid_interval_debug_log.txt", "a", encoding="utf-8"
                    ) as f:
                        f.write(f"PHONEME: {phoneme}\n")
                        if sentence_interval:
                            f.write(
                                f"  SENTENCE: start={sentence_interval.start:.3f}, phoneme_start={start:.3f}, end={sentence_interval.end:.3f}, phoneme_end={end:.3f}, label={sentence_interval.label}\n"
                            )
                        if word_interval:
                            f.write(
                                f"  WORD: start={word_interval.start:.3f}, phoneme_start={start:.3f}, end={word_interval.end:.3f}, phoneme_end={end:.3f}, label={word_interval.label}\n"
                            )
                        f.write("\n")
                    self.add_phoneme(
                        phoneme=phoneme,
                        sentence_timestamp=sentence_timestamp,
                        word_timestamp=word_timestamp,
                        start_time=start,
                        end_time=end,
                    )
        print("Phonemes added from TextGrid.")

    def toggle_gender(self):
        if self.current_gender == "F":
            self.current_gender = "M"
            self.gender_toggle_button.setText("Gender: M")
        else:
            self.current_gender = "F"
            self.gender_toggle_button.setText("Gender: F")
        self.update_all_formant_scatter_plots()

    def export_plot_data_to_csv(self, plot_widget):
        """Export the scatter plot data to CSV, plus a 300 DPI PNG and SVG of the current plot state with the same base filename."""
        if not hasattr(plot_widget, "scatter_data") or not plot_widget.scatter_data:
            print("No data to export")
            return

        # Determine plot type from title
        plot_title = plot_widget.plotItem.titleLabel.text
        # Replace spaces with underscores and remove any special characters
        filename = (
            "".join(c for c in plot_title if c.isalnum() or c.isspace()).replace(
                " ", "_"
            )
            + ".csv"
        )

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV, PNG & SVG", filename, "CSV Files (*.csv)"
        )

        if not file_path:
            return

        base_path, ext = os.path.splitext(file_path)
        if not base_path:
            base_path = file_path
        csv_path = f"{base_path}.csv"
        png_path = f"{base_path}.png"
        svg_path = f"{base_path}.svg"

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["label", "x", "y"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for data in plot_widget.scatter_data:
                    # Data is already properly encoded when stored
                    writer.writerow(data)
            print(f"Data exported to {csv_path}")
        except Exception as e:
            print(f"Error exporting CSV data: {e}")

        # Export current plot state (zoomed range, axes limits/units, visible points/labels) to 300 DPI PNG and SVG
        plot_widget._exporting_lock = True
        try:
            # Ensure label visibility matches current view state before exporting
            plot_widget._exporting_lock = False
            self.update_scatter_label_visibility(plot_widget)
            plot_widget._exporting_lock = True

            # 1. Export SVG of current state
            svg_exporter = pg_exporters.SVGExporter(plot_widget.plotItem)
            svg_exporter.export(svg_path)
            print(f"SVG exported to {svg_path}")

            # 2. Export 300 DPI high-resolution PNG of current state
            target_dpi = 300
            screen_dpi = (
                plot_widget.logicalDpiX()
                if plot_widget.logicalDpiX() > 0
                else 96
            )
            scale_factor = target_dpi / float(screen_dpi)
            img_exporter = pg_exporters.ImageExporter(plot_widget.plotItem)
            base_width = max(int(plot_widget.plotItem.width()), plot_widget.width(), 400)
            img_exporter.parameters()["width"] = int(round(base_width * scale_factor))
            qimg = img_exporter.export(toBytes=True)
            dots_per_meter = int(round(target_dpi / 0.0254))
            qimg.setDotsPerMeterX(dots_per_meter)
            qimg.setDotsPerMeterY(dots_per_meter)
            qimg.save(png_path, "PNG")
            print(f"300 DPI PNG exported to {png_path}")
        except Exception as e:
            print(f"Error exporting plot images: {e}")
        finally:
            plot_widget._exporting_lock = False


if __name__ == "__main__":
    # Ensure UTF-8 encoding for the application
    try:
        locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, "C.UTF-8")
        except locale.Error:
            pass  # Fallback to default locale

    app = QApplication(sys.argv)
    # Set default text codec to UTF-8
    app.setAttribute(QtCore.Qt.ApplicationAttribute.AA_DontUseNativeDialogs, False)
    window = AutoSpeechAnalyzer()
    window.show()
    sys.exit(app.exec())
