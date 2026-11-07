# Main UI Workflows Documentation

This document describes the workflows triggered by each UI operation in `player_pyqt6.py`, starting from the top of the interface.

---

## 1. Menu Bar Operations

### 1.1 File → Open Audio
**Triggered by:** Menu action "Open Audio"  
**Entry point:** `menu_actions()` → `load_audio()`

**Workflow:**
1. Opens file dialog to select audio file (`.wav`)
2. Stops any currently playing audio
3. Sets `self.AUDIO_FILE` path and updates UI label
4. Loads audio using VLC media player
5. Clears all existing data:
   - `clear_sentences()` - removes all sentence data and UI elements
   - `clear_phonemes()` - removes all phoneme data and UI elements
   - `clear_formant_data()` - resets formant analysis data
6. Loads audio using **Parselmouth** library:
   - `Sound(self.AUDIO_FILE)` - creates Sound object
   - Extracts first channel only
7. Initializes formant extraction:
   - Creates `FormantData` object from **formants.py**
   - Parameters: 5 formants, sampling rate, ignore F0
8. Generates visualizations:
   - `plot_waveform()` - plots time-domain waveform
   - `plot_spectrogram()` - calls `extract_formants()` which uses `FormantData.extract_formant_data_from_sound()`
   - `plot_phoneme_labels()` - initializes phoneme label plot area
9. Sets up directory structure:
   - Creates experiment directory (default: `mfa_env1`)
   - Sets paths for sentence audio files and aligned phonemes
10. Attempts to load existing sentence annotations:
    - `load_sentence_timestamps_from_textgrid()` - reads `.TextGrid` file if exists
    - Uses **praatio** library to parse TextGrid format
    - Calls `add_sentence_timestamp()` for each sentence interval
11. Validates and displays sentences:
    - `validate_and_display_all_sentences()` - filters valid timestamps
    - For each valid sentence: `validate_and_display_sentence()`

**External functions called:**
- **formants.py:** `FormantData.__init__()`, `extract_formant_data_from_sound()`, `get_formant_times()`, `get_formant_values()`
- **helper.py:** `ensure_utf8_display()` - ensures proper UTF-8 text encoding
- **Parselmouth:** `Sound()`, `extract_channel()`, `to_spectrogram()`, `pre_emphasize()`
- **praatio:** `textgrid.openTextgrid()` - parses Praat TextGrid files

---

## 2. Playback Controls

### 2.1 Play Button
**Triggered by:** `play_button.clicked`  
**Entry point:** `toggle_playback()`

**Workflow:**
1. Checks playback state (`self.playing`)
2. If playing: calls `pause_audio()`
3. If paused: calls `play_audio()`

**play_audio() workflow:**
- Checks current playback position
- If past viewbox end, resets to viewbox start
- Starts VLC media player: `self.media_player.play()`
- Sets optional start time: `self.media_player.set_time(set_time)`
- Updates UI: button text to "Pause"
- Starts timer: `self.timer.start(10)` - triggers `update_progress()` every 10ms

**pause_audio() workflow:**
- Stops timer
- Pauses VLC player: `self.media_player.pause()`
- Updates UI: button text to "Play"

---

### 2.2 Play Window Button
**Triggered by:** `play_viewbox_button.clicked`  
**Entry point:** `play_audio_in_viewbox()`

**Workflow:**
1. Pauses current playback
2. Gets current viewbox start time from waveform plot
3. Calls `play_audio(set_time=viewbox_start_time)` to play from visible window start

---

### 2.3 Stop Button
**Triggered by:** `stop_button.clicked`  
**Entry point:** `stop_audio()`

**Workflow:**
1. Stops timer
2. Stops VLC player: `self.media_player.stop()`
3. Resets progress line to 0
4. Updates UI: button text to "Play"

---

## 3. Load Stimulus Button
**Triggered by:** `load_stimulus_button.clicked`  
**Entry point:** `load_stimulus()`

**Workflow:**
1. Opens file dialog to select stimulus text file (`.txt`)
2. Sets `self.STIMULUS_FILE` path
3. Updates UI label with file path
4. Stimulus file contains expected sentences for ASR validation

---

## 4. Run ASR Button
**Triggered by:** `run_asr_button.clicked`  
**Entry point:** `show_asr_menu()` → `run_ASR(asr_method)`

**Workflow:**
1. Shows dropdown menu with ASR method options:
   - **Whisper AI** - neural network-based ASR
   - **Subsequence DTW** - dynamic time warping-based ASR
2. Validates audio and stimulus files exist
3. Selects ASR engine:
   - `self.whisper_ASR` (WhisperX_ASR from **asr.py**)
   - `self.subseq_dtw_ASR` (SubsequenceDTW from **subsequence_dtw.py**)
4. Configures ASR target:
   - Calls ASR method's `set_asr_target()` with audio, stimulus, and TextGrid paths
5. Gets detected sentences:
   - Calls `ASR.get_valid_sentence_timestamps()` - returns list of `SentenceTimestamp` objects
6. Validates and displays:
   - `validate_and_display_all_sentences()` - filters and shows valid sentences
7. Saves results:
   - Calls `ASR.save_sentences_as_textgrid()` - exports to Praat TextGrid format

**External functions called:**
- **asr.py:** `WhisperX_ASR.set_asr_target()`, `WhisperX_ASR.get_valid_sentence_timestamps()`, `WhisperX_ASR.save_sentences_as_textgrid()`
- **subsequence_dtw.py:** `SubsequenceDTW.set_asr_target()`, `SubsequenceDTW.get_valid_sentence_timestamps()`, `SubsequenceDTW.save_sentences_as_textgrid()`
- **helper.py:** `SentenceTimestamp` dataclass

---

## 5. Run MFA Button
**Triggered by:** `run_mfa_button.clicked`  
**Entry point:** `run_MFA()`

**Workflow:**
1. Validates audio and stimulus files exist
2. Checks if phoneme TextGrid already exists (skips if found)
3. **Sentence segmentation:**
   - Iterates through valid sentence timestamps
   - For each sentence:
     - Extracts audio segment: `sound.extract_part(start_time, end_time)`
     - Saves as WAV: `sentence_N.wav`
     - Saves transcript as LAB file: `sentence_N.lab` (UTF-8 encoded)
     - Records TextGrid output path for later combination
4. **MFA alignment subprocess:**
   - Runs Montreal Forced Aligner in conda environment
   - Command: `conda run -n mfa_env1 python mfa_align.py --input_dir <SENTENCES_DIR>`
   - This calls **mfa_align.py** which:
     - Loads MFA acoustic and dictionary models
     - Performs forced alignment of audio to text
     - Generates word and phoneme-level TextGrids for each sentence
5. **TextGrid combination:**
   - Calls `combine_phoneme_textgrids(textgrid_offsets)`
   - Loads each sentence's TextGrid using **praatio**
   - Extracts three tiers:
     - **phones tier:** individual phonemes with timestamps
     - **words tier:** word boundaries
     - **sentences tier:** full sentence boundaries
   - Adjusts timestamps by adding sentence start offset
   - **Phoneme mapping:** converts British English to Australian English phonemes
     - Uses `map_british_to_australian()` with mapping from CSV
     - Special case: merges "ɪ" + "ə" → "ɪə" (centring diphthong)
   - Combines all tiers into single TextGrid file
   - Saves as `phones.TextGrid` with UTF-8 encoding
6. **Load phonemes into UI:**
   - Calls `add_phonemes_from_textgrid()`
   - Parses combined TextGrid
   - For each phoneme:
     - Finds parent sentence using timestamp overlap
     - Finds parent word using timestamp overlap
     - Creates `PhonemeTimestamp` object with references
     - Calls `add_phoneme()` to add to data structures and UI

**External functions called:**
- **mfa_align.py:** Entire script executed as subprocess (forced alignment)
- **helper.py:** `SentenceTimestamp`, `WordTimestamp`, `PhonemeTimestamp`, `ensure_utf8_display()`
- **praatio:** `textgrid.openTextgrid()`, `textgrid.Textgrid()`, `textgrid.IntervalTier()`
- **Parselmouth:** `sound.extract_part()`, `sound.save()`

---

## 6. Load Landmark Info Button
**Triggered by:** `load_landmark_info_button.clicked`  
**Entry point:** `show_landmark_info_menu()` → `load_landmark_info(cleaning_algo, use_ground_truth)`

**Workflow:**
1. Shows dropdown menu with formant cleaning options:
   - **with Viterbi cleaning** - uses Viterbi algorithm for formant tracking, with sub-options:
     - **with Ground Truth** - sets `self.USE_GROUND_TRUTH = True` (uses ground truth targets for emission cost)
     - **without Ground Truth** - sets `self.USE_GROUND_TRUTH = False`
   - **with NO cleaning** - raw formant values
2. Opens file dialog to select landmark info CSV
3. Sets `self.LANDMARK_INFO_FILE` path
4. Sets `self.USE_GROUND_TRUTH` (if Viterbi sub-option selected) and `self.formant_cleaning_algo` based on selection
5. Updates UI label
6. **Sets formant targets for all phonemes:**
   - Calls `set_formant_targets()`
   - Sets landmark file in FormantData: `formant_data.set_landmark_info_filepath()`
   - For each phoneme: `set_formant_target_for_phoneme()`

**set_formant_target_for_phoneme() workflow:**
- Removes old target lines from spectrogram
- Calls `FormantData.set_formant_target_for_phoneme()`:
  - Reads landmark CSV for phoneme-specific formant targets
  - Extracts formant values during phoneme duration
  - Applies selected cleaning algorithm (Viterbi/None)
  - Calculates formant targets (F1, F2, F3, F4, F5)
  - Stores in `phoneme_timestamp.formant_targets` list
- Plots formant tracks on spectrogram:
  - White circles: F1, Magenta: F2, Cyan: F3, Yellow: F4, Gray: F5
- Adds phoneme to UI descriptor list
- Adds movable target lines on spectrogram for interactive adjustment

**setup_formant_scatter_plot() workflow:**
- Populates axis selection dropdowns (Formant 1-5)
- Updates phoneme picker list: `update_phoneme_picker_list_widget()`
- Connects UI signals to update scatter plots
- Calls `update_all_formant_scatter_plots()` to render initial plots

**External functions called:**
- **formants.py:** `FormantData.set_landmark_info_filepath()`, `FormantData.set_formant_target_for_phoneme()`, `FormantData.formant_targets_to_descriptor_text()`
- **helper.py:** `PhonemeTimestamp`, `FormantTarget`, `get_phoneme_hash()`

---

## 7. Export Formant Targets Button
**Triggered by:** `export_formant_targets_button.clicked`  
**Entry point:** `export_phoneme_formant_target_data()`

**Workflow:**
1. Clears existing formant target record file
2. Iterates through all phoneme timestamps
3. For each phoneme: calls `FormantData.write_formant_target_record()`
   - Writes phoneme, parent word, parent sentence
   - Writes formant target values (F1, F2, F3, etc.)
   - Appends to CSV file

**External functions called:**
- **formants.py:** `FormantData.clear_formant_target_record_file()`, `FormantData.write_formant_target_record()`

---

## 8. Manual Sentence Input

### 8.1 Add Sentence Button
**Triggered by:** `add_sentence_button.clicked`  
**Entry point:** `load_sentence_from_ui()`

**Workflow:**
1. Reads input fields:
   - Sentence text
   - Start time (seconds)
   - End time (seconds)
2. Validates inputs (non-empty, valid time range)
3. Creates sentence timestamp: `add_sentence_timestamp()`
4. Validates and displays: `validate_and_display_sentence()`
   - Calls `add_sentence_region()` - creates visual regions on plots
   - Calls `add_sentence_to_ui_list()` - adds to sentence list with delete button
5. Clears input fields

**add_sentence_region() workflow:**
- Creates three synchronized `CustomLinearRegionItem` objects:
  - Waveform plot region
  - Sentence label plot region (with text label)
  - Spectrogram plot region
- Regions are movable and linked - dragging one updates all
- Connects signals:
  - `sigRegionChanged` → `sync_sentence_regions()` - synchronizes region positions
  - `sigHoverEvent` → `sync_hover_sentence_regions()` - synchronizes hover states
- Adds to `self.sentence_regions` dictionary
- Adds text label showing sentence at region center

---

## 9. Manual Phoneme Input

### 9.1 Add Phoneme Button
**Triggered by:** `add_phoneme_button.clicked`  
**Entry point:** `load_phoneme_from_ui()`

**Workflow:**
1. Reads input fields:
   - Phoneme symbol
   - Start time (seconds)
   - End time (seconds)
   - Parent sentence text (optional)
   - Parent word text (optional)
2. Validates inputs
3. **Finds parent sentence:**
   - Calls `find_sentence_timestamp(parent_sentence_text)`
   - Uses fuzzy matching with `difflib.get_close_matches()` if exact match fails
4. **Finds parent word:**
   - Calls `find_word_in_textgrid(parent_word_text, start, end)`
   - Searches word tier in combined TextGrid
   - Falls back to creating WordTimestamp with phoneme boundaries
5. **Adds phoneme:**
   - Calls `add_phoneme()` with all parameters
   - Creates `PhonemeTimestamp` object with parent references
   - Calls `add_phoneme_timestamp()` to add to list
   - Calls `add_phoneme_region()` to create visual elements
6. **Sets formant targets (if landmark info loaded):**
   - Calls `set_formant_target_for_phoneme()`
   - Determines phoneme type from ground truth CSV
   - Updates appropriate scatter plot (monophthong/centring/rising diphthong)
7. Clears input fields

**add_phoneme_region() workflow:**
- Creates two synchronized `CustomLinearRegionItem` objects:
  - Spectrogram plot region
  - Phoneme label plot region (with phoneme symbol)
- Regions are movable and linked
- Connects signals:
  - `sigRegionChanged` → `sync_phoneme_regions()` - synchronizes positions and updates formant targets
  - `sigHoverEvent` → `sync_hover_phoneme_regions()` - synchronizes hover states
- Adds to `self.phoneme_regions` dictionary

**External functions called:**
- **helper.py:** `PhonemeTimestamp`, `WordTimestamp`, `SentenceTimestamp`, `get_phoneme_hash()`, `ensure_utf8_display()`
- **formants.py:** `FormantData.set_formant_target_for_phoneme()`, `FormantData.formant_targets_to_descriptor_text()`

---

## 10. Formant Scatter Plots

### 10.1 Axis Selection (X/Y Formant Dropdowns)
**Triggered by:** `x_axis_combo.currentIndexChanged`, `y_axis_combo.currentIndexChanged`  
**Entry point:** `update_all_formant_scatter_plots()`

**Workflow:**
1. Calls `update_formant_scatter_plot()` for three plot widgets:
   - Monophthong plot
   - Centring diphthong plot
   - Rising diphthong plot
2. Each update:
   - Clears plot
   - Gets selected X and Y formant indices
   - Filters phonemes by type (reads from ground truth CSV)
   - Filters by selected phonemes in list widget
   - Plots scatter points colored by phoneme
   - Adds legend
   - Plots ground truth quadrilateral with ellipses (F1/F2 only)
   - Stores data for CSV export

**plot_formant_ground_truth_quadrilateral() workflow:**
- Reads `landmark_identification_ground_truth.csv`
- Filters by current gender (M/F) and plot type
- Only plots if axes are F1 and F2
- Extracts mean and SD for each phoneme
- Creates convex hull connecting all monophthong centroids
- Draws quadrilateral boundary
- Draws ellipse for each phoneme using SDs
- Labels each phoneme at its centroid

**External functions called:**
- **helper.py:** `PhonemeTimestamp.formant_targets`
- **scipy.spatial:** `ConvexHull` - computes convex hull of vowel space

---

### 10.2 Phoneme Selection List
**Triggered by:** `phoneme_picker_list_widget.itemSelectionChanged`  
**Entry point:** `update_all_formant_scatter_plots()`

**Workflow:**
1. Gets currently selected phonemes from list
2. Filters scatter plot data to show only selected phonemes
3. Updates all three scatter plots

---

### 10.3 Gender Toggle Button
**Triggered by:** `gender_toggle_button.clicked`  
**Entry point:** `toggle_gender()`

**Workflow:**
1. Toggles `self.current_gender` between "F" and "M"
2. Updates button text
3. Calls `update_all_formant_scatter_plots()`
4. Redraws ground truth quadrilaterals for new gender

---

### 10.4 Export CSV Buttons
**Triggered by:** Export button clicks under each scatter plot  
**Entry point:** `export_plot_data_to_csv(plot_widget)`

**Workflow:**
1. Checks if plot has data (`plot_widget.scatter_data`)
2. Opens save file dialog
3. Writes CSV with columns: label, x, y
4. Each row represents one phoneme instance with its formant values

---

## 11. Delete Operations

### 11.1 Delete Sentence
**Triggered by:** Delete button in sentence list  
**Entry point:** `remove_sentence()`

**Workflow:**
1. Calls `remove_sentence_from_display()`:
   - `remove_sentence_region()` - removes visual regions from plots
   - `remove_sentence_from_ui_list()` - removes from list widget
2. Calls `remove_sentence_timestamp()` - removes from data list

---

### 11.2 Delete Phoneme
**Triggered by:** Delete button in phoneme list  
**Entry point:** `remove_phoneme()`

**Workflow:**
1. Calls `remove_phoneme_from_display()`:
   - `remove_phoneme_region()` - removes visual regions
   - `remove_phoneme_target_lines()` - removes formant target lines from spectrogram
2. Calls `remove_phoneme_timestamp()` - removes from data list
3. Updates scatter plots

---

## 12. Interactive Plot Modifications

### 12.1 Dragging Sentence Region
**Triggered by:** Mouse drag on sentence region  
**Entry point:** `sync_sentence_regions()`

**Workflow:**
1. Gets new region bounds from dragged region
2. Identifies which sentence was modified
3. Updates all three region items (waveform, label, spectrogram)
4. Updates corresponding `SentenceTimestamp` start/end times
5. Re-sorts sentence list by start time
6. Updates UI label text with new timestamps

---

### 12.2 Dragging Phoneme Region
**Triggered by:** Mouse drag on phoneme region  
**Entry point:** `sync_phoneme_regions()`

**Workflow:**
1. Gets new region bounds from dragged region
2. Identifies which phoneme was modified
3. Updates both region items (spectrogram, phoneme label)
4. Updates corresponding `PhonemeTimestamp` start/end times
5. **Recalculates formant targets:**
   - If phoneme has formant targets, calls `set_formant_target_for_phoneme()`
   - Extracts new formant values for new time boundaries
   - Updates formant target lines on spectrogram
6. Re-sorts phoneme list by start time
7. Updates UI descriptor text
8. Updates corresponding scatter plot

**External functions called:**
- **formants.py:** `FormantData.set_formant_target_for_phoneme()`, `FormantData.formant_targets_to_descriptor_text()`

---

### 12.3 Dragging Progress Line
**Triggered by:** Mouse drag on red vertical line in waveform  
**Entry point:** `jump_to_progress_line()`

**Workflow:**
1. Pauses audio playback
2. Gets new time from progress line position
3. Converts to milliseconds
4. Calls `play_audio(set_time=new_time)` to jump to new position

---

## 13. Background Processes

### 13.1 Progress Update Timer
**Triggered by:** Timer every 10ms during playback  
**Entry point:** `update_progress()`

**Workflow:**
1. Gets current playback time from VLC player
2. Checks if reached end of audio or viewbox - pauses if true
3. Updates progress line position to current time

---

### 13.2 X-Axis Synchronization
**Triggered by:** Zoom or pan on any plot  
**Entry point:** `sync_x_axes()`

**Workflow:**
1. Gets new X-axis range from changed plot
2. Applies same range to all four plots:
   - Waveform plot
   - Sentence label plot
   - Spectrogram plot
   - Phoneme label plot
3. Ensures synchronized time navigation across all views

---

## Summary of External Module Dependencies

### **asr.py**
- `WhisperX_ASR` class - Neural ASR using WhisperX
  - `set_asr_target()` - configures ASR target
  - `get_valid_sentence_timestamps()` - returns detected sentences
  - `save_sentences_as_textgrid()` - exports to TextGrid format

### **subsequence_dtw.py**
- `SubsequenceDTW` class - DTW-based ASR
  - Same interface as WhisperX_ASR

### **formants.py**
- `FormantData` class - Formant extraction and analysis
  - `extract_formant_data_from_sound()` - extracts raw formant data
  - `get_formant_times()` / `get_formant_values()` - accessor methods
  - `set_landmark_info_filepath()` - loads phoneme-specific targets
  - `set_formant_target_for_phoneme()` - calculates targets with cleaning
  - `formant_targets_to_descriptor_text()` - formats for display
  - `write_formant_target_record()` - exports to CSV
  - `clear_formant_target_record_file()` - clears export file

### **helper.py**
- `SentenceTimestamp` dataclass - stores sentence data
- `WordTimestamp` dataclass - stores word data
- `PhonemeTimestamp` dataclass - stores phoneme data with parent references
- `FormantTarget` dataclass - stores formant measurements
- `get_phoneme_hash()` - generates unique phoneme identifier
- `ensure_utf8_display()` - fixes UTF-8 encoding issues

### **mfa_align.py**
- Executed as subprocess via conda
- Performs Montreal Forced Alignment
- Inputs: directory of sentence WAV + LAB files
- Outputs: TextGrid files with word and phoneme alignments

### **External Libraries:**
- **Parselmouth** - Praat functionality in Python (audio analysis)
- **praatio** - TextGrid file parsing
- **VLC** - Audio playback
- **PyQtGraph** - High-performance plotting
- **scipy.spatial** - Convex hull computation
- **pandas** - CSV data handling
