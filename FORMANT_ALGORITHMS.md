# Formant Analysis Algorithms Documentation

## Overview
The `formants.py` module provides comprehensive formant extraction, cleaning, and target identification for speech analysis. It uses Praat for raw formant data extraction and implements advanced outlier detection algorithms.

---

## Class: FormantData

### Configuration Constants
- **Praat Settings**: `PRAAT_FORMANT_TIME_STEP=0.0015s`, `PRAAT_N_FORMANTS=5`, `PRAAT_FORMANT_MAX_FREQ=5500Hz`, `PRAAT_FORMANT_WINDOW_LENGTH=0.025s`, `PRAAT_FORMANT_PREEMPHASIS_FROM=50Hz`
- **Viterbi Weights**: `VITERBI_EMISSION_MONOPHTHONG_WEIGHT=100.0`, `VITERBI_EMISSION_DIPHTHONG_WEIGHT=1000.0`, `VITERBI_DELTA=50.0`
- **Log Files**: `debug_log.txt`, `outlier_fix_log.txt`
- **Formant Data Files**: `formant_targets.csv`

---

## Core Functions

### 1. Formant Extraction

#### `extract_formant_data_from_sound(sound)`
**Purpose**: Extract formants from audio using Praat's Burg algorithm  
**Triggered**: Initialization, when processing new audio  
**Why**: Obtains raw formant frequency values (F1-F5) at regular time intervals  
**Output**: Populates `formant_times` and `formant_values` arrays; saves to `formants_values.csv`

#### `insert_formant_data_points(sound, start_time, end_time, n_points, praat_formant_max_freq)` (for future use)
**Purpose**: Extract formants for specific time segment with custom settings  
**Triggered**: When dynamic frequency ceiling is needed or re-analysis required  
**Why**: Allows targeted re-extraction with different parameters (e.g., max_freq=5000Hz for close F1-F2)  
**Output**: Inserts/replaces formant data in specified time range

---

### 2. Outlier Detection & Cleaning

#### `formant_clean_viterbi_fb(formant_values, formant_times, n_formants_to_clean, delta, targets, overall_emission_weight)`
**Purpose**: Clean formant trajectories using forward-backward Viterbi algorithm 
**Triggered**: Called by `fix_formant_values_outliers_for_phoneme_with_viterbi()`  
**Why**: Selects optimal formant candidate sequence that minimizes transition cost (Huber loss) and deviation from ground-truth targets (emission cost)
**Algorithm**:
- Generates all valid ascending formant candidates per frame
- Computes emission costs based on z-score deviation from target means/SDs
- Forward pass: accumulates minimum cumulative transition + emission cost
- Backward pass: accumulates minimum cumulative cost from future states (i.e., in backward direction)
- Combines passes to find globally optimal path, irrespective of direction (forward or backward)
**Parameters**:
- `targets`: List of dicts with `{start_frac, end_frac, mean, sd, weight}` from ground truth
- `delta`: Huber loss threshold for robust transition cost
- `overall_emission_weight`: Scaling factor for emission cost, applied on top of monophthong and diphthong emission weights

#### `formant_clean_shift_outliers(formant_values, formant_times, n_formants_to_clean, threshold)` (not used)
**Purpose**: Clean formants using modified z-score detection with cascading shift  
**Triggered**: Called by `fix_formant_values_outliers_for_phoneme_with_zscore_shift()`  
**Why**: Statistical outlier removal without requiring ground truth; shifts values from higher formants down if beneficial  
**Algorithm**:
- Detects outliers via modified z-score (MAD-based, threshold=3.5)
- Tests replacing outlier with value from higher formant
- If improvement, cascades all formants upward and marks top formant as NaN
- Optionally cascades rejected value downward to lower formants

#### `modified_z_score(data)` (not used)
**Purpose**: Compute robust outlier scores using Median Absolute Deviation (MAD)  
**Triggered**: Within cleaning algorithms  
**Why**: More robust to outliers than standard z-score; uses `0.6745 * (x - median) / MAD`

#### `huber_loss(x, delta)`
**Purpose**: Compute robust transition cost for Viterbi  
**Triggered**: During Viterbi forward/backward passes  
**Why**: Quadratic penalty for small changes (`≤delta`), linear for large changes; prevents over-penalization of legitimate formant movements

---

### 3. Formant Target Identification

#### `set_formant_target_for_phoneme(phoneme_timestamp, cleaning_algo, use_ground_truth, dynamic_frequency_ceiling)`
**Purpose**: Main entry point for finding formant targets in a phoneme  
**Triggered**: By UI or batch processing for each vowel  
**Why**: Coordinates cleaning, target identification, and recording  
**Steps**:
1. Optional: Apply dynamic frequency ceiling if needed (for future use)
2. Call `find_formant_target_for_phoneme()` with selected cleaning algorithm
3. Return cleaned formant data

#### `find_formant_target_for_phoneme(phoneme_timestamp, cleaning_algo, use_ground_truth)`
**Purpose**: Identify formant target timestamps based on rules provided in CSV
**Triggered**: After cleaning, or directly if `cleaning_algo=None`  
**Why**: Determines acoustic target points (e.g., "max_2" for F2 maximum, "mid" for midpoint)  
**Process**:
1. Load landmark info and phoneme mapping CSVs
2. Fuzzy match sentence and word
3. Map British→Australian phoneme if needed
4. Apply cleaning algorithm (viterbi/none)
5. Process targets via `process_target()` helper
6. Populate `phoneme_timestamp.formant_targets` list

#### `process_target(target_type, target_pct, target_pct_start, target_pct_end, phoneme_timestamp)`
**Purpose**: Calculate target timestamp based on type (time/max/min/mid/diff)  
**Triggered**: Within `find_formant_target_for_phoneme()`  
**Why**: Implements landmark identification rules from CSV  
**Target Types**:
- `time`: Fixed percentage of vowel duration
- `max_N`: Maximum of Nth formant in search window
- `min_N`: Minimum of Nth formant in search window
- `max_diff_N_M`: Maximum difference between Nth and Mth formants
- `min_diff_N_M`: Minimum difference between Nth and Mth formants
- `mid`: Median of formants in search window

---

### 4. Ground Truth Integration

#### `fix_formant_values_outliers_for_phoneme_with_viterbi(phoneme_timestamp, use_ground_truth)`
**Purpose**: Viterbi-based cleaning with optional ground truth targets  
**Triggered**: When `cleaning_algo='viterbi'` in `set_formant_target_for_phoneme()`  
**Why**: Uses phoneme-specific F1/F2 means and SDs to guide formant selection  
**Data Sources**:
- `landmark_identification_ground_truth.csv`: Gender-specific F1/F2 means/SDs for each phoneme
- `landmark_identification_mod.csv`: Target windows
**Target Construction**:
- Reads T1_F1_Mean/SD, T2_F1_Mean/SD, T1_F2_Mean/SD, T2_F2_Mean/SD
- Builds emission cost based on squared z-score deviation
- Applies higher weight (1000) for diphthongs vs monophthongs (100)

#### `generate_formant_data_points_with_dynamic_frequency_ceiling(phoneme_timestamp)` (for future use)
**Purpose**: Re-extract formants with max_freq=5000Hz if F1-F2 overlap risk detected  
**Triggered**: When `dynamic_frequency_ceiling=True`  
**Why**: Praat's default 5500Hz ceiling can cause F1-F2 misidentification when formants are close  
**Condition**: `(F2_mean - F2_sd) - (F1_mean + F1_sd) <= 1000 AND F2_mean < 2000`

---

### 5. Utility Functions

#### `get_formant_values_at_time(time)`
**Purpose**: Retrieve formant values at specific timestamp  
**Triggered**: UI queries, target synchronization  
**Why**: Interpolates to closest available timestamp

#### `write_formant_target_record(phoneme_timestamp)`
**Purpose**: Append formant targets to CSV log  
**Triggered**: After target identification  
**Why**: Records all identified targets for analysis/validation  
**Format**: `sentence;word;phoneme;start;end;target1_timestamp;target1_f1;...;target2_f4`

#### `sync_formant_targets_with_target_lines(phoneme_timestamp)`
**Purpose**: Update target timestamps when user moves vertical lines (representing targets) in UI  
**Triggered**: UI drag events  
**Why**: Ensures targets reflect user adjustments

#### `formant_targets_to_descriptor_text(phoneme_timestamp)`
**Purpose**: Generate human-readable target summary  
**Triggered**: UI display updates  
**Why**: Shows targets in format: `"0.123s - 0.456s: word | phoneme --- Target 1 at 0.234s, Target 2 at 0.345s"`

---

## Workflow Integration

### Typical Processing Flow:
1. **Extract**: `extract_formant_data_from_sound()` → Raw formant data
2. **Clean**: `fix_formant_values_outliers_for_phoneme_with_viterbi()` → Outlier-corrected trajectories
3. **Identify**: `find_formant_target_for_phoneme()` → Target timestamps
4. **Record**: `write_formant_target_record()` → CSV logging
5. **Display**: `formant_targets_to_descriptor_text()` → UI feedback

### Algorithm Selection:
- **Viterbi (`cleaning_algo='viterbi'`)**: Use with (`use_ground_truth=True`, via `Load Landmark Info -> with Viterbi cleaning -> with Ground Truth`) or without (`use_ground_truth=False`, via `Load Landmark Info -> with Viterbi cleaning -> without Ground Truth`) ground truth
- **None (`cleaning_algo=None`)**: Skip cleaning; use raw Praat output

---

## Dependencies
- **External**: `parselmouth` (Praat interface), `pandas`, `numpy`, `scipy.stats`
- **Internal**: `helper.py` (PhonemeTimestamp, FormantTarget classes)
- **CSV Files**:
  - `landmark_identification_mod.csv`: Target rules per phoneme
  - `landmark_identification_ground_truth.csv`: F1/F2 statistics per phoneme/gender
  - `landmark_identification_phoneme_mapping.csv`: British↔Australian phoneme mappings

---

## Key Design Decisions
- **Forward-Backward Viterbi**: Combines forward and backward passes for globally optimal trajectory
- **Huber Loss**: Balances continuity (small changes) with flexibility (large legitimate changes, for example, diphthongs)
- **Modified Z-Score**: Uses MAD instead of standard deviation for outlier robustness (not used)
- **Dynamic Frequency Ceiling**: Addresses erroneous formant readings by Praat when Praat's default ceiling may misidentify formants (for future use)
- **Cascading Shifts**: Allows higher formant values to "drop down" when beneficial
