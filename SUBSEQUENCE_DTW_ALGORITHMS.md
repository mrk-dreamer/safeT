# SubsequenceDTW Algorithm Documentation

## Overview

The `subsequence_dtw.py` module implements a Dynamic Time Warping (DTW) based approach for sentence segmentation in speech recordings. The system uses subsequence DTW alignment between synthesized reference audio and target recordings to accurately identify and timestamp sentences within audio files.

## Core Components

### SubsequenceDTW Class

The main class that orchestrates the Subsequence DTW-based sentence segmentation pipeline.

---

## Algorithms

### 1. Initialization and Model Setup

#### Algorithm: `__init__()`

**Purpose**: Initialize the SubsequenceDTW system with appropriate audio processing parameters and TTS models.

**Input Parameters**:
- `sr` (int, default=16000): Sample rate for audio processing in Hz
- `hop_length` (int, default=512): Temporal resolution in samples for feature extraction
- `n_mfcc` (int, default=13): Number of Mel-Frequency Cepstral Coefficients computed at each frame
- `delta_order` (int, default=2): Order of delta features (0=static only, 1=+delta (1st order derivative), 2=+delta-delta (2nd order derivative))
- `smoothing_sigma` (float, default=0.5): Gaussian smoothing parameter for cost curve
- `min_duration_frac` (float, default=0.7): Threshold for acceptable minimum matched duration as fraction of reference duration
- `max_duration_frac` (float, default=2.3): Threshold for acceptable maximum matched duration as fraction of reference duration
- `min_rms` (float, default=1e-4): Minimum RMS energy threshold for acceptable sentence segment speech
- `tts_tld` (str, default="com.au"): TTS accent/locale (e.g., "com.au" for Australian English) used by gTTS
- `timestamp_tolerance` (float, default=0.05): Tolerance in seconds for timestamp boundary adjustments
- `voice_model` (str, default="piper_male"): Default TTS voice model used to synthesize reference audio (`"piper_male"`, `"piper_female"`, `"gtts"`, or `"kitten:<voice>"`)

**Process**:

1. **Audio Processing Configuration**:
   - Sets sample rate and hop length to be used throughout the pipeline
   - Configures MFCC feature extraction parameters
   - Sets delta feature order for enhanced temporal dynamics

2. **DTW Alignment Configuration**:
   - Sets smoothing parameters for cost curve refinement
   - Configures duration constraints to reject unrealistic matches
   - Sets minimum energy threshold to filter silence

3. **TTS Model Loading**:
   - **Google TTS (gTTS)**: TTS model (Australian English female voice used)
   - **KittenTTS**: Neural TTS with male and female voice options (kitten-tts-nano-0.2 model used. No accent options. Speed adjustment available)
   - **Piper TTS**: Neural TTS with male/female voices
     - Female voice: en_GB-alba-medium.onnx (British English)
     - Male voice: en_GB-alan-medium.onnx (British English)
     - Configured with custom synthesis parameters (volume, speed, variation)

4. **Path Initialization**:
   - Initializes placeholders for audio, stimulus, and TextGrid paths
   - To be set via `set_asr_target()` method

**Output**: Initialized SubsequenceDTW object with loaded TTS models and configured parameters

---

### 2. Feature Extraction

#### Algorithm: `compute_mfcc_feat()`

**Purpose**: Extract MFCC features with optional delta and delta-delta coefficients for robust acoustic representation.

**Input Parameters**:
- `y`: Audio time series (numpy array)
- `sr`: Sample rate (uses self.sr if None)
- `n_mfcc`: Number of MFCCs to be generated (uses self.n_mfcc if None)
- `hop_length`: Hop length for frame extraction (uses self.hop_length if None)
- `delta_order`: Delta order (derivative order to be used for MFCC generation, uses self.delta_order if None)

**Mathematical Formulation**:

**MFCC Computation**:
$$\text{MFCC}(n, f) = \sum_{k=0}^{K-1} \log(S_{\text{mel}}(k, f)) \cos\left(n\left(k + \frac{1}{2}\right)\frac{\pi}{K}\right)$$

Where:
- $S_{\text{mel}}(k, f)$ = Mel-spectrogram at mel-band $k$, frame $f$
- $n$ = MFCC coefficient index (0 to `n_mfcc-1`)
- $K$ = Number of mel-bands

**Delta Features**:
$$\Delta(n, f) = \frac{\sum_{t=1}^{T} t \cdot (\text{MFCC}(n, f+t) - \text{MFCC}(n, f-t))}{2\sum_{t=1}^{T} t^2}$$

**Delta-Delta Features**:
$$\Delta\Delta(n, f) = \Delta[\Delta(n, f)]$$

**Process**:

1. **Static MFCC Extraction**:
   ```
   mfcc = librosa.feature.mfcc(
       y=audio,
       sr=sample_rate,
       n_mfcc=n_mfcc,
       hop_length=hop_length
   )
   features = [mfcc]
   ```

2. **Dynamic Features** (if delta_order ≥ 1):
   ```
   IF delta_order >= 1:
       delta = librosa.feature.delta(mfcc, order=1)
       features.append(delta)
   
   IF delta_order >= 2:
       delta_delta = librosa.feature.delta(mfcc, order=2)
       features.append(delta_delta)
   ```

3. **Feature Concatenation**:
   ```
   feature_matrix = vstack(features)  // Shape: (n_features, n_frames)
   ```

**Output**: Feature matrix with shape `(n_features, n_frames)` where:
- `n_features` = n_mfcc × (1 + delta_order)
- For default parameters: 13 × 3 = 39 features per frame

**Feature Dimensionality**:
- delta_order=0: 13 dimensions (static only)
- delta_order=1: 26 dimensions (static + delta)
- delta_order=2: 39 dimensions (static + delta + delta-delta)

---

#### Algorithm: `cmvn()`

**Purpose**: Apply Cepstral Mean and Variance Normalization to reduce speaker and channel variability.

**Input Parameters**:
- `X`: Feature matrix of shape `(n_features, n_frames)`
- `eps` (default=1e-8): Small constant for numerical stability

**Mathematical Formulation**:

$$\hat{X}(i, j) = \frac{X(i, j) - \mu_i}{\sigma_i + \epsilon}$$

Where:
- $\mu_i = \frac{1}{T}\sum_{j=1}^{T} X(i, j)$ (mean of feature $i$ across all frames)
- $\sigma_i = \sqrt{\frac{1}{T}\sum_{j=1}^{T} (X(i, j) - \mu_i)^2}$ (standard deviation)
- $\epsilon$ = small constant to prevent division by zero

**Process**:
```
FOR each feature dimension i:
    mu[i] = mean(X[i, :])
    sigma[i] = std(X[i, :])
    X_norm[i, :] = (X[i, :] - mu[i]) / (sigma[i] + eps)
```

**Output**: Normalized feature matrix with zero mean and unit variance per dimension

**Benefits**:
- Removes channel effects and speaker characteristics
- Improves robustness to recording conditions
- Essential for DTW distance calculations

---

### 3. Reference Audio Generation

#### Algorithm: `set_voice_model()`

**Purpose**: Select the TTS voice model used by `generate_reference_audio()` (also selectable from the GUI under `Run ASR -> Subsequence DTW`).

**Input Parameters**:
- `voice_model` (str, default=`"piper_male"`):
  - `"piper_male"` (or `"piper_alan"`): Piper TTS Male (`en_GB-alan-medium.onnx`) [default]
  - `"piper_female"` (or `"piper_alba"`): Piper TTS Female (`en_GB-alba-medium.onnx`)
  - `"gtts"`: Google TTS (Australian English via `self.tts_tld`)
  - `"kitten:<voice>"` (e.g., `"kitten:expr-voice-2-f"`, `"kitten:expr-voice-2-m"`, ..., `"kitten:expr-voice-5-m"`) or `"kitten_female"` / `"kitten_male"`

---

#### Algorithm: `generate_reference_audio()`

**Purpose**: Synthesize reference audio from target sentence text using the selected Text-to-Speech voice model.

**Input Parameters**:
- `sentence` (str): Text to synthesize
- `output_path` (str): Path to save the generated audio file
- `voice_model` (str | None, default=None): Optional per-call override for the TTS voice model (defaults to `self.voice_model` or `"piper_male"`)

**Process**:

Dispatches based on `model = (voice_model or self.voice_model or "piper_male").lower()`:

**Piper TTS (`"piper_male"` [default] / `"piper_female"`)**:
```
voice = piper_tts_model_male  // or piper_tts_model_female
voice.synthesize_wav(sentence, output_file, config)
ref_audio = load_audio(output_path, sr=self.sr)
```

Configuration:
- volume: 0.5 (half loudness)
- length_scale: 1.3 (30% slower speech)
- noise_scale: 1.0 (audio variation)
- noise_w_scale: 1.0 (speaking variation)

**Google TTS (`"gtts"`)**:
```
tts = gTTS(
    text=sentence,
    lang="en",
    tld=self.tts_tld,  // Default: "com.au" for Australian accent
    slow=False
)
tts.save(output_path)
ref_audio = load_audio(output_path, sr=self.sr)
```

**KittenTTS (`"kitten:<voice>"` / `"kitten_female"` / `"kitten_male"`)**:
```
audio = kitten_tts_model.generate(
    sentence,
    voice=kitten_voice,  // e.g. 'expr-voice-2-f' or 'expr-voice-2-m'
    speed=1.2
)
save_audio(output_path, audio, sr=24000)
ref_audio = resample(audio, 24000 -> self.sr)
```

Available voices:
- `expr-voice-2-m/f`: Style 2 (male/female)
- `expr-voice-3-m/f`: Style 3 (male/female)
- `expr-voice-4-m/f`: Style 4 (male/female)
- `expr-voice-5-m/f`: Style 5 (male/female)

**Output**: 
- Audio array (numpy array) at 16kHz sample rate
- WAV file saved to specified path

**TTS Model Selection**:
- **gTTS**: Simple, reliable, cloud-based (requires internet)
- **KittenTTS**: Fast, local
- **Piper TTS**: Highest quality, local, configurable prosody

---

### 4. Subsequence DTW Alignment

#### Algorithm: `find_sentence_match()`

**Purpose**: Find the best temporal match for reference audio within a long audio recording using subsequence Dynamic Time Warping.

**Input Parameters**:
- `ref_audio`: Reference audio array (from TTS)
- `long_audio`: Target audio array (full recording)

**Mathematical Formulation**:

**Cost Matrix**:
$$C(i, j) = d(\text{ref}[i], \text{long}[j])$$

Where $d(\cdot, \cdot)$ is cosine distance:
$$d(\mathbf{x}, \mathbf{y}) = 1 - \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}$$

**Subsequence DTW Accumulation**:
$$D(i, j) = C(i, j) + \min\begin{cases}
D(i-1, j-1) & \text{(diagonal)} \\
D(i-1, j) & \text{(vertical)} \\
D(i, j-1) & \text{(horizontal)}
\end{cases}$$

**Initialization** (key difference from standard DTW):
$$D(0, j) = 0 \quad \forall j \in [0, M]$$

This allows the reference to match anywhere in the long audio.

**Normalized Cost with Duration Penalty**:
$$\text{Cost}_{\text{norm}}(j_{\text{end}}) = \frac{D(N, j_{\text{end}})}{N} \times \left(1 + \frac{|\text{dur}_{\text{match}} - \text{dur}_{\text{ref}}|}{\text{dur}_{\text{ref}}}\right)$$

**Process**:

1. **Feature Extraction**:
   ```
   ref_feat = compute_mfcc_feat(ref_audio)
   long_feat = compute_mfcc_feat(long_audio)
   
   ref_feat = cmvn(ref_feat)    // Normalize features
   long_feat = cmvn(long_feat)
   
   N = ref_feat.shape[1]        // Reference length
   M = long_feat.shape[1]       // Long audio length
   ref_duration = N * hop_length / sr
   ```

2. **Silence Detection and Masking**:
   ```
   frame_rms = compute_rms(long_audio, hop_length)
   silent_mask = (frame_rms < min_rms)
   
   // Create cost matrix
   C = cosine_distance(ref_feat.T, long_feat.T)
   C[:, silent_mask] = 1.0  // Heavily penalize silent frames
   ```

3. **Short Audio Handling** (M < N):
   ```
   IF M < N:
       // Use full DTW instead of subsequence DTW
       D[0, :] = 0
       D[0, 0] = 0  // Enforce single start
       
       // Standard DTW accumulation
       FOR i in 1..N:
           FOR j in 1..M:
               D[i,j] = C[i-1,j-1] + min(D[i-1,j-1], D[i-1,j], D[i,j-1])
       
       j_end = argmin(D[N, 1:]) + 1
       // Backtrack to find j_start
       path = backtrack(D, N, j_end)
       j_start = path[0][1]
   ```

4. **Subsequence DTW** (M ≥ N):
   ```
   // Initialize: allow match to start anywhere
   D[0, :] = 0
   D[0, 0] = 0
   
   // Forward pass
   FOR i in 1..N:
       FOR j in 1..M:
           D[i,j] = C[i-1,j-1] + min(D[i-1,j-1], D[i-1,j], D[i,j-1])
   
   // Evaluate all possible endpoints
   FOR j_end in 1..M:
       IF D[N, j_end] is finite:
           // Backtrack to find path
           path = backtrack(D, N, j_end)
           j_start = path[0][1]
           
           // Calculate normalized cost with duration penalty
           duration = (j_end - j_start) * hop_length / sr
           dur_penalty = 1.0 + |duration - ref_duration| / ref_duration
           norm_costs[j_end] = D[N, j_end] / N * dur_penalty
           
           // Store path information
           starts[j_end] = j_start
           paths[j_end] = path
   ```

5. **Cost Curve Smoothing**:
   ```
   // Replace infinities with large value
   costs_vec = norm_costs[1:]
   costs_vec[isinf(costs_vec)] = max(finite_costs) * 2
   
   // Apply Gaussian smoothing
   smooth_costs = gaussian_filter1d(costs_vec, sigma=smoothing_sigma)
   ```

6. **Best Match Selection with Duration Constraints**:
   ```
   // Get top 10 candidates by cost
   sorted_indices = argsort(smooth_costs)[:10]
   
   best_j_end = None
   FOR candidate in sorted_indices:
       j_end = candidate + 1
       j_start = starts[j_end]
       duration = (j_end - j_start) * hop_length / sr
       
       // Check duration constraints
       IF min_duration_frac * ref_duration <= duration <= max_duration_frac * ref_duration:
           best_j_end = j_end
           best_j_start = j_start
           BREAK
   
   // Fallback to best cost if no valid duration found
   IF best_j_end is None:
       best_j_end = argmin(smooth_costs) + 1
       best_j_start = starts[best_j_end]
   ```

7. **Energy Validation**:
   ```
   start_time = best_j_start * hop_length / sr
   end_time = best_j_end * hop_length / sr
   
   matched_audio = long_audio[start_sample:end_sample]
   matched_rms = sqrt(mean(matched_audio^2))
   
   IF matched_rms < min_rms:
       WARN("Low energy in matched region - possible false positive")
   ```

**Output**: Dictionary containing:
```python
{
    "start": float,              # Start time in seconds
    "end": float,                # End time in seconds
    "norm_cost": float,          # Normalized DTW cost with duration penalty
    "matched_duration": float,   # Duration of matched segment
    "matched_rms": float        # RMS energy of matched segment
}
```

**Key Differences from Standard DTW**:
1. **Initialization**: D[0, :] = 0 allows flexible starting points
2. **Multiple end points**: Evaluates all possible end points, not just one
3. **Duration Penalty**: Penalizes unrealistic duration deviations
4. **Silence Masking**: Explicitly penalizes silent regions
5. **Cost Smoothing**: Reduces sensitivity to local minima

**Duration Constraint Logic**:
- Minimum: 0.7 × reference duration (allows for fast speech)
- Maximum: 2.3 × reference duration (allows for slow speech)
- Rejects matches outside this range unless no alternatives exist

---

### 5. Overlap Resolution Algorithm

#### Algorithm: `remove_timestamp_overlaps()`

**Purpose**: Eliminate temporal overlaps between consecutive sentence timestamps by adjusting boundaries.

**Input Parameters**:
- `timestamps`: List of `SentenceTimestamp` objects
- `tolerance` (float, default=0.05): Minimum gap between sentences in seconds

**Mathematical Formulation**:

Given two consecutive sentences $S_i$ and $S_{i+1}$:

**No Overlap Condition**:
$$t_{\text{start}}^{i+1} > t_{\text{end}}^{i} + \tau$$

Where $\tau$ is the tolerance parameter.

**Overlap Resolution**:
$$t_{\text{midpoint}} = \frac{t_{\text{end}}^{i} + t_{\text{start}}^{i+1}}{2}$$

$$t_{\text{end}}^{i} \leftarrow t_{\text{midpoint}} - \frac{\tau}{2}$$

$$t_{\text{start}}^{i+1} \leftarrow t_{\text{midpoint}} + \frac{\tau}{2}$$

**Process**:

1. **Validation and Filtering**:
   ```
   valid_timestamps = [ts for ts in timestamps 
                       if ts.start is not None and ts.end is not None]
   invalid_timestamps = [ts for ts in timestamps 
                         if ts.start is None or ts.end is None]
   
   IF no valid_timestamps:
       RETURN original timestamps
   ```

2. **Sort by Time**:
   ```
   valid_timestamps.sort(key=lambda x: x.start)
   non_overlapping = [valid_timestamps[0]]
   ```

3. **Overlap Detection and Resolution**:
   ```
   FOR current in valid_timestamps[1:]:
       last = non_overlapping[-1]
       
       IF current.start > last.end + tolerance:
           // No overlap - add segment as-is
           non_overlapping.append(current)
       
       ELSE:
           // Overlap detected - adjust boundaries
           midpoint = (last.end + current.start) / 2
           
           // Create adjusted timestamps
           adjusted_last = SentenceTimestamp(
               sentence=last.sentence,
               start=last.start,
               end=midpoint - (tolerance / 2)
           )
           non_overlapping[-1] = adjusted_last
           
           adjusted_current = SentenceTimestamp(
               sentence=current.sentence,
               start=midpoint + (tolerance / 2),
               end=current.end
           )
           non_overlapping.append(adjusted_current)
   ```

4. **Preserve Invalid Timestamps**:
   ```
   // Append sentences without valid timestamps at end
   RETURN non_overlapping + invalid_timestamps
   ```

**Output**: List of non-overlapping `SentenceTimestamp` objects

**Properties**:
- Preserves temporal ordering
- Creates minimum gap of `tolerance` between segments
- Splits overlap evenly between adjacent segments
- Preserves sentences that couldn't be detected (None timestamps)

**Example**:
```
Input (tolerance = 0.05):
  Sentence 1: [1.0, 3.2]  "The cat sat"
  Sentence 2: [3.0, 5.5]  "on the mat"  // Overlaps by 0.2s
  Sentence 3: [6.0, 8.0]  "yesterday"   // No overlap

Process:
  S1 and S2 overlap (3.0 < 3.2 + 0.05)
  midpoint = (3.2 + 3.0) / 2 = 3.1
  S1.end = 3.1 - 0.025 = 3.075
  S2.start = 3.1 + 0.025 = 3.125
  S3 unchanged (no overlap)

Output:
  Sentence 1: [1.0, 3.075]
  Sentence 2: [3.125, 5.5]
  Sentence 3: [6.0, 8.0]
```

---

### 6. Main Pipeline Algorithm

#### Algorithm: `get_valid_sentence_timestamps()`

**Purpose**: Complete end-to-end pipeline for extracting sentence timestamps from audio using DTW alignment.

**Process Flow**:

1. **Validation**:
   ```
   IF audio_path does not exist:
       RAISE ValueError("No valid audio path")
   
   stimulus_sentences = load_stimulus_sentences()
   IF stimulus_sentences is empty:
       RAISE ValueError("No valid sentences in stimulus file")
   ```

2. **Audio Loading**:
   ```
   long_audio = load_audio(audio_path, sr=16000)
   ```

3. **Temporary Directory Setup**:
   ```
   temp_dir = create_directory(audio_dir / "dtw_temp")
   ```

4. **Sentence Processing Loop**:
   ```
   valid_timestamps = []
   metadata_list = []
   
   FOR each sentence in stimulus_sentences:
       PRINT("Processing sentence {idx+1}/{total}: '{sentence}'")
       
       TRY:
           // Generate reference audio
           ref_path = temp_dir / f"ref_sentence_{idx+1}.wav"
           ref_audio = generate_reference_audio(sentence, ref_path)
           
           // Find match using DTW
           match = find_sentence_match(ref_audio, long_audio)
           
           IF match exists:
               // Apply timestamp tolerance
               audio_duration = get_audio_duration(audio_path)
               start_time = max(0, match["start"] - timestamp_tolerance)
               end_time = min(audio_duration, match["end"] + timestamp_tolerance)
               
               // Store result
               valid_timestamps.append(
                   SentenceTimestamp(
                       sentence=sentence,
                       start=start_time,
                       end=end_time
                   )
               )
               metadata_list.append({
                   "norm_cost": match["norm_cost"],
                   "matched_rms": match["matched_rms"]
               })
               
               PRINT(f"✓ Match found: {start_time:.3f}s - {end_time:.3f}s")
           
           ELSE:
               PRINT(f"✗ No match found")
               valid_timestamps.append(
                   SentenceTimestamp(sentence, start=None, end=None)
               )
               metadata_list.append({
                   "norm_cost": None,
                   "matched_rms": None
               })
       
       CATCH Exception as e:
           PRINT(f"✗ Error: {e}")
           valid_timestamps.append(
               SentenceTimestamp(sentence, start=None, end=None)
           )
           metadata_list.append({
               "norm_cost": None,
               "matched_rms": None
           })
   ```

5. **Optional Overlap Removal** (currently commented out):
   ```
   // valid_timestamps = remove_timestamp_overlaps(
   //     valid_timestamps,
   //     tolerance=timestamp_tolerance
   // )
   ```

6. **Debug Logging**:
   ```
   save_results_to_json(valid_timestamps, metadata_list)
   ```

**Output**: List of `SentenceTimestamp` objects containing:
- Sentence text
- Start time (or None if not found)
- End time (or None if not found)

**Pipeline Stages Visualization**:

```
Audio Input + Stimulus Text
    ↓
[Load Long Audio] → Full recording at 16kHz
    ↓
FOR each stimulus sentence:
    ↓
    [TTS Generation] → Reference audio for sentence
    ↓
    [Feature Extraction] → MFCC + Delta features
    ↓
    [CMVN Normalization] → Channel-independent features
    ↓
    [Cost Matrix + Silence Mask] → Penalize silent regions
    ↓
    [Subsequence DTW] → Find all potential matches
    ↓
    [Cost Smoothing] → Reduce local minima sensitivity
    ↓
    [Duration Filtering] → Select matches of expected duration
    ↓
    [Energy Validation] → Verify speech presence
    ↓
    [Timestamp + Tolerance] → Final boundaries
    ↓
END FOR
    ↓
[Optional Overlap Resolution] → Non-overlapping timestamps
    ↓
[JSON Export] → Debug information
    ↓
Output: SentenceTimestamp list
```

**Timestamp Tolerance Application**:
- Extends boundaries slightly to capture full sentence
- `start_time = match_start - tolerance` (minimum: 0)
- `end_time = match_end + tolerance` (maximum: audio duration)

**Error Handling**:
- Individual sentence failures don't stop pipeline
- Failed matches recorded with None timestamps
- All errors logged with sentence context

---

### 7. Debug Output Algorithm

#### Algorithm: `save_results_to_json()`

**Purpose**: Export DTW results to JSON format for debugging and analysis.

**Input Parameters**:
- `timestamps`: List of `SentenceTimestamp` objects
- `metadata_list`: List of dictionaries with `norm_cost` and `matched_rms`

**JSON Structure**:
```json
{
    "audio_file": "filename.wav",
    "timestamp": "2025-11-09T10:30:45",
    "parameters": {
        "sr": 16000,
        "hop_length": 512,
        "n_mfcc": 13,
        "delta_order": 2,
        "smoothing_sigma": 0.5,
        "min_duration_frac": 0.7,
        "max_duration_frac": 2.3,
        "min_rms": 0.0001,
        "tts_tld": "com.au",
        "voice_model": "piper_male"
    },
    "sentences": [
        {
            "text": "The cat sat on the mat",
            "start": 1.234,
            "end": 3.567,
            "norm_cost": 0.245,
            "matched_rms": 0.023
        },
        {
            "text": "Yesterday was rainy",
            "start": null,
            "end": null,
            "norm_cost": null,
            "matched_rms": null
        }
    ]
}
```

**Process**:
```
// Construct results dictionary
results = {
    "audio_file": basename(audio_path),
    "timestamp": current_datetime_iso(),
    "parameters": extract_all_parameters(),
    "sentences": []
}

FOR each timestamp, metadata in zip(timestamps, metadata_list):
    sentence_data = {
        "text": timestamp.sentence,
        "start": float(timestamp.start) IF not None ELSE null,
        "end": float(timestamp.end) IF not None ELSE null,
        "norm_cost": float(metadata["norm_cost"]) IF not None ELSE null,
        "matched_rms": float(metadata["matched_rms"]) IF not None ELSE null
    }
    results["sentences"].append(sentence_data)

// Write to file
json_path = audio_dir / f"dtw_results_{audio_basename}.json"
write_json(json_path, results, indent=2, ensure_ascii=False)
```

**Output File Naming**:
- Pattern: `dtw_results_{audio_filename}.json`
- Example: `dtw_results_Participant10.json`
- Location: Same directory as audio file

**Metadata Fields**:
- **norm_cost**: Lower is better (typical range: 0.1-0.5)
- **matched_rms**: Energy level (typical range: 0.01-0.1)
- Both are `null` for failed matches

**Use Cases**:
- Parameter tuning and optimization
- Match quality assessment
- Failure pattern analysis
- Reproducibility documentation

---

### 8. TextGrid Export Algorithm

#### Algorithm: `save_sentences_as_textgrid()`

**Purpose**: Export sentence timestamps to Praat TextGrid format for visualization and analysis.

**Input Parameters**:
- `sentence_timestamps`: List of `SentenceTimestamp` objects

**Process**:

1. **Validation**:
   ```
   IF sentence_timestamps is empty:
       RAISE ValueError("Timestamps list is empty")
   
   IF audio_path does not exist:
       RAISE ValueError("No valid audio path")
   
   IF textgrid_path is not set:
       RAISE ValueError("No valid TextGrid path")
   ```

2. **Entry Construction**:
   ```
   tg_entries = []
   
   FOR each timestamp in sentence_timestamps:
       IF timestamp.start is not None AND 
          timestamp.end is not None AND 
          timestamp.end > timestamp.start:
           
           tg_entries.append((
               timestamp.start,
               timestamp.end,
               timestamp.sentence
           ))
   
   IF tg_entries is empty:
       RAISE ValueError("No valid timestamps found")
   ```

3. **TextGrid Creation**:
   ```
   audio_duration = get_audio_duration(audio_path)
   
   tier = IntervalTier(
       name="sentences",
       entries=tg_entries,
       minT=0,
       maxT=audio_duration
   )
   
   textgrid = Textgrid()
   textgrid.addTier(tier)
   ```

4. **UTF-8 Encoding Handling**:
   ```
   TRY:
       textgrid.save(
           textgrid_path,
           format="long_textgrid",
           includeBlankSpaces=True
       )
       PRINT("TextGrid saved successfully")
   
   CATCH UnicodeEncodeError:
       PRINT("Warning: Unicode encoding issue - fixing...")
       
       // Sanitize labels
       FOR tier in textgrid.tierList:
           FOR interval in tier.intervalList:
               IF interval.label:
                   interval.label = encode_utf8_safe(interval.label)
       
       // Retry save
       textgrid.save(
           textgrid_path,
           format="long_textgrid",
           includeBlankSpaces=True
       )
       PRINT("TextGrid saved successfully after fixing encoding")
   ```

**TextGrid Format**:

The output follows Praat's long TextGrid format:
```
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = <audio_duration>
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "sentences"
        xmin = 0
        xmax = <audio_duration>
        intervals: size = <n>
        intervals [1]:
            xmin = <start1>
            xmax = <end1>
            text = "<sentence1>"
        intervals [2]:
            xmin = <start2>
            xmax = <end2>
            text = "<sentence2>"
        ...
```

**Output**: TextGrid file saved to disk at specified path

**Features**:
- Single interval tier named "sentences"
- Includes blank spaces between intervals
- Full audio duration coverage
- UTF-8 encoding with fallback handling

**UTF-8 Safety**:
- Primary save attempt with UTF-8
- Fallback: manual UTF-8 encoding with error replacement
- Ensures compatibility with non-ASCII characters (e.g., IPA symbols)

---

## Utility Functions

### set_asr_target()

**Purpose**: Configure paths for audio, stimulus, and output files.

**Input Parameters**:
- `audio_path` (str): Path to WAV audio file
- `stimulus_path` (str, keyword-only): Path to stimulus text file
- `textgrid_path` (str | None, keyword-only): Optional TextGrid output path

**Validation**:
- Audio must exist and be .wav format
- Stimulus file must exist
- TextGrid path auto-generated if not provided (replaces .wav with .TextGrid)

**Example**:
```python
dtw = SubsequenceDTW()
dtw.set_asr_target(
    audio_path="participant_01.wav",
    stimulus_path="sentences.txt",
    textgrid_path="participant_01.TextGrid"  # Optional
)
```

---

### get_stimulus_sentences()

**Purpose**: Load expected sentences from stimulus file.

**Format**: 
- Plain text file
- One sentence per line
- Empty lines ignored
- UTF-8 encoding

**Returns**: List of sentence strings

**Error Handling**: 
- ValueError if file empty
- ValueError if file doesn't exist

**Example stimulus file**:
```
The cat sat on the mat.
She sells seashells by the seashore.
Peter Piper picked a peck of pickled peppers.
```

---

## Error Handling and Edge Cases

### 1. No Match Found
- Returns `SentenceTimestamp` with `None` timestamps
- Warning logged: "✗ No match found for sentence '{sentence}'"
- Allows pipeline to continue with remaining sentences
- Recorded in JSON output with null values

### 2. Short Audio (M < N)
- Falls back to full DTW instead of subsequence DTW
- Enforces single starting point
- Still finds best ending point
- Useful when audio is shorter than expected

### 3. Overlapping Timestamps
- Can be resolved using `remove_timestamp_overlaps()`
- Currently commented out in main pipeline
- Available for post-processing if needed

### 4. Low Energy Matches
- Warning: "⚠️ Low energy in matched region"
- Indicates possible false positive
- Match still returned but flagged
- Can be filtered using `matched_rms` threshold

### 5. Empty or Invalid Segments
- Early validation prevents processing empty data
- Clear error messages guide users
- Invalid timestamps preserved in output

### 6. Unicode Encoding Issues
- Automatic detection and fallback handling
- Ensures TextGrid compatibility
- Preserves special characters where possible

---

## Parameters and Tuning Guidelines

### MFCC Parameters

**n_mfcc** (default=13):
- **Lower (8-10)**: Faster, less detailed
- **Default (13)**: Good balance
- **Higher (20-26)**: More detail, slower

**delta_order** (default=2):
- **0**: Static features only (fast, less robust)
- **1**: + velocity
- **2**: + acceleration

### DTW Parameters

**smoothing_sigma** (default=0.5):
- **Lower (0.1-0.3)**: More sensitive to local minima
- **Default (0.5)**: Good balance
- **Higher (1.0-2.0)**: More robust but may miss short sentences

**min_duration_frac** (default=0.7):
- Controls minimum acceptable match duration
- 0.7 = match must be at least 70% of reference duration
- Lower values allow faster speech rates

**max_duration_frac** (default=2.3):
- Controls maximum acceptable match duration
- 2.3 = match can be up to 230% of reference duration
- Higher values accommodate slower speech

**min_rms** (default=1e-4):
- Threshold for silence detection
- Lower values: more sensitive to quiet speech
- Higher values: reject more silence/background noise

## Dependencies and Requirements

### Core Libraries
- **librosa**: Audio processing and MFCC extraction
- **numpy**: Numerical operations and matrix computations
- **scipy**: Distance metrics and signal processing
- **parselmouth**: Praat integration for audio duration
- **praatio**: TextGrid file handling

### TTS Libraries
- **gTTS**: Google Text-to-Speech (requires internet)
- **kittentts**: Local neural TTS (KittenML/kitten-tts-nano-0.2)
- **piper**: High-quality local neural TTS

### Audio I/O
- **soundfile**: WAV file reading/writing
- **wave**: Standard library WAV support

### Utilities
- **helper**: Custom module providing `SentenceTimestamp` class
- **json**: Results logging and debugging

---

## Usage Examples

### Basic Usage
```python
from subsequence_dtw import SubsequenceDTW

# Initialize with default parameters
dtw = SubsequenceDTW()

# Set input files
dtw.set_asr_target(
    audio_path="recording.wav",
    stimulus_path="sentences.txt"
)

# Extract timestamps
timestamps = dtw.get_valid_sentence_timestamps()

# Export to TextGrid
dtw.save_sentences_as_textgrid(timestamps)
```

### Advanced Usage with Custom Parameters
```python
# Initialize with custom parameters for dysarthric speech
dtw = SubsequenceDTW(
    sr=16000,
    hop_length=512,
    n_mfcc=20,                    # More detail
    delta_order=2,                 # Full dynamics
    smoothing_sigma=1.0,           # More smoothing
    min_duration_frac=0.5,         # Allow faster speech
    max_duration_frac=3.0,         # Allow slower speech
    min_rms=5e-5,                  # More sensitive to quiet speech
    tts_tld="com.au",              # Australian accent
    timestamp_tolerance=0.1,       # Wider boundaries
    voice_model="piper_male"       # Default TTS voice model (or set via dtw.set_voice_model(...))
)

# Set target
dtw.set_asr_target(
    audio_path="participant_dysarthric.wav",
    stimulus_path="sentences.txt",
    textgrid_path="output.TextGrid"
)

# Process
timestamps = dtw.get_valid_sentence_timestamps()

# Optional: Remove overlaps
timestamps = dtw.remove_timestamp_overlaps(timestamps, tolerance=0.1)

# Export
dtw.save_sentences_as_textgrid(timestamps)
```

---

## Output Files

### 1. TextGrid File
- **Filename**: `{audio_basename}.TextGrid`
- **Format**: Praat long format
- **Content**: Sentence-level intervals with timestamps
- **Tier Name**: "sentences"

### 2. JSON Debug File
- **Filename**: `dtw_results_{audio_basename}.json`
- **Format**: JSON with indentation
- **Content**: 
  - All parameters used
  - All sentences with timestamps
  - Match quality metrics (norm_cost, matched_rms)
  - Processing metadata

### 3. Temporary Reference Audio
- **Location**: `{audio_dir}/dtw_temp/`
- **Files**: `ref_sentence_{N}.wav` for each sentence
- **Format**: 16kHz mono WAV
- **Purpose**: Debugging and verification

---

## Troubleshooting

### Common Issues

**Problem**: No matches found for any sentences
- **Solution**: Check TTS accent matches speaker accent
- **Solution**: Reduce `min_rms` threshold
- **Solution**: Increase `max_duration_frac` for slow speech

**Problem**: Matches have wrong boundaries
- **Solution**: Increase `timestamp_tolerance`
- **Solution**: Adjust `smoothing_sigma`

**Problem**: Too many false positives
- **Solution**: Increase `min_rms` threshold
- **Solution**: Tighten duration constraints
- **Solution**: Check silence masking is working

**Problem**: Processing too slow
- **Solution**: Reduce `n_mfcc` (e.g., to 10)
- **Solution**: Set `delta_order=1` instead of 2
- **Solution**: Increase `hop_length` (e.g., to 1024)

**Problem**: TextGrid encoding errors
- **Solution**: Automatic fallback should handle this
- **Solution**: Check for unusual characters in stimulus file
- **Solution**: Ensure UTF-8 encoding of stimulus file

---
