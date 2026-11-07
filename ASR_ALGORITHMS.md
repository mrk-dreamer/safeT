# WhisperX ASR Algorithm Documentation

## Overview

The `asr.py` module implements an automatic speech recognition (ASR) system using WhisperX for sentence segmentation in speech recordings. The system combines deep learning-based transcription with advanced text matching algorithms to accurately identify and timestamp sentences within audio files.

## Core Components

### WhisperX_ASR Class

The main class that orchestrates the ASR pipeline, providing sentence-level segmentation with word-aligned timestamps.

---

## Algorithms

### 1. Initialization and Model Setup

#### Algorithm: `__init__()`

**Purpose**: Initialize the WhisperX ASR system with appropriate models and device configuration.

**Input Parameters**:
- `whisper_arch` (str, default="tiny"): The Whisper model architecture to use
  - Options: "tiny", "base", "small", "medium", "large"
  - Larger models provide better accuracy at the cost of speed and memory
- `language` (str, default="en"): Target language for transcription
- `timestamp_tolerance` (float, default=0.05): Tolerance in seconds for timestamp adjustments

**Process**:
1. **Device Detection**: Automatically detects CUDA GPU availability
   - If GPU available: Uses CUDA with float16 precision
   - If CPU only: Uses CPU with int8 quantization for efficiency

2. **Model Loading**:
   - Loads WhisperX transcription model with specified architecture
   - Loads language-specific alignment model for word-level timestamps
   - Stores alignment metadata for subsequent processing

3. **Configuration**:
   - Sets timestamp tolerance for boundary adjustments
   - Defines chunk size choices [20, 23, 26, 29] for audio processing (this is to be used if we want Whisper to use a random chunk size, not tested)

**Output**: Initialized WhisperX_ASR object with loaded models

---

### 2. Text Matching Metrics

#### Algorithm: `calculate_wer()`

**Purpose**: Compute Word Error Rate (WER) between hypothesis and reference text.

**Input Parameters**:
- `hypothesis`: Transcribed text from ASR
- `reference`: Ground truth text

**Mathematical Formulation**:

$$\text{WER} = \frac{S + D + I}{N}$$

Where:
- $S$ = Number of substitutions
- $D$ = Number of deletions
- $I$ = Number of insertions
- $N$ = Total number of words in reference

**Output**: Float value representing WER (0.0 = perfect match, higher = more errors)

---

#### Algorithm: `fuzzy_match_score()`

**Purpose**: Compute similarity score using fuzzy string matching.

**Input Parameters**:
- `hypothesis`: Transcribed text
- `reference`: Target text

**Mathematical Formulation**:

Uses RapidFuzz library implementing Levenshtein distance at character level:

$$\text{Similarity} = 1 - \frac{\text{Levenshtein}(\text{hypothesis}, \text{reference})}{\max(|\text{hypothesis}|, |\text{reference}|)}$$

**Process**:
1. Calculate character-level Levenshtein distance
2. Normalize to range [0, 100]
3. Scale to [0, 1] range

**Output**: Float in [0, 1] where 1.0 = perfect match

---

### 3. Word Shaving Algorithm

#### Algorithm: `refine_match_using_shaving()`

**Purpose**: Iteratively remove words from segment boundaries to improve match quality with target text.

**Input Parameters**:
- `segment`: Dictionary containing transcribed words with timestamps
- `target_text`: Reference sentence to match
- `wer_threshold` (default=0.6): Maximum acceptable WER
- `similarity_threshold` (default=0.6): Minimum acceptable similarity score

**Process**:

1. **Initialization**:
   ```
   Extract words from segment
   Strip punctuation (commas, periods)
   Initialize best_match = (0, len(words)-1) # indices of starting and ending words in current best match
   Calculate initial scores (WER and fuzzy match)
   ```

2. **Iterative Shaving Loop**:
   ```
   start_idx = best_match[0]
   end_idx = best_match[1]
   improvement_found = True
   best_score = initial_similarity
   best_wer = initial_wer
   WHILE improvement_found AND end_idx > start_idx:
       improvement_found = False
       
       // Try removing first word from current best match
       hypothesis = words[start_idx+1 : end_idx+1]
       score = fuzzy_match_score(hypothesis, target_text)
       wer = calculate_wer(hypothesis, target_text)
       
       IF score > best_score AND wer <= best_wer:
           best_score = score
           best_wer = wer
           start_idx = start_idx + 1
           improvement_found = True
       
       // Try removing last word from same
       IF start_idx < end_idx:
           hypothesis = words[start_idx : end_idx]
           score = fuzzy_match_score(hypothesis, target_text)
           wer = calculate_wer(hypothesis, target_text)
           
           IF score > best_score AND wer <= best_wer:
               best_score = score
               best_wer = wer
               end_idx = end_idx - 1
               improvement_found = True
   ```

3. **Result Construction**:
   ```
   Create shaved_segment with:
       - start: timestamp of starting word: words[start_idx]
       - end: timestamp of ending word: words[end_idx]
       - text: concatenated words[start_idx : end_idx+1]
       - words: dictionary of words and their timestamps
   ```

**Output**: Tuple containing:
- Start timestamp (float)
- End timestamp (float)
- Best similarity score (float)
- Best WER (float)
- Dictionary of words in shaved segment

**Optimization Strategy**:
- Greedy approach: accepts improvements immediately
- Dual criterion: both fuzzy match must improve AND WER must not worsen
- Bidirectional: tests both front and back removal

**Example**:
```
Input segment: "Well I think the cat sat on the mat yesterday"
Target text: "the cat sat on the mat"

Iteration 1: Remove "Well" -> Improvement ✓, Remove "yesterday" -> Improvement ✓
Iteration 2: Remove "I" -> Improvement ✓, Remove "mat" -> no improvement bur Improvement flag stays positive ✓
Iteration 3: Remove "think" -> Improvement ✓
Final result: "the cat sat on the mat"
```

---

### 4. Sentence Matching Algorithm

#### Algorithm: `find_last_occurrence()`

**Purpose**: Find the best matching occurrence of a target sentence in ASR transcription segments.

**Input Parameters**:
- `segments`: List of transcription segments with word-level timestamps
- `target_text`: Sentence to locate in transcription
- `wer_threshold` (default=0.6): Maximum acceptable WER
- `similarity_threshold` (default=0.6): Minimum acceptable similarity

**Process**:

1. **Candidate Generation**:
   ```
   FOR each segment in segments:
       refined_segment = refine_match_using_shaving(segment, target_text)
       wer = calculate_wer(refined_segment.text, target_text)
       similarity = fuzzy_match_score(refined_segment.text, target_text)
       
       IF wer < wer_threshold OR similarity > similarity_threshold:
           Add (refined_segment, similarity, wer) to list of prospective candidates
   ```

2. **Best Match Selection**:
   ```
   IF no prospective candidates:
       RETURN None
   
   best_segment = argmax over prospective candidates by:
       1. Maximize similarity_score (primary criterion)
       2. Minimize WER (secondary criterion)
       3. Maximize start_time (tertiary - prefer later occurrences)
   ```

3. **Debugging Output**:
   - Saves all prospective candidate segments to JSON file
   - Includes timestamp, similarity scores, and WER for each candidate
   - Appends to existing log file (cumulative logging)

**Output**: Dictionary containing:
```python
{
    "start": float,           # Start timestamp in seconds
    "end": float,             # End timestamp in seconds
    "text": str,              # Matched text
    "similarity": float,      # Similarity score [0,1]
    "wer": float             # Word error rate
}
```

**Selection Criteria**:

The best match is selected using following ordering:
1. **Primary**: Highest similarity score
2. **Secondary**: Lowest WER (if similarity tied)
3. **Tertiary**: Latest occurrence (if both tied)

This prioritization ensures:
- Exact matches preferred over partial matches
- Among similar matches, fewer word errors preferred
- For repeated phrases, later occurrences selected (because the correct utterance of a sentence will be its last occurrence in most recordings)

**Example Scenario**:
```
Target: "the quick brown fox"
Segments:
  1. "the quick brown fox jumped" -> After shaving: "the quick brown fox" (sim=1.0, wer=0.0)
  2. "a quick brown fox runs" -> After shaving: "quick brown fox" (sim=0.85, wer=0.25)
  3. "the quick brown fox sleeps" -> After shaving: "the quick brown fox" (sim=1.0, wer=0.0)

Result: Segment 3 selected (later occurrence with perfect match)
```

---

### 5. Overlap Resolution Algorithm

#### Algorithm: `remove_segment_overlaps()`

**Purpose**: Eliminate temporal overlaps between consecutive segments by adjusting boundaries.

**Input Parameters**:
- `segments`: List of segment dictionaries with 'start', 'end', and 'words' keys
- `tolerance` (default=0.05): Gap tolerance in seconds

**Process**:

1. **Sort by Time**:
   ```
   Sort segments by start time (ascending)
   Initialize result = [first segment]
   ```

2. **Overlap Detection and Resolution**:
   ```
   FOR each current_segment in segments[1:]:
       last_segment = result[-1]
       
       IF current_segment.start > last_segment.end + tolerance:
           // No overlap - add segment as-is
           result.append(current_segment)
       ELSE:
           // Overlap detected - adjust boundaries
           midpoint = (last_segment.end + current_segment.start) / 2
           
           // Update previous segment
           last_segment.end = midpoint - (tolerance / 2)
           last_segment.words[-1].end = midpoint - (tolerance / 2)
           
           // Update current segment
           current_segment.start = midpoint + (tolerance / 2)
           current_segment.words[0].start = midpoint + (tolerance / 2)
           
           result.append(current_segment)
   ```

**Mathematical Formulation**:

Given two consecutive segments $S_i$ and $S_{i+1}$:

**No Overlap Condition**:
$$t_{\text{start}}^{i+1} > t_{\text{end}}^{i} + \tau$$

Where $\tau$ is the tolerance parameter.

**Overlap Resolution**:
$$t_{\text{midpoint}} = \frac{t_{\text{end}}^{i} + t_{\text{start}}^{i+1}}{2}$$

$$t_{\text{end}}^{i} \leftarrow t_{\text{midpoint}} - \frac{\tau}{2}$$

$$t_{\text{start}}^{i+1} \leftarrow t_{\text{midpoint}} + \frac{\tau}{2}$$

**Output**: List of non-overlapping segments with adjusted timestamps

**Properties**:
- Preserves temporal ordering
- Creates minimum gap of `tolerance` between segments
- Splits overlap evenly between adjacent segments
- Updates both segment-level and word-level timestamps

**Example**:
```
Input (tolerance = 0.05):
  Segment A: [1.0, 3.2]
  Segment B: [3.0, 5.5]  // Overlaps with A by 0.2s
  Segment C: [6.0, 8.0]  // No overlap

Process:
  A and B overlap (3.0 < 3.2 + 0.05)
  midpoint = (3.2 + 3.0) / 2 = 3.1
  A.end = 3.1 - 0.025 = 3.075
  B.start = 3.1 + 0.025 = 3.125
  
Output:
  Segment A: [1.0, 3.075]
  Segment B: [3.125, 5.5]
  Segment C: [6.0, 8.0]
```

---

### 6. Main Pipeline Algorithm

#### Algorithm: `get_valid_sentence_timestamps()`

**Purpose**: Complete end-to-end pipeline for extracting sentence timestamps from audio.

**Input Parameters**:
- `shuffle_chunk_size` (bool, default=False): Whether to randomize chunk size

**Process Flow**:

1. **Validation**:
   ```
   Verify audio_path exists and is valid
   Load stimulus sentences from file
   ```

2. **Chunk Size Selection**:
   ```
   IF shuffle_chunk_size:
       chunk_size = random.choice([20, 23, 26, 29])
   ELSE:
       chunk_size = 29  // Default optimal value
   ```

3. **Transcription**:
   ```
   raw_segments = whisperx.transcribe(
       audio_path,
       language="en",
       task="transcribe",
       chunk_size=chunk_size
   )
   ```

4. **Word Alignment**:
   ```
   aligned_segments = whisperx.align(
       raw_segments,
       align_model,
       metadata,
       audio_path
   )
   ```

5. **Overlap Resolution**:
   ```
   clean_segments = remove_segment_overlaps(
       aligned_segments,
       tolerance=timestamp_tolerance
   )
   ```

6. **Sentence Matching**:
   ```
   valid_timestamps = []
   FOR each sentence in stimulus_sentences:
       match = find_last_occurrence(clean_segments, sentence)
       IF match exists:
           valid_timestamps.append(
               SentenceTimestamp(
                   sentence=sentence,
                   start=match['start'],
                   end=match['end']
               )
           )
       ELSE:
           valid_timestamps.append(
               SentenceTimestamp(
                   sentence=sentence,
                   start=None,
                   end=None
               )
           )
   ```

7. **Debug Logging**:
   ```
   Save aligned_segments to JSON file with metadata:
       - audio_file name
       - timestamp
       - chunk_size used
       - all segments with word alignments
   ```

**Output**: List of `SentenceTimestamp` objects containing:
- Sentence text
- Start time (or None if not found)
- End time (or None if not found)

**Chunk Size Impact**:

The `chunk_size` parameter affects where the audio is split for processing, which in turn affects where transcription generally misses words.

**Pipeline Stages Visualization**:

```
Audio Input
    ↓
[WhisperX Transcription] → Raw segments with approximate timestamps
    ↓
[Word-Level Alignment] → Precise word boundaries
    ↓
[Overlap Resolution] → Clean, non-overlapping segments
    ↓
[Sentence Matching] → Match to stimulus sentences
    ↓
[Timestamp Extraction] → Final sentence boundaries
    ↓
Output: SentenceTimestamp list
```

**Error Handling**:
- Invalid audio paths → ValueError
- Non-WAV files → ValueError
- No stimulus sentences → ValueError
- Unmatched sentences → Warning + None timestamps

---

### 7. TextGrid Export Algorithm

#### Algorithm: `save_sentences_as_textgrid()`

**Purpose**: Export sentence timestamps to Praat TextGrid format.

**Input Parameters**:
- `sentence_timestamps`: List of `SentenceTimestamp` objects

**Process**:

1. **Validation**:
   ```
   Verify sentence_timestamps is not empty
   Verify audio_path exists
   Verify textgrid_path is set
   ```

2. **Entry Construction**:
   ```
   tg_entries = []
   FOR each timestamp in sentence_timestamps:
       IF timestamp.start AND timestamp.end AND timestamp.end > timestamp.start:
           tg_entries.append((timestamp.start, timestamp.end, timestamp.sentence))
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
       textgrid.save(path, format="long_textgrid")
   CATCH UnicodeEncodeError:
       FOR each tier in textgrid:
           FOR each interval in tier:
               interval.label = encode_utf8_safe(interval.label)
       textgrid.save(path, format="long_textgrid")
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
        ...
```

**UTF-8 Safety**:
- Primary save attempt with UTF-8
- Fallback: manual UTF-8 encoding with error replacement
- Ensures compatibility with non-ASCII characters

**Output**: TextGrid file saved to disk at specified path

---

## Utility Functions

### set_asr_target()

**Purpose**: Configure paths for audio, stimulus, and output files.

**Validation**:
- Audio must exist and be .wav format
- Stimulus file must exist
- TextGrid path auto-generated if not provided

---

### get_stimulus_sentences()

**Purpose**: Load expected sentences from stimulus file.

**Format**: Plain text file, one stimulus sentence per line

**Error Handling**: ValueError if file empty or invalid

---

## Error Handling and Edge Cases

### 1. No Match Found
- Returns `SentenceTimestamp` with `None` timestamps
- Warning logged for user awareness
- Allows pipeline to continue with remaining sentences

### 2. Overlapping Segments
- Automatically resolved using midpoint algorithm

### 3. Empty Segments
- Early validation prevents processing empty data
- Clear error messages guide users

---

## Dependencies and Requirements

### Core Libraries
- **whisperx**: ASR transcription and alignment
- **torch**: Deep learning backend
- **parselmouth**: Praat integration for audio
- **praatio**: TextGrid file handling
- **jiwer**: WER calculation
- **rapidfuzz**: Fuzzy string matching
- **numpy**: Numerical operations

---

## Usage Examples

### Basic Usage
```python
# Initialize ASR system
asr = WhisperX_ASR(whisper_arch="base", language="en", timestamp_tolerance=0.05)

# Set input files
asr.set_asr_target(
    audio_path="recording.wav",
    stimulus_path="sentences.txt"
)

# Extract timestamps
timestamps = asr.get_valid_sentence_timestamps()

# Export to TextGrid
asr.save_sentences_as_textgrid(timestamps)
```

### Advanced Usage with Randomization
```python
timestamps = asr.get_valid_sentence_timestamps(shuffle_chunk_size=True)
```
Note: Not tested yet.
---

## Future Enhancements

### Potential Improvements
1. **Multi-language Support**: Extend beyond English
2. **Parallel Processing**: Process multiple files simultaneously
3. **Confidence Scoring**: Return confidence metrics with timestamps

---

## References

### Academic Foundations
1. **Whisper**: Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision", 2022
2. **WhisperX**: Bain et al., "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio", 2023
3. **WER Metric**: Standard speech recognition evaluation metric
4. **Levenshtein Distance**: String similarity algorithm for fuzzy matching