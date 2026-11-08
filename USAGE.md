# Steps to get formant targets from audio files via SAFE-T GUI

## Loading audio files
1. Launch the application by double-clicking `SAFE-T.exe` (if using the standalone package/installer) or by running `python player_pyqt6.py` (if running from source). The executable should be left placed in the `SAFE-T` directory, after unpacking `SAFE-T.zip`. `player_pyqt6.py` script should also be in `SAFE-T` repository when running from source. The setup instructions for whichever application mode you choose, provided in `README.md`, should be carried out before launching the application.
2. Use the `Open Audio` button under `File` at top left of application window to select and load any audio file for analysis. Only `.wav` files are supported. Use the sample audio file inside `Test_data` directory for testing.
3. Upon successful loading, the waveform of the audio file will be displayed in the 'Audio Waveform' section. The spectrogram of audio will be displayed in the 'Spectrogram' section, with formant readings by Praat shown as red dots. The audio file location will be shown in the 'Audio Path' field in box below waveform plot.
4. You can play the audio using the Play/Pause, Play Window, and Stop buttons below the waveform plot.

## Analyzing audio files

### Segment sentences
1. Click the `Load Stimulus` button below waveform plot to load the stimulus text file corresponding to the audio file. The stimulus text file should be a plain text file with one stimulus sentence per line. The text file at `Test_data/Stimulus_sentences.txt` can be used as an example. After choosing the stimulus file, the path to this file will be shown in the 'Stimulus Path' field in box below waveform plot.
2. Click the `Run ASR` button below waveform plot and choose one of the methods for sentence segmentation:
   - `Whisper AI`: Uses WhisperX automatic speech recognition and post-processing for sentence segmentation
   - `Subsequence DTW`: Hover over `Subsequence DTW` to open its sub-dropdown and choose the TTS reference voice model (`Piper TTS - Male (Alan)` [default], `Piper TTS - Female (Alba)`, `gTTS (Google TTS)`, or any of the `KittenTTS` voices) for Subsequence Dynamic Time Warping sentence segmentation
3. After processing, the waveform and spectrogram plots will update to show sentence boundaries. The segmented sentences will be listed in the same box where 'Audio Path' is shown (scroll down inside box, if not visible). And the path to the generated TextGrid file will be shown in the 'Sentences TextGrid Path' field in box below waveform plot. A separate plot for sentence boundaries with sentence labels, `Sentence Labels`, will be visible below the box. You can play whole audio or only the window by using the Play/Pause, Play Window, and Stop buttons below the waveform plot, to verify the sentence boundaries
4. (Optional) Adjust sentence boundaries manually by dragging the vertical lines on the waveform or spectrogram plots, or on plot `Sentence Labels`. You can zoom in/out using the mouse scroll wheel for finer adjustments. You can delete a sentence by clicking on `Delete` button beside its label in the sentences list(in the box where 'Audio Path' is shown. Scroll inside the box if not visible). You can also add a new sentence through the text input fields below this box. Input the start time, end time, and text of the new sentence, then click the `Add Sentence` button to add it to the list and update the plots.

### Align phonemes/Segment vowels
1. Click the `Run MFA` button to perform phoneme alignment using Montreal Forced Aligner (MFA).
2. After processing, the waveform and spectrogram plots will update to show phoneme boundaries. A separate plot for phoneme boundaries with phoneme labels, `Phoneme Labels`, will be visible below the spectrogram. The path to a temporary directory created by SAFE-T (directory named PyPraat_sentences by default), containing separate audio files for each sentence and their corresponding text transcriptions and TextGrid files for aligned phonemes of each sentence, will be shown in the 'Sentences Path' field in box below waveform plot.
3. (Optional) Adjust phoneme boundaries manually by dragging the vertical lines on the waveform or spectrogram plots or on plot `Phoneme Labels`. You can zoom in/out using the mouse scroll wheel for finer adjustments. You can add phonemes by using the text input fields below the plot `Phoneme Labels`. Input the start time, end time, phoneme label, parent word and parent sentence of the new phoneme, then click the `Add Phoneme` button to add it to the list and update the plots.

### Extract formant targets
1. Click the `Load Landmark Info` button to extract formant targets for each vowel segment. First choose a cleaning method:
   - `with Viterbi cleaning`: Hover over `with Viterbi cleaning` to open its sub-dropdown and choose `with Ground Truth` (to include emission cost from reference formant targets) or `without Ground Truth`.
   - `with NO cleaning`: Uses raw Praat formant readings without cleaning.
   Then a file picker will pop up to choose the landmark info file. Choose the CSV file `landmark_identification_mod.csv` (or your own formant target identification rules with similar CSV file structure and vocabulary). If you are using `SAFE-T.exe` offered in `SAFE-T.zip` package, the landmark info file is not included and must be taken from this repository and placed in the same directory as the executable.
2. After processing, the path to landmark info file will be shown in the 'Landmark Info Path' field in box below waveform plot. Spectrogram will show the formant readings used for target extraction as colored dots, overlayed on top of raw Praat formant readings (red dots). And target positions will be shown with pink vertical lines on spectrogram. The targets will be plotted in vowel plots at bottom. Boundaries of vowels and timestamps of their targets will be shown in the box below the plot with correspending phoneme label and phoneme boundaries.
3. (Optional) You can adjust target positions by dragging the pink vertical lines on the spectrogram plot. You can zoom in/out using the mouse scroll wheel for finer adjustments. You can also delete a vowel and its targets by clicking on Delete button beside its label in the box below plot `Phoneme Labels`.
4. There is a scrollable list above the vowel plots, showing all stimulus sentences, their words of interest, and vowels, within those words that are plotted in the vowel plots. You can click on any vowel in the list to toggle its visibility in the vowel plots. You choose which formant to plot in which axis of the vowel plots using the dropdown menus beside `X Axis (Formant):` and `Y Axis (Formant):`. You can right-click on vowel plots to get more options like flipping the axes.
5. The 'Gender:' toggle button changes the reference vowel plot (in red). When toggled, the values in `landmark_identification_ground_truth.csv`, are used as reference. If gender is 'F' female gender data is used, otherwise male. Only female data is provided for now, from the paper: Cox, Felicity. (2006). The Acoustic Characteristics of /hVd/ Vowels in the Speech of some Australian Teenagers, Australian Journal of Linguistics. 26. 10.1080/07268600600885494. The values in the CSV file can be changed as needed.
6. Click the `Export Formant Targets` button below waveform plot to save formant targets' data as visible in spectrogram and vowel plots. The formant targets for each vowel segment will be saved in a CSV file named `formant_targets.csv` in the same directory as the main project.
7. Click the export buttons under each vowel space plot to export the plot data as CSV, PNG and SVG.

## Additional settings

### Adjusting settings of Viterbi-based formant cleaning
You can activate or deactivate emission cost in Viterbi-based formant cleaning (penalty for deviating from reference formant targets) directly from the GUI under `Load Landmark Info -> with Viterbi cleaning -> with Ground Truth` or `without Ground Truth`.

Once activated, the reference formant targets from `landmark_identification_ground_truth.csv` will be used during Viterbi cleaning to compute emission costs.

To set scaling factor for emission cost, modify the constants `VITERBI_EMISSION_MONOPHTHONG_WEIGHT` and `VITERBI_EMISSION_DIPHTHONG_WEIGHT` in `formants.py` to scale emission cost for monophthongs and diphthongs respectively. This factor scales the influence of emission cost relative to transition cost during Viterbi-based cleaning.

Change the value of `VITERBI_DELTA` in `formants.py` to adjust the threshold for Huber loss in transition cost, changing the transition cost function from quadratic to linear below this threshold.

### Adjusting Praat formant extraction settings
To adjust the Praat formant extraction settings (maximum formant frequency, number of formants, window length, time step, preemphasis base), open `formants.py` and modify the following constants as needed:
```
PRAAT_FORMANT_TIME_STEP = 0.0015 # in seconds
PRAAT_N_FORMANTS = 5
PRAAT_FORMANT_MAX_FREQ = 5500 # in Hz
PRAAT_FORMANT_WINDOW_LENGTH = 0.025 # in seconds
PRAAT_FORMANT_PREEMPHASIS_FROM = 50 # in Hz
```

### Adjusting log output location
To change the location where log files are saved, open `formants.py` and modify the constant `DEBUG_LOG_FILE` to the desired directory path:
```python
DEBUG_LOG_FILE = r"C:\path\to\desired\log\directory"
```
To change the location of log output for formant cleaning process, open `formants.py` and modify the constant `OUTLIER_FIX_LOG_FILE` to the desired directory path:
```python
OUTLIER_FIX_LOG_FILE = r"C:\path\to\desired\log\directory"
```
Make sure you have write access in these locations.

### Adjusting Subsequence DTW settings
You can select the text-to-speech (TTS) voice model directly from the GUI under `Run ASR -> Subsequence DTW` (`Piper TTS - Male (Alan)`, `Piper TTS - Female (Alba)`, `gTTS (Google TTS)`, or `KittenTTS -> expr-voice-*`).

To customize voice synthesis settings in code, see `set_voice_model` and `generate_reference_audio` in `subsequence_dtw.py`:

For piper text-to-speech engine, `*.onnx` model and `*.onnx.json` file corresponding to a voice model must be downloaded from Piper model repository and kept in project root directory (already present in this repository). See Piper model repository here:
`https://rhasspy.github.io/piper-samples/`. If using `SAFE-T.exe`, the unpacked `SAFE-T` directory already includes these files.

## Note

After any changes in code and/or dependencies, main program script `player_pyqt6.py` needs to be relaunched and SAFE-T executable needs to be rebuilt to reflect changes.

The build scripts and build verification scripts might also need updates after changes in code and/or dependencies.
