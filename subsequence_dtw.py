# -*- coding: utf-8 -*-
import os
import numpy as np
import librosa
import parselmouth
from praatio import textgrid
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
import soundfile as sf
from gtts import gTTS
from kittentts import KittenTTS
from piper import SynthesisConfig, PiperVoice
import wave
from helper import SentenceTimestamp
import json


class SubsequenceDTW:
    def __init__(
        self,
        *,
        sr=16000,
        hop_length=512,
        n_mfcc=13,
        delta_order=2,
        smoothing_sigma=0.5,
        min_duration_frac=0.7,
        max_duration_frac=2.3,
        min_rms=1e-4,
        tts_tld="com.au",
        timestamp_tolerance=0.05,
        voice_model: str = "piper_male",
    ):
        """
        Initialize SubsequenceDTW for sentence detection using DTW alignment.

        Args:
            sr: Sample rate for audio processing
            hop_length: Temporal resolution in samples
            n_mfcc: Number of MFCC coefficients
            delta_order: Order of delta features (0, 1, or 2)
            smoothing_sigma: Gaussian smoothing for cost curve
            min_duration_frac: Minimum matched duration fraction (relative to reference)
            max_duration_frac: Maximum matched duration fraction (relative to reference)
            min_rms: Minimum RMS energy threshold
            tts_tld: TTS accent/locale (e.g., "com.au" for Australian)
            timestamp_tolerance: Tolerance for timestamp boundaries
        """
        self.sr = sr
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.delta_order = delta_order
        self.smoothing_sigma = smoothing_sigma
        self.min_duration_frac = min_duration_frac
        self.max_duration_frac = max_duration_frac
        self.min_rms = min_rms
        self.tts_tld = tts_tld
        self.timestamp_tolerance = timestamp_tolerance
        self.voice_model = voice_model
        self.kitten_tts_model = KittenTTS("KittenML/kitten-tts-nano-0.2")
        self.piper_tts_model_female = PiperVoice.load("en_GB-alba-medium.onnx")
        self.piper_tts_model_male = PiperVoice.load("en_GB-alan-medium.onnx")
        self.piper_syn_config = SynthesisConfig(
            volume=0.5,  # half as loud
            length_scale=1.3,  # 2.0 means twice as slow
            noise_scale=1.0,  # more audio variation
            noise_w_scale=1.0,  # more speaking variation
            normalize_audio=False,  # use raw audio from voice
        )

        # Paths (to be set via set_asr_target)
        self.audio_path = None
        self.stimulus_path = None
        self.textgrid_path = None

    def set_asr_target(
        self, audio_path: str, *, stimulus_path: str, textgrid_path: str | None = None
    ):
        """
        Set the target audio and stimulus files for processing.

        Args:
            audio_path: Path to the WAV audio file
            stimulus_path: Path to the stimulus text file
            textgrid_path: Optional path for TextGrid output (auto-generated if not provided)
        """
        if not audio_path or not os.path.exists(audio_path):
            raise ValueError("Error: No valid audio path provided.")
        if not audio_path.endswith(".wav"):
            raise ValueError("Error: Provided audio is not WAV file.")
        self.audio_path = audio_path

        if not stimulus_path or not os.path.exists(stimulus_path):
            raise ValueError("Error: No valid stimulus path provided.")
        self.stimulus_path = stimulus_path

        if not textgrid_path:
            textgrid_path = audio_path.replace(".wav", ".TextGrid")
        self.textgrid_path = textgrid_path

    def get_stimulus_sentences(self):
        """
        Read stimulus sentences from the stimulus file.

        Returns:
            List of sentence strings
        """
        if not self.stimulus_path or not os.path.exists(self.stimulus_path):
            raise ValueError("Error: No valid stimulus path provided.")

        stimulus_sentences = []
        with open(self.stimulus_path, "r", encoding="utf-8") as f:
            for line in f:
                sentence = line.strip()
                if sentence:  # Skip empty lines
                    stimulus_sentences.append(sentence)

        if len(stimulus_sentences) == 0:
            raise ValueError("Error: No valid sentences found in the stimulus file.")

        return stimulus_sentences

    def compute_mfcc_feat(
        self, y, sr=None, n_mfcc=None, hop_length=None, delta_order=None
    ):
        """
        Compute MFCC features with delta features.

        Args:
            y: Audio time series
            sr: Sample rate (uses self.sr if None)
            n_mfcc: Number of MFCCs (uses self.n_mfcc if None)
            hop_length: Hop length (uses self.hop_length if None)
            delta_order: Delta order (uses self.delta_order if None)

        Returns:
            Feature matrix with shape (n_features, n_frames)
        """
        sr = sr or self.sr
        n_mfcc = n_mfcc or self.n_mfcc
        hop_length = hop_length or self.hop_length
        delta_order = delta_order if delta_order is not None else self.delta_order

        mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        feats = [mf]

        if delta_order >= 1:
            feats.append(librosa.feature.delta(mf, order=1))
        if delta_order >= 2:
            feats.append(librosa.feature.delta(mf, order=2))

        feat = np.vstack(feats)
        return feat

    def cmvn(self, X, eps=1e-8):
        """
        Apply Cepstral Mean and Variance Normalization.

        Args:
            X: Feature matrix
            eps: Small constant for numerical stability

        Returns:
            Normalized feature matrix
        """
        mu = np.mean(X, axis=1, keepdims=True)
        sd = np.std(X, axis=1, keepdims=True)
        return (X - mu) / (sd + eps)

    def set_voice_model(self, voice_model: str = "piper_male"):
        """
        Set the TTS voice model used to generate reference audio.

        Supported values:
            - "piper_male": Piper TTS Male (en_GB-alan-medium.onnx) [default]
            - "piper_female": Piper TTS Female (en_GB-alba-medium.onnx)
            - "gtts": Google TTS (Australian English via self.tts_tld)
            - "kitten:<voice>" (e.g. "kitten:expr-voice-2-f") or "kitten_female" / "kitten_male"
        """
        self.voice_model = voice_model

    def generate_reference_audio(
        self,
        sentence: str,
        output_path: str,
        voice_model: str | None = None,
    ):
        """
        Generate reference audio from text using the selected TTS voice model.

        Args:
            sentence: Text to synthesize
            output_path: Path to save the generated audio
            voice_model: Optional override for the TTS voice model (defaults to self.voice_model or "piper_male")

        Returns:
            Audio array resampled to self.sr
        """
        model = (voice_model or self.voice_model or "piper_male").lower()

        if model == "gtts":
            tts = gTTS(text=sentence, lang="en", tld=self.tts_tld, slow=False)
            tts.save(output_path)
            ref_audio, _ = librosa.load(output_path, sr=self.sr)
            return ref_audio

        if model in ("piper_female", "piper_alba"):
            voice = self.piper_tts_model_female
            with wave.open(output_path, "wb") as wav_file:
                voice.synthesize_wav(sentence, wav_file, self.piper_syn_config)
            ref_audio, _ = librosa.load(output_path, sr=self.sr)
            return ref_audio

        if model in ("piper_male", "piper_alan"):
            voice = self.piper_tts_model_male
            with wave.open(output_path, "wb") as wav_file:
                voice.synthesize_wav(sentence, wav_file, self.piper_syn_config)
            ref_audio, _ = librosa.load(output_path, sr=self.sr)
            return ref_audio

        if model.startswith("kitten"):
            # available_voices: ['expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',
            #                    'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f']
            if ":" in model:
                kitten_voice = model.split(":", 1)[1]
            elif model.endswith("male") and not model.endswith("female"):
                kitten_voice = "expr-voice-2-m"
            else:
                kitten_voice = "expr-voice-2-f"
            audio = self.kitten_tts_model.generate(
                sentence, voice=kitten_voice, speed=1.2
            )
            sf.write(output_path, audio, 24000)
            ref_audio, _ = librosa.load(output_path, sr=self.sr)
            return ref_audio

        raise ValueError(f"Unknown Subsequence DTW voice model: {self.voice_model}")

    def find_sentence_match(self, ref_audio, long_audio):
        """
        Find the best match for reference audio in long audio using improved subsequence DTW.
        """

        # ---------------- Feature extraction ----------------
        ref_feat = self.compute_mfcc_feat(ref_audio)
        long_feat = self.compute_mfcc_feat(long_audio)
        ref_feat = self.cmvn(ref_feat)
        long_feat = self.cmvn(long_feat)

        N = ref_feat.shape[1]
        M = long_feat.shape[1]
        ref_duration = N * self.hop_length / self.sr

        # ---------------- RMS-based silence masking ----------------
        frame_rms = librosa.feature.rms(
            y=long_audio, frame_length=self.hop_length * 2, hop_length=self.hop_length
        )[0]
        silent_mask = frame_rms < self.min_rms

        # ---------------- Cost matrix ----------------
        C = cdist(ref_feat.T, long_feat.T, metric="cosine")
        C[:, silent_mask] = 1.0  # penalize silent frames heavily

        # ---------------- Full DTW for short recordings ----------------
        if M < N:
            print(
                "Long audio shorter than reference. Doing full DTW between ref and long."
            )
            D = np.full((N + 1, M + 1), np.inf)
            D[0, :] = 0.0
            D[0, 0] = 0  # enforce single start
            for i in range(1, N + 1):
                for j in range(1, M + 1):
                    D[i, j] = C[i - 1, j - 1] + min(
                        D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
                    )

            j_end = int(np.argmin(D[N, 1:]) + 1)
            i, j = N, j_end
            path = []
            while i > 0 and j > 0:
                path.append((i - 1, j - 1))
                moves = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]
                costs = [D[a, b] for a, b in moves]
                move = moves[int(np.argmin(costs))]
                i, j = move
            path = path[::-1]
            j_start = path[0][1]
            start_time = j_start * self.hop_length / self.sr
            end_time = j_end * self.hop_length / self.sr
            norm_cost = D[N, j_end] / N  # consistent normalization

        # ---------------- Subsequence DTW ----------------
        else:
            D = np.full((N + 1, M + 1), np.inf)
            D[0, :] = 0.0
            D[0, 0] = 0.0  # start only at frame 0

            for i in range(1, N + 1):
                for j in range(1, M + 1):
                    D[i, j] = C[i - 1, j - 1] + min(
                        D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
                    )

            norm_costs = np.full(M + 1, np.inf)
            starts = np.zeros(M + 1, dtype=int)
            path_lengths = np.zeros(M + 1, dtype=int)
            back_paths = [None] * (M + 1)

            for j_end in range(1, M + 1):
                if not np.isfinite(D[N, j_end]):
                    continue
                i, j = N, j_end
                path = []
                while i > 0 and j > 0:
                    path.append((i - 1, j - 1))
                    moves = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]
                    costs = [D[a, b] for a, b in moves]
                    move = moves[int(np.argmin(costs))]
                    i, j = move
                path = path[::-1]
                if not path:
                    continue
                j_start = path[0][1]
                norm_cost = D[N, j_end] / N  # use constant normalization
                duration = (j_end - j_start) * self.hop_length / self.sr
                dur_penalty = 1.0 + abs(duration - ref_duration) / ref_duration
                norm_costs[j_end] = norm_cost * dur_penalty
                starts[j_end] = j_start
                path_lengths[j_end] = len(path)
                back_paths[j_end] = np.array(path)

            valid_idx = np.where(np.isfinite(norm_costs))[0]
            if valid_idx.size == 0:
                raise RuntimeError("No valid DTW endpoint found.")

            costs_vec = norm_costs[1:]
            costs_vec[np.isinf(costs_vec)] = (
                np.max(costs_vec[np.isfinite(costs_vec)]) * 2
            )
            smooth = gaussian_filter1d(costs_vec, sigma=min(self.smoothing_sigma, 1.0))

            # pick best endpoint but reject too-short paths
            sorted_idxs = np.argsort(smooth)
            best_j_end, best_j_start = None, None
            for cand in sorted_idxs[:10]:
                j_end = int(cand) + 1
                j_start = starts[j_end]
                dur = (j_end - j_start) * self.hop_length / self.sr
                if (
                    self.min_duration_frac * ref_duration
                    <= dur
                    <= self.max_duration_frac * ref_duration
                ):
                    best_j_end, best_j_start = j_end, j_start
                    break
            if best_j_end is None:
                best_j_end = int(np.argmin(smooth)) + 1
                best_j_start = starts[best_j_end]

            start_time = best_j_start * self.hop_length / self.sr
            end_time = best_j_end * self.hop_length / self.sr
            norm_cost = norm_costs[best_j_end]

        # ---------------- Energy checks ----------------
        matched_duration = end_time - start_time
        s_idx = int(start_time * self.sr)
        e_idx = int(end_time * self.sr)
        matched_audio = long_audio[s_idx:e_idx] if e_idx > s_idx else np.array([])
        matched_rms = (
            np.sqrt(np.mean(matched_audio**2)) if matched_audio.size > 0 else 0.0
        )

        if matched_rms < self.min_rms:
            print(
                f"⚠️ Low energy in matched region (RMS={matched_rms:.6f}). Might be false positive."
            )

        print(
            f"Selected match {start_time:.3f}s - {end_time:.3f}s, norm_cost={norm_cost:.4f}, duration={matched_duration:.3f}s, rms={matched_rms:.6f}"
        )

        return {
            "start": start_time,
            "end": end_time,
            "norm_cost": norm_cost,
            "matched_duration": matched_duration,
            "matched_rms": matched_rms,
        }

    def remove_timestamp_overlaps(
        self, timestamps: list[SentenceTimestamp], tolerance: float = 0.05
    ):
        """
        Remove overlap from sentence timestamps by adjusting boundaries at midpoint.

        Args:
            timestamps: List of SentenceTimestamp objects
            tolerance: Minimum gap between sentences (default: 0.05s)

        Returns:
            List of non-overlapping SentenceTimestamp objects
        """
        if not timestamps:
            return []

        # Filter out invalid timestamps
        valid_timestamps = [
            ts for ts in timestamps if ts.start is not None and ts.end is not None
        ]
        invalid_timestamps = [
            ts for ts in timestamps if ts.start is None or ts.end is None
        ]

        if not valid_timestamps:
            return timestamps

        # Sort by start time
        valid_timestamps.sort(key=lambda x: x.start)
        non_overlapping = [valid_timestamps[0]]

        for current in valid_timestamps[1:]:
            last = non_overlapping[-1]

            # Check for overlap
            if current.start > last.end + tolerance:
                # No overlap, add as is
                non_overlapping.append(current)
            else:
                # Overlap detected, adjust boundaries at midpoint
                midpoint = (last.end + current.start) / 2

                # Adjust last timestamp's end
                non_overlapping[-1] = SentenceTimestamp(
                    sentence=last.sentence,
                    start=last.start,
                    end=midpoint - (tolerance / 2),
                )

                # Adjust current timestamp's start
                current = SentenceTimestamp(
                    sentence=current.sentence,
                    start=midpoint + (tolerance / 2),
                    end=current.end,
                )
                non_overlapping.append(current)

        # Add back invalid timestamps at the end (preserving original order)
        return non_overlapping + invalid_timestamps

    def get_valid_sentence_timestamps(self):
        """
        Process all stimulus sentences and find their timestamps in the audio.

        Returns:
            List of SentenceTimestamp objects
        """
        if not self.audio_path or not os.path.exists(self.audio_path):
            raise ValueError("Error: No valid audio path provided.")

        stimulus_sentences = self.get_stimulus_sentences()

        # Load long audio
        long_audio, _ = librosa.load(self.audio_path, sr=self.sr)

        valid_timestamps: list[SentenceTimestamp] = []
        # Store metadata separately (norm_cost and rms for each sentence)
        metadata_list = []

        # Create directory for temporary reference files
        temp_dir = os.path.join(os.path.dirname(self.audio_path), "dtw_temp")
        os.makedirs(temp_dir, exist_ok=True)

        for idx, sentence in enumerate(stimulus_sentences):
            print(f"\n{'='*60}")
            print(
                f"Processing sentence {idx+1}/{len(stimulus_sentences)}: '{sentence}'"
            )
            print(f"{'='*60}")

            try:
                # Generate reference audio
                ref_path = os.path.join(temp_dir, f"ref_sentence_{idx+1}.wav")
                ref_audio = self.generate_reference_audio(sentence, ref_path)

                # Find match
                match = self.find_sentence_match(ref_audio, long_audio)

                if match:
                    # Apply timestamp tolerance
                    audio_duration = parselmouth.Sound(
                        self.audio_path
                    ).get_total_duration()
                    start_time = max(0, match["start"] - self.timestamp_tolerance)
                    end_time = min(
                        audio_duration, match["end"] + self.timestamp_tolerance
                    )

                    valid_timestamps.append(
                        SentenceTimestamp(
                            sentence=sentence, start=start_time, end=end_time
                        )
                    )
                    # Store metadata
                    metadata_list.append(
                        {
                            "norm_cost": match["norm_cost"],
                            "matched_rms": match["matched_rms"],
                        }
                    )
                    print(f"✓ Match found: {start_time:.3f}s - {end_time:.3f}s")
                else:
                    print(f"✗ No match found for sentence '{sentence}'")
                    valid_timestamps.append(
                        SentenceTimestamp(sentence=sentence, start=None, end=None)
                    )
                    metadata_list.append({"norm_cost": None, "matched_rms": None})

            except Exception as e:
                print(f"✗ Error processing sentence '{sentence}': {str(e)}")
                valid_timestamps.append(
                    SentenceTimestamp(sentence=sentence, start=None, end=None)
                )
                metadata_list.append({"norm_cost": None, "matched_rms": None})

        # Remove overlaps before saving and returning
        # valid_timestamps = self.remove_timestamp_overlaps(
        #     valid_timestamps, tolerance=self.timestamp_tolerance
        # )

        # Save results to JSON for debugging (with metadata)
        self.save_results_to_json(valid_timestamps, metadata_list)

        return valid_timestamps

    def save_results_to_json(
        self, timestamps: list[SentenceTimestamp], metadata_list: list[dict] = None
    ):
        """
        Save DTW results to JSON file for debugging and analysis.

        Args:
            timestamps: List of sentence timestamps
            metadata_list: List of metadata dictionaries containing norm_cost and matched_rms
        """
        json_filename = (
            f"dtw_results_{os.path.basename(self.audio_path).replace('.wav', '')}.json"
        )
        json_filepath = os.path.join(os.path.dirname(self.audio_path), json_filename)

        results = {
            "audio_file": os.path.basename(self.audio_path),
            "timestamp": str(np.datetime64("now")),
            "parameters": {
                "sr": self.sr,
                "hop_length": self.hop_length,
                "n_mfcc": self.n_mfcc,
                "delta_order": self.delta_order,
                "smoothing_sigma": self.smoothing_sigma,
                "min_duration_frac": self.min_duration_frac,
                "max_duration_frac": self.max_duration_frac,
                "min_rms": self.min_rms,
                "tts_tld": self.tts_tld,
                "voice_model": self.voice_model,
            },
            "sentences": [],
        }

        for idx, ts in enumerate(timestamps):
            sentence_data = {
                "text": ts.sentence,
                "start": float(ts.start) if ts.start is not None else None,
                "end": float(ts.end) if ts.end is not None else None,
            }

            # Add metadata if provided
            if metadata_list and idx < len(metadata_list):
                norm_cost = metadata_list[idx].get("norm_cost")
                matched_rms = metadata_list[idx].get("matched_rms")
                sentence_data["norm_cost"] = (
                    float(norm_cost) if norm_cost is not None else None
                )
                sentence_data["matched_rms"] = (
                    float(matched_rms) if matched_rms is not None else None
                )

            results["sentences"].append(sentence_data)

        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"DTW results saved to: {json_filepath}")
        print(f"{'='*60}")

    def save_sentences_as_textgrid(self, sentence_timestamps: list[SentenceTimestamp]):
        """
        Save sentence timestamps to a TextGrid file.

        Args:
            sentence_timestamps: List of SentenceTimestamp objects
        """
        if not sentence_timestamps:
            raise ValueError("Error: The timestamps list is empty.")

        if not self.audio_path or not os.path.exists(self.audio_path):
            raise ValueError("Error: No valid audio path provided.")

        if not self.textgrid_path:
            raise ValueError("Error: No valid TextGrid path provided.")

        tg = textgrid.Textgrid()
        tg_entries = []

        for timestamp in sentence_timestamps:
            start, end, text = timestamp.start, timestamp.end, timestamp.sentence

            # Ensure timestamps are valid
            if start is not None and end is not None and end > start:
                tg_entries.append((start, end, text))

        if not tg_entries:
            raise ValueError("Error: No valid timestamps found.")

        audio_duration = parselmouth.Sound(self.audio_path).get_total_duration()
        tier = textgrid.IntervalTier(
            name="sentences", entries=tg_entries, minT=0, maxT=audio_duration
        )

        tg.addTier(tier)

        # Save with UTF-8 encoding
        try:
            tg.save(self.textgrid_path, format="long_textgrid", includeBlankSpaces=True)
            print(f"TextGrid saved to: {self.textgrid_path}")
        except UnicodeEncodeError:
            print("Warning: Unicode encoding issue detected. Attempting to fix...")
            for tier in tg.tierList:
                for interval in tier.intervalList:
                    if hasattr(interval, "label") and interval.label:
                        interval.label = interval.label.encode(
                            "utf-8", errors="replace"
                        ).decode("utf-8")
            tg.save(self.textgrid_path, format="long_textgrid", includeBlankSpaces=True)
            print(f"TextGrid saved to: {self.textgrid_path}")
