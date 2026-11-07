import os
from typing import Literal
import numpy as np
import pandas as pd
from parselmouth import Sound, praat
from helper import FormantTarget, PhonemeTimestamp, ensure_utf8_display
import difflib
from itertools import combinations


class FormantData:
    PRAAT_FORMANT_TIME_STEP = 0.0015
    PRAAT_N_FORMANTS = 5
    PRAAT_FORMANT_MAX_FREQ = 5500
    PRAAT_FORMANT_WINDOW_LENGTH = 0.025
    PRAAT_FORMANT_PREEMPHASIS_FROM = 50
    VITERBI_EMISSION_MONOPHTHONG_WEIGHT = 100.0
    VITERBI_EMISSION_DIPHTHONG_WEIGHT = 1000.0
    VITERBI_DELTA = 50.0

    DEBUG_LOG_FILE = "debug_log.txt"
    OUTLIER_FIX_LOG_FILE = "outlier_fix_log.txt"

    @staticmethod
    def debug_print(msg):
        print(msg)
        with open(FormantData.DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

    @staticmethod
    def outlier_fix_log_print(msg):
        print(msg)
        with open(FormantData.OUTLIER_FIX_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

    FORMANT_TARGET_RECORD_FILE = "formant_targets.csv"

    @staticmethod
    def clear_formant_target_record_file():
        if os.path.exists(FormantData.FORMANT_TARGET_RECORD_FILE):
            os.remove(FormantData.FORMANT_TARGET_RECORD_FILE)
        with open(FormantData.FORMANT_TARGET_RECORD_FILE, "w", encoding="utf-8") as f:
            f.write(
                "sentence;word;phoneme;start(in s);end(in s);target1_timestamp(in s);target1_f1(in Hz);target1_f2(in Hz);target1_f3(in Hz);target1_f4(in Hz);target2_timestamp(in s);target2_f1(in Hz);target2_f2(in Hz);target2_f3(in Hz);target2_f4(in Hz)\n"
            )

    @staticmethod
    def write_formant_target_record(phoneme_timestamp: PhonemeTimestamp):
        if not os.path.exists(FormantData.FORMANT_TARGET_RECORD_FILE):
            FormantData.debug_print(
                f"DEBUG: Creating new formant target record file: {FormantData.FORMANT_TARGET_RECORD_FILE}"
            )
            with open(
                FormantData.FORMANT_TARGET_RECORD_FILE, "w", encoding="utf-8"
            ) as f:
                f.write(
                    "sentence;word;phoneme;start(in s);end(in s);target1_timestamp(in s);target1_f1(in Hz);target1_f2(in Hz);target1_f3(in Hz);target1_f4(in Hz);target2_timestamp(in s);target2_f1(in Hz);target2_f2(in Hz);target2_f3(in Hz);target2_f4(in Hz)\n"
                )
        with open(FormantData.FORMANT_TARGET_RECORD_FILE, "a", encoding="utf-8") as f:
            if len(phoneme_timestamp.formant_targets) == 0:
                return  # No targets to record
            else:
                target1 = phoneme_timestamp.formant_targets[0]
                target1_values = np.round(target1.targets).tolist()[:4]
            target1_timestamp = np.round(target1.timestamp, 3) if target1 else None

            if len(phoneme_timestamp.formant_targets) > 1:
                target2 = phoneme_timestamp.formant_targets[1]
                target2_values = np.round(target2.targets).tolist()[:4]
            else:
                target2 = None
                target2_values = [None] * 4
            target2_timestamp = np.round(target2.timestamp, 3) if target2 else None
            target_details = (
                [target1_timestamp]
                + target1_values
                + [target2_timestamp]
                + target2_values
            )
            sentence = phoneme_timestamp.parent_sentence_timestamp.sentence
            # Ensure all values are strings for CSV compatibility
            if "," in sentence:
                sentence = '"' + sentence + '"'
            record = [
                sentence,
                phoneme_timestamp.parent_word_timestamp.word,
                phoneme_timestamp.phoneme,
                np.round(phoneme_timestamp.start, 3),
                np.round(phoneme_timestamp.end, 3),
            ] + target_details
            f.write(";".join([str(x) for x in record]) + "\n")
            FormantData.debug_print("DEBUG: Wrote formant target record:")
            FormantData.debug_print(";".join([str(x) for x in record]))

    @staticmethod
    def huber_loss(x, delta=50.0):
        """Huber loss for robust transition cost."""
        x = np.array(x)
        mask = np.abs(x) <= delta
        loss = np.empty_like(x)
        loss[mask] = 0.5 * x[mask] ** 2
        loss[~mask] = delta * (np.abs(x[~mask]) - 0.5 * delta)
        return np.sum(loss)

    @staticmethod
    def modified_z_score(data):
        """
        Calculate modified z-score using median absolute deviation (MAD).
        Modified z-score = 0.6745 * (x - median) / MAD
        More robust to outliers than standard z-score.

        Args:
            data: 1D array of values

        Returns:
            np.ndarray: Modified z-scores for the data
        """
        from scipy.stats import median_abs_deviation

        data = np.array(data)

        # Check if all values are NaN
        if np.all(np.isnan(data)):
            return np.zeros_like(data)

        median = np.nanmedian(data)
        mad = median_abs_deviation(data, nan_policy="omit")

        if mad == 0 or np.isnan(mad):
            return np.zeros_like(data)

        modified_z = 0.6745 * (data - median) / mad
        return modified_z

    @staticmethod
    def formant_clean_shift_outliers(
        *,
        formant_values,
        formant_times,
        n_formants_to_clean=4,
        threshold=3.5,
    ):
        """
        Clean formant values by detecting outliers using modified z-scores
        and shifting values from higher formants down when it improves the score.
        Also cascades rejected values downward if they benefit lower formants.

        Parameters:
        -----------
        formant_values : np.ndarray, shape (T, M)
            Array of formant values where T is time points and M is number of formants
        formant_times : np.ndarray, shape (T,)
            Array of timestamps (not used in computation but kept for API consistency)
        n_formants_to_clean : int, default=4
            Number of formants to clean (typically 4: F1-F4)
        threshold : float, default=3.5
            Modified z-score threshold for outlier detection

        Returns:
        --------
        cleaned_formants : np.ndarray, shape (T, M)
            Cleaned formant array with outliers replaced by shifted values
        """
        T, M = formant_values.shape
        cleaned_formants = formant_values.copy()

        # Limit to the formants we want to clean
        n = min(n_formants_to_clean, M)

        # Calculate modified z-scores for each formant trajectory (along time)
        for formant_idx in range(n):
            trajectory = cleaned_formants[:, formant_idx]
            z_scores = FormantData.modified_z_score(trajectory)

            # Find outliers in this formant trajectory
            outlier_times = np.where(np.abs(z_scores) > threshold)[0]

            for t in outlier_times:
                if np.isnan(cleaned_formants[t, formant_idx]):
                    continue

                # Store the rejected value for potential downward cascade
                rejected_value = cleaned_formants[t, formant_idx]

                # Try to replace with value from formant above (higher frequency)
                for higher_formant_idx in range(formant_idx + 1, M):
                    if np.isnan(cleaned_formants[t, higher_formant_idx]):
                        continue

                    # Calculate current modified z-score at time t for formant i
                    current_trajectory = cleaned_formants[:, formant_idx].copy()
                    current_z_score = np.abs(
                        FormantData.modified_z_score(current_trajectory)[t]
                    )

                    # Test replacement: shift value from higher formant down
                    test_trajectory = current_trajectory.copy()
                    test_trajectory[t] = cleaned_formants[t, higher_formant_idx]
                    test_z_score = np.abs(
                        FormantData.modified_z_score(test_trajectory)[t]
                    )

                    # If the shift improves (lowers) the z-score, apply it
                    if test_z_score < current_z_score:
                        # Track the last successfully shifted formant index
                        last_shifted_idx = (
                            formant_idx - 1
                        )  # Will be updated as we shift
                        shift_accepted = False

                        # Iteratively shift all formants (including the first one) with donor check
                        for shift_idx in range(formant_idx, M - 1):
                            # Calculate z-score for the current formant before shift
                            current_shift_trajectory = cleaned_formants[
                                :, shift_idx
                            ].copy()
                            current_shift_z_before = np.abs(
                                FormantData.modified_z_score(current_shift_trajectory)[
                                    t
                                ]
                            )

                            # Check if we can take value from next higher formant
                            if not np.isnan(cleaned_formants[t, shift_idx + 1]):
                                # Check if the current formant improves
                                test_current_shift_trajectory = (
                                    current_shift_trajectory.copy()
                                )
                                test_current_shift_trajectory[t] = cleaned_formants[
                                    t, shift_idx + 1
                                ]
                                current_shift_z_after = np.abs(
                                    FormantData.modified_z_score(
                                        test_current_shift_trajectory
                                    )[t]
                                )

                                # Check if it improves the receiver
                                if current_shift_z_after < current_shift_z_before:
                                    # Only check donor health if it's NOT the last formant (F5)
                                    donor_ok = True
                                    if shift_idx + 1 < M - 1:  # Not the last formant
                                        # Check if taking from the next higher formant worsens its z-score
                                        donor_trajectory = cleaned_formants[
                                            :, shift_idx + 1
                                        ].copy()
                                        donor_z_before = np.abs(
                                            FormantData.modified_z_score(
                                                donor_trajectory
                                            )[t]
                                        )

                                        # Simulate the donor losing this value
                                        test_donor_trajectory = donor_trajectory.copy()
                                        # If donor is not the last formant, it will take from above, otherwise it becomes NaN
                                        if shift_idx + 2 < M and not np.isnan(
                                            cleaned_formants[t, shift_idx + 2]
                                        ):
                                            test_donor_trajectory[t] = cleaned_formants[
                                                t, shift_idx + 2
                                            ]
                                        else:
                                            test_donor_trajectory[t] = np.nan
                                        donor_z_after = np.abs(
                                            FormantData.modified_z_score(
                                                test_donor_trajectory
                                            )[t]
                                        )

                                        # Check if donor doesn't worsen too much
                                        donor_ok = donor_z_after <= donor_z_before * 1.5

                                    # Apply shift if receiver improves and donor is OK (or is last formant)
                                    if donor_ok:
                                        cleaned_formants[t, shift_idx] = (
                                            cleaned_formants[t, shift_idx + 1]
                                        )
                                        last_shifted_idx = shift_idx
                                        shift_accepted = True
                                    else:
                                        # Stop shifting if donor worsens too much
                                        break
                                else:
                                    # Stop shifting if it doesn't improve receiver
                                    break
                            else:
                                # No more formants above to shift from
                                break

                        # Only proceed if at least the first shift was accepted
                        if not shift_accepted:
                            # Revert the first shift that we were considering
                            continue

                        # Mark the topmost shifted formant as NaN (only the last one that was shifted)
                        if last_shifted_idx < M - 1:
                            cleaned_formants[t, last_shifted_idx + 1] = np.nan

                        # Now cascade the rejected value downward if it benefits lower formants
                        if formant_idx > 0:
                            # Check if any lower formant can benefit from the rejected value
                            for lower_formant_idx in range(formant_idx - 1, -1, -1):
                                if np.isnan(cleaned_formants[t, lower_formant_idx]):
                                    continue

                                # Calculate current z-score for lower formant
                                lower_trajectory = cleaned_formants[
                                    :, lower_formant_idx
                                ].copy()
                                lower_z_before = np.abs(
                                    FormantData.modified_z_score(lower_trajectory)[t]
                                )

                                # Test if rejected value improves lower formant
                                test_lower_trajectory = lower_trajectory.copy()
                                test_lower_trajectory[t] = rejected_value
                                lower_z_after = np.abs(
                                    FormantData.modified_z_score(test_lower_trajectory)[
                                        t
                                    ]
                                )

                                # If it improves, apply the shift downward
                                if lower_z_after < lower_z_before:
                                    # Store the value being replaced for further downward cascade
                                    next_rejected_value = cleaned_formants[
                                        t, lower_formant_idx
                                    ]
                                    cleaned_formants[t, lower_formant_idx] = (
                                        rejected_value
                                    )
                                    rejected_value = next_rejected_value
                                else:
                                    # Stop cascading downward if it doesn't improve
                                    break

                        break  # Stop searching higher formants for this outlier

        return cleaned_formants

    @staticmethod
    def safe_float_convert(value, default=np.nan):
        """
        Safely convert a value to float, handling empty strings, None, and invalid types.

        Args:
            value: The value to convert (can be str, int, float, None, etc.)
            default: The default value to return if conversion fails (default: np.nan)

        Returns:
            float: The converted value or default if conversion fails
        """
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return default
            try:
                return float(value)
            except ValueError:
                return default
        # For any other type, try direct conversion
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def generate_candidates(peaks, n_formants=4):
        """Generate all ascending candidate subsets of size n_formants."""
        candidates = []
        for subset in combinations(peaks, n_formants):
            if all(~np.isnan(subset)):
                candidates.append(np.sort(subset))
        return candidates

    @staticmethod
    def formant_clean_viterbi_fb(
        *,
        formant_values,
        formant_times,
        n_formants_to_clean=4,
        delta=50.0,
        smooth_window=7,
        polyorder=2,
        targets=None,  # list of target dicts (see below)
        overall_emission_weight=1.0,
    ):
        """
        formant_values: (T, M) Praat peaks
        formant_times: (T,) timestamps (float seconds)
        targets: list of dicts, each:
        {
            "start_frac": 0.05,        # fraction of vowel duration (0..1)
            "end_frac": 0.20,
            "mean": np.array([...]),   # length >= n_formants_to_clean, use np.nan for unknown
            "sd": np.array([...]),     # same length
            "weight": 1.0              # optional scalar
        }
        """
        EPS_SD = 1e-2  # floor for sd to avoid division by zero

        T, M = formant_values.shape
        n = n_formants_to_clean
        # map frame index -> relative position in vowel (0..1)
        t0 = formant_times[0]
        tf = formant_times[-1]
        dur = max(tf - t0, 1e-8)
        rel_pos = np.array([(tt - t0) / dur for tt in formant_times], dtype=float)

        # Preprocess targets list
        targets = [] if targets is None else targets
        # ensure mean/sd numpy arrays and weights
        proc_targets = []
        for tgt in targets:
            mean = np.asarray(tgt.get("mean", []), dtype=float)
            sd = np.asarray(tgt.get("sd", []), dtype=float)
            weight = FormantData.safe_float_convert(tgt.get("weight", 1.0), default=1.0)
            proc_targets.append(
                {
                    "start_frac": FormantData.safe_float_convert(
                        tgt["start_frac"], default=0.0
                    ),
                    "end_frac": FormantData.safe_float_convert(
                        tgt["end_frac"], default=1.0
                    ),
                    "mean": mean,
                    "sd": sd,
                    "weight": weight,
                }
            )

        # ---- build candidates ----
        all_candidates = []
        C_max = 0
        for t in range(T):
            peaks = formant_values[t, :]
            cands = FormantData.generate_candidates(peaks, n)
            if not cands:
                cands = [np.full(n, np.nan)]
            all_candidates.append(cands)
            C_max = max(C_max, len(cands))

        # pad to C_max
        for t in range(T):
            while len(all_candidates[t]) < C_max:
                all_candidates[t].append(np.full(n, np.nan))

        # ---- precompute emission cost matrix (T x C_max) ----
        emission = np.zeros((T, C_max), dtype=float)
        emission_log = []
        for t in range(T):
            rel = rel_pos[t]
            for j, cand in enumerate(all_candidates[t]):
                c = 0.0
                vals = np.asarray(cand, dtype=float)
                # for each target whose window contains this frame, add penalty
                for tgt in proc_targets:
                    if (
                        rel + 1e-12 >= tgt["start_frac"]
                        and rel <= tgt["end_frac"] + 1e-12
                    ):
                        mu = tgt["mean"]
                        sd = tgt["sd"] if "sd" in tgt else None
                        w = tgt.get("weight", 1.0)
                        FormantData.outlier_fix_log_print(f"emission calc t={t}, j={j}: applying target window [{tgt['start_frac']:.3f}, {tgt['end_frac']:.3f}] with weight={w}")
                        # for each formant k up to n, if mu present and not nan, add squared z-score
                        for k in range(n):
                            if k < mu.size and not np.isnan(mu[k]):
                                denom = (
                                    sd[k]
                                    if k < sd.size and not np.isnan(sd[k])
                                    else EPS_SD
                                )
                                denom = max(denom, EPS_SD)
                                if not np.isnan(vals[k]):
                                    z = (vals[k] - mu[k]) / denom
                                    c += w * (z * z)  # squared z-score
                                    FormantData.outlier_fix_log_print(f"emission calc t={t}, j={j}, k={k}: val={vals[k]:.2f}, mu={mu[k]:.2f}, sd={denom:.2f}, z={z:.4f}, w={w} emission_cost_contrib={w * (z * z):.4f}")
                                else:
                                    # missing candidate value where mean exists: penalize moderately
                                    c += w * 4.0
                emission[t, j] = float(c) * overall_emission_weight
                emission_log.append(f"emission[t={t},j={j}] = {emission[t, j]:.4f} (raw={c:.4f}, weight={overall_emission_weight})")
        FormantData.outlier_fix_log_print(f"VITERBI overall emission_weight={overall_emission_weight}")
        FormantData.outlier_fix_log_print(f"VITERBI emission matrix shape: {emission.shape}")
        FormantData.outlier_fix_log_print(f"VITERBI emission matrix sum: {np.sum(emission):.4f}")
        for log_entry in emission_log[:min(20, len(emission_log))]:
            FormantData.outlier_fix_log_print(log_entry)
        if len(emission_log) > 20:
            FormantData.outlier_fix_log_print(f"... {len(emission_log)-20} more emission entries omitted ...")

        # ---- Forward DP (include emission at each frame) ----
        INF = np.inf
        forward_cost = np.full((T, C_max), INF, dtype=float)
        forward_bp = np.zeros((T, C_max), dtype=int)

        # init first frame: emission cost
        for j in range(C_max):
            forward_cost[0, j] = emission[0, j]

        for t in range(1, T):
            for j, cand_j in enumerate(all_candidates[t]):
                best_cost = INF
                best_i = 0
                vals_j = cand_j
                for i, cand_i in enumerate(all_candidates[t - 1]):
                    vals_i = cand_i
                    # skip if either candidate entirely nan
                    if np.all(np.isnan(vals_i)) or np.all(np.isnan(vals_j)):
                        continue
                    trans = FormantData.huber_loss(vals_j - vals_i, delta)
                    cost = forward_cost[t - 1, i] + trans + emission[t, j]
                    if cost < best_cost:
                        best_cost = cost
                        best_i = i
                forward_cost[t, j] = best_cost
                forward_bp[t, j] = best_i

        # ---- Backward DP (include emission at each frame) ----
        backward_cost = np.full((T, C_max), INF, dtype=float)
        backward_bp = np.zeros((T, C_max), dtype=int)
        for j in range(C_max):
            backward_cost[-1, j] = emission[-1, j]

        for t in range(T - 2, -1, -1):
            for i, cand_i in enumerate(all_candidates[t]):
                best_cost = INF
                best_j = 0
                vals_i = cand_i
                for j, cand_j in enumerate(all_candidates[t + 1]):
                    vals_j = cand_j
                    if np.all(np.isnan(vals_i)) or np.all(np.isnan(vals_j)):
                        continue
                    trans = FormantData.huber_loss(vals_j - vals_i, delta)
                    cost = backward_cost[t + 1, j] + trans + emission[t, i]
                    if cost < best_cost:
                        best_cost = cost
                        best_j = j
                backward_cost[t, i] = best_cost
                backward_bp[t, i] = best_j

        # ---- Combine forward + backward, subtract emission once ----
        total_cost = forward_cost + backward_cost - emission
        best_indices = np.argmin(total_cost, axis=1)

        # ---- Reconstruct cleaned formants (first n formants) ----
        cleaned_formants = np.zeros((T, M), dtype=float)
        for t in range(T):
            cleaned_formants[t, :n] = all_candidates[t][best_indices[t]]

        # # Optional smoothing on cleaned tracks (only on the cleaned F1..Fn)
        # for k in range(n):
        #     col = cleaned_formants[:, k]
        #     # if NaNs present, interpolate first
        #     good = ~np.isnan(col)
        #     if np.sum(good) >= 3:
        #         xi = np.arange(T)
        #         interp = np.interp(xi, xi[good], col[good])
        #         # ensure window odd and <= T
        #         w = int(smooth_window) if int(smooth_window) % 2 == 1 else int(smooth_window) + 1
        #         w = min(w, T if T % 2 == 1 else T - 1)
        #         if w >= 3:
        #             cleaned_formants[:, k] = FormantData.savgol_filter(interp, w, polyorder)
        #         else:
        #             cleaned_formants[:, k] = interp
        #     else:
        #         # leave as-is
        #         cleaned_formants[:, k] = col

        # remaining formants set to PRAAT max if desired
        if M > n:
            cleaned_formants[:, n:] = (
                np.ones((T, M - n)) * FormantData.PRAAT_FORMANT_MAX_FREQ
            )

        return cleaned_formants

    def __init__(
        self,
        *,
        sound: Sound | None = None,
        n_formants: int = 5,
        sampling_rate: int = 16000,
        ignore_f0: bool = True,
        landmark_info_filepath: str | None = None,
    ):
        self.sound = sound
        self.n_formants = n_formants
        self.sampling_rate = sampling_rate
        self.formant_times: np.ndarray = np.array([])
        self.formant_values: np.ndarray = np.array([])
        self.ignore_f0 = ignore_f0
        self.landmark_info_filepath = landmark_info_filepath
        self.ground_truth_filepath = "landmark_identification_ground_truth.csv"
        self.aus_brit_phoneme_mapping_filepath = (
            "landmark_identification_phoneme_mapping.csv"
        )
        # Clear debug log before writing the first debug message
        with open(FormantData.DEBUG_LOG_FILE, "w", encoding="utf-8") as f:
            pass
        with open(FormantData.OUTLIER_FIX_LOG_FILE, "w", encoding="utf-8") as f:
            pass
        with open(FormantData.FORMANT_TARGET_RECORD_FILE, "w", encoding="utf-8") as f:
            f.write(
                "sentence;word;phoneme;start(in s);end(in s);target1_timestamp(in s);target1_f1(in Hz);target1_f2(in Hz);target1_f3(in Hz);target1_f4(in Hz);target2_timestamp(in s);target2_f1(in Hz);target2_f2(in Hz);target2_f3(in Hz);target2_f4(in Hz)\n"
            )
        self.gender = "F"

    def set_landmark_info_filepath(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Landmark info file not found: {filepath}")
        self.landmark_info_filepath = filepath

    def set_gender(self, gender: str = "F"):
        self.gender = gender

    def extract_formant_data_from_sound(self, sound: Sound | None = None):
        if sound is None:
            sound = self.sound
        if not isinstance(sound, Sound):
            raise ValueError("Input must be a parselmouth Sound object.")
        formants = praat.call(
            sound,
            "To Formant (burg)",
            self.PRAAT_FORMANT_TIME_STEP,
            self.PRAAT_N_FORMANTS,
            self.PRAAT_FORMANT_MAX_FREQ,
            self.PRAAT_FORMANT_WINDOW_LENGTH,
            self.PRAAT_FORMANT_PREEMPHASIS_FROM,
        )
        formant_times = np.array([])
        formant_values = np.array([])

        for i in range(1, praat.call(formants, "Get number of frames")):
            t = praat.call(formants, "Get time from frame number", i)
            formant_times = np.append(formant_times, t)
            f_at_t = []
            if self.ignore_f0:
                for formant_num in range(1, self.n_formants + 1):
                    f = praat.call(
                        formants, "Get value at time", formant_num, t, "Hertz", "Linear"
                    )
                    f_at_t.append(f)
            else:
                for formant_num in range(self.n_formants + 1):
                    f = praat.call(
                        formants, "Get value at time", formant_num, t, "Hertz", "Linear"
                    )
                    f_at_t.append(f)
            formant_values = (
                np.vstack((formant_values, f_at_t))
                if formant_values.size
                else np.array(f_at_t)
            )
        # Write formant values to CSV
        csv_filename = "formants_values.csv"
        n_formants = formant_values.shape[1] if formant_values.ndim > 1 else 1
        headings = ["timestamp(in s)"] + [f"F{i+1}" for i in range(n_formants)]
        with open(csv_filename, "w", encoding="utf-8") as f:
            f.write(";".join(headings) + "\n")
            for idx in range(formant_values.shape[0]):
                row = [str(round(formant_times[idx], 6))] + [
                    str(round(val, 2)) for val in formant_values[idx]
                ]
                f.write(";".join(row) + "\n")
        self.set_formant_times(formant_times)
        self.set_formant_values(formant_values)
        print(
            f"Extracted {self.formant_values.shape[1]} formants, {self.formant_values.shape[0]} formant values from {self.formant_times.shape[0]} time points."
        )

    def insert_formant_data_points(
        self,
        sound: Sound,
        start_time: float,
        end_time: float,
        n_points: int | None = None,
        praat_formant_max_freq: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(sound, Sound):
            raise ValueError("Input must be a parselmouth Sound object.")
        if n_points is not None and n_points <= 0:
            raise ValueError("Number of points must be a positive integer.")

        sound_section = sound.extract_part(
            from_time=start_time, to_time=end_time, preserve_times=False
        )
        if n_points is None:
            time_step = self.PRAAT_FORMANT_TIME_STEP
        else:
            time_step = (end_time - start_time) / n_points
        if praat_formant_max_freq is None:
            praat_formant_max_freq = self.PRAAT_FORMANT_MAX_FREQ
        sound_section_formants = praat.call(
            sound_section,
            "To Formant (burg)",
            time_step,
            self.PRAAT_N_FORMANTS,
            praat_formant_max_freq,
            self.PRAAT_FORMANT_WINDOW_LENGTH,
            self.PRAAT_FORMANT_PREEMPHASIS_FROM,
        )
        formant_times = np.array([])
        formant_values = np.array([])

        for i in range(1, praat.call(sound_section_formants, "Get number of frames")):
            t = praat.call(sound_section_formants, "Get time from frame number", i)
            formant_times = np.append(formant_times, t)
            f_at_t = []
            if self.ignore_f0:
                for formant_num in range(1, self.n_formants + 1):
                    f = praat.call(
                        sound_section_formants,
                        "Get value at time",
                        formant_num,
                        t,
                        "Hertz",
                        "Linear",
                    )
                    f_at_t.append(f)
            else:
                for formant_num in range(self.n_formants + 1):
                    f = praat.call(
                        sound_section_formants,
                        "Get value at time",
                        formant_num,
                        t,
                        "Hertz",
                        "Linear",
                    )
                    f_at_t.append(f)
            formant_values = (
                np.vstack((formant_values, f_at_t))
                if formant_values.size
                else np.array(f_at_t)
            )
        formant_times += start_time  # Adjust times to original sound
        # Only insert if formant_values is not empty
        if formant_values.size > 0:
            # Remove existing formant values in range
            if self.formant_times.size > 0:
                mask = (self.formant_times < start_time) | (
                    self.formant_times > end_time
                )
                self.formant_times = self.formant_times[mask]
                self.formant_values = self.formant_values[mask]
            insert_idx = np.searchsorted(self.formant_times, formant_times)
            self.formant_times = np.insert(
                self.formant_times, insert_idx, formant_times
            )
            self.formant_values = np.insert(
                self.formant_values, insert_idx, formant_values, axis=0
            )
            FormantData.debug_print(
                f"DEBUG: Generated {formant_values.shape[0]} formant values for time range {start_time}-{end_time}s and inserted into existing formant data."
            )
        else:
            FormantData.debug_print(
                f"DEBUG: No formant values generated for time range {start_time}-{end_time}s. Skipping insertion."
            )
        return formant_times, formant_values

    def set_formant_times(self, times: np.ndarray | list):
        if not isinstance(times, (np.ndarray, list)) or not isinstance(
            times[0], (int, float)
        ):
            raise ValueError(
                "Formant times must be a numpy array or list of int or float values."
            )
        formant_times = times if isinstance(times, np.ndarray) else np.array(times)
        if not np.equal(np.sort(formant_times), formant_times).all():
            raise ValueError("Formant times must be in ascending order.")
        # check if times are not duplicated
        if np.unique(formant_times).size != formant_times.size:
            raise ValueError("Formant times must not contain duplicates.")
        self.formant_times = formant_times

    def set_formant_values(self, values: np.ndarray | list):
        if not isinstance(values, (np.ndarray, list)) or not isinstance(
            values[0][0], (int, float)
        ):
            raise ValueError(
                "Formant values must be a numpy array or list of data points, with one or more int or float values at each data point."
            )
        formant_values = values if isinstance(values, np.ndarray) else np.array(values)
        if formant_values.shape[0] != self.formant_times.shape[0]:
            raise ValueError(
                "Formant values must have the same number of rows as formant times."
            )
        if formant_values.shape[1] != self.n_formants:
            raise ValueError(
                f"Formant values must have {self.n_formants} columns, one for each formant."
            )
        self.formant_values = formant_values

    def get_formant_times(self) -> np.ndarray:
        if self.formant_times.size == 0:
            raise ValueError("Formant times have not been set.")
        return self.formant_times

    def get_formant_values(self) -> np.ndarray:
        if self.formant_values.size == 0:
            raise ValueError("Formant values have not been set.")
        return self.formant_values

    def get_formant_values_at_time(self, time: float) -> np.ndarray:
        if self.formant_times.size == 0 or self.formant_values.size == 0:
            raise ValueError("Formant times and values have not been set.")
        if time < self.formant_times[0] or time > self.formant_times[-1]:
            raise ValueError(f"Time {time} is out of bounds for the formant data.")

        # Find the closest index to the given time
        idx = np.argmin(np.abs(self.formant_times - time))
        # idx = np.where(self.formant_times == time)[0]
        if idx.size == 0:
            raise ValueError(f"No formant data found for time {time}.")
        if idx.size > 1:
            raise ValueError(f"Multiple formant data found for time {time}.")
        if self.formant_values[idx].size != self.n_formants:
            raise ValueError(
                f"Formant values at time {time} do not match the expected number of formants ({self.n_formants})."
            )
        return self.formant_values[idx]

    def fix_formant_values_outliers_for_phoneme_with_viterbi(
        self, phoneme_timestamp: PhonemeTimestamp, use_ground_truth: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        FormantData.outlier_fix_log_print(
            f"Starting outlier fix for phoneme '{phoneme_timestamp.phoneme}' in word '{phoneme_timestamp.parent_word_timestamp.word}' sentence '{phoneme_timestamp.parent_sentence_timestamp.sentence}'"
        )
        # ... (initial data extraction code remains the same) ...
        target_indices = np.where(
            (self.formant_times >= phoneme_timestamp.start)
            & (self.formant_times <= phoneme_timestamp.end)
        )[0]
        phoneme_formant_times, phoneme_formant_values = (
            self.formant_times[target_indices],
            self.formant_values[target_indices],
        )
        if phoneme_formant_times.shape[0] == 0 or phoneme_formant_values.shape[0] == 0:
            return phoneme_formant_times, phoneme_formant_values

        N_FORMANTS_TO_KEEP = 5
        N_FORMANTS_TO_CLEAN = 4

        cleaned_data = phoneme_formant_values.copy()
        if cleaned_data.shape[1] > N_FORMANTS_TO_KEEP:
            cleaned_data = cleaned_data[
                :, :N_FORMANTS_TO_KEEP
            ]  # Limit to first 5 formants for processing
            cleaned_data[:, N_FORMANTS_TO_KEEP - 1] = np.where(
                np.isnan(cleaned_data[:, N_FORMANTS_TO_KEEP - 1]),
                FormantData.PRAAT_FORMANT_MAX_FREQ,
                cleaned_data[:, N_FORMANTS_TO_KEEP - 1],
            )
        FormantData.outlier_fix_log_print(
            "Starting Viterbi-based formant cleaning with forward-backward algorithm..."
        )

        # Construct targets array from ground truth CSV
        targets = []
        if use_ground_truth:
            try:
                # Load ground truth data
                ground_truth_filepath = self.ground_truth_filepath
                if os.path.exists(ground_truth_filepath):
                    ground_truth_pd = pd.read_csv(ground_truth_filepath, header=0)
                    ground_truth_pd.columns = ground_truth_pd.columns.str.strip()

                    # Look up the phoneme data for the current gender
                    phoneme_row = ground_truth_pd[
                        (ground_truth_pd["Gender"] == self.gender)
                        & (ground_truth_pd["Phoneme"] == phoneme_timestamp.phoneme)
                    ]

                    if not phoneme_row.empty:
                        phoneme_row = phoneme_row.iloc[0]

                        # Load landmark info to get target percentages
                        if os.path.exists(self.landmark_info_filepath):
                            landmark_info_pd = pd.read_csv(
                                self.landmark_info_filepath, header=0
                            )
                            landmark_info_pd.columns = (
                                landmark_info_pd.columns.str.strip()
                            )

                            # Match the phoneme in landmark_info (vowel column)
                            landmark_row = landmark_info_pd[
                                landmark_info_pd["Vowel"] == phoneme_timestamp.phoneme
                            ]

                            if not landmark_row.empty:
                                landmark_row = landmark_row.iloc[0]
                                n_targets = int(landmark_row["No. of Targets"])

                                # Target 1
                                if n_targets >= 1 and pd.notna(
                                    phoneme_row["T1_F1_Mean"]
                                ):
                                    # Convert to float safely
                                    t1_start = (
                                        FormantData.safe_float_convert(
                                            landmark_row[
                                                "DefaultTarget1Start (% of total vowel duration)"
                                            ]
                                        )
                                        / 100.0
                                    )
                                    t1_end = (
                                        FormantData.safe_float_convert(
                                            landmark_row[
                                                "DefaultTarget1End (% of total vowel duration)"
                                            ]
                                        )
                                        / 100.0
                                    )

                                    # Build mean and sd arrays (4 formants for cleaning)
                                    t1_mean = np.array(
                                        [
                                            FormantData.safe_float_convert(
                                                phoneme_row["T1_F1_Mean"]
                                            ),
                                            FormantData.safe_float_convert(
                                                phoneme_row["T1_F2_Mean"]
                                            ),
                                            np.nan,
                                            np.nan,
                                        ]
                                    )
                                    t1_sd = np.array(
                                        [
                                            FormantData.safe_float_convert(
                                                phoneme_row["T1_F1_SD"]
                                            ),
                                            FormantData.safe_float_convert(
                                                phoneme_row["T1_F2_SD"]
                                            ),
                                            np.nan,
                                            np.nan,
                                        ]
                                    )

                                    if n_targets > 1:
                                        weight = FormantData.VITERBI_EMISSION_DIPHTHONG_WEIGHT
                                    else:
                                        weight = FormantData.VITERBI_EMISSION_MONOPHTHONG_WEIGHT

                                    targets.append(
                                        {
                                            "start_frac": t1_start,
                                            "end_frac": t1_end,
                                            "mean": t1_mean,
                                            "sd": t1_sd,
                                            "weight": weight,
                                        }
                                    )

                                # Target 2
                                if n_targets >= 2 and pd.notna(
                                    phoneme_row["T2_F1_Mean"]
                                ):
                                    # Convert to float safely
                                    t2_start = (
                                        FormantData.safe_float_convert(
                                            landmark_row[
                                                "DefaultTarget2Start (% of total vowel duration)"
                                            ]
                                        )
                                        / 100.0
                                    )
                                    t2_end = (
                                        FormantData.safe_float_convert(
                                            landmark_row[
                                                "DefaultTarget2End (% of total vowel duration)"
                                            ]
                                        )
                                        / 100.0
                                    )

                                    # Only add target if we have valid start/end values
                                    if not (np.isnan(t2_start) or np.isnan(t2_end)):
                                        t2_mean = np.array(
                                            [
                                                FormantData.safe_float_convert(
                                                    phoneme_row["T2_F1_Mean"]
                                                ),
                                                FormantData.safe_float_convert(
                                                    phoneme_row["T2_F2_Mean"]
                                                ),
                                                np.nan,
                                                np.nan,
                                            ]
                                        )
                                        t2_sd = np.array(
                                            [
                                                FormantData.safe_float_convert(
                                                    phoneme_row["T2_F1_SD"]
                                                ),
                                                FormantData.safe_float_convert(
                                                    phoneme_row["T2_F2_SD"]
                                                ),
                                                np.nan,
                                                np.nan,
                                            ]
                                        )

                                        targets.append(
                                            {
                                                "start_frac": t2_start,
                                                "end_frac": t2_end,
                                                "mean": t2_mean,
                                                "sd": t2_sd,
                                                "weight": FormantData.VITERBI_EMISSION_DIPHTHONG_WEIGHT,
                                            }
                                        )
                                    else:
                                        FormantData.outlier_fix_log_print(
                                            "Skipping Target 2: invalid percentage values"
                                        )

                                FormantData.outlier_fix_log_print(
                                    f"Constructed {len(targets)} target(s) for phoneme '{phoneme_timestamp.phoneme}' (Gender: {self.gender})"
                                )
                            else:
                                FormantData.outlier_fix_log_print(
                                    f"No landmark info found for phoneme '{phoneme_timestamp.phoneme}'"
                                )
                        else:
                            FormantData.outlier_fix_log_print(
                                f"Landmark info file not found: {self.landmark_info_filepath}"
                            )
                    else:
                        FormantData.outlier_fix_log_print(
                            f"No ground truth data found for phoneme '{phoneme_timestamp.phoneme}' with gender '{self.gender}'"
                        )
                else:
                    FormantData.outlier_fix_log_print(
                        f"Ground truth file not found: {ground_truth_filepath}"
                    )
            except Exception as e:
                FormantData.outlier_fix_log_print(
                    f"Error constructing targets: {str(e)}"
                )
                targets = []  # Fall back to empty targets

        # Use the Viterbi forward-backward algorithm for formant cleaning
        cleaned_data = FormantData.formant_clean_viterbi_fb(
            formant_values=cleaned_data,
            formant_times=phoneme_formant_times,
            n_formants_to_clean=N_FORMANTS_TO_CLEAN,
            targets=targets,
            delta=FormantData.VITERBI_DELTA,
            # overall_emission_weight=1.0,
        )

        FormantData.outlier_fix_log_print("Viterbi formant cleaning completed.")

        self.formant_times[target_indices] = phoneme_formant_times
        self.formant_values[target_indices] = cleaned_data

        return phoneme_formant_times, cleaned_data

    def fix_formant_values_outliers_for_phoneme_with_zscore_shift(
        self, phoneme_timestamp: PhonemeTimestamp, threshold: float = 3.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fix formant outliers using modified z-score detection and cascading shift algorithm.

        This method is an alternative to the Viterbi-based approach, using statistical outlier
        detection (modified z-scores with MAD) and a bidirectional value cascade system.

        Parameters:
        -----------
        phoneme_timestamp : PhonemeTimestamp
            The phoneme timestamp containing start/end times and parent word/sentence info
        threshold : float, default=3.5
            Modified z-score threshold for outlier detection. Lower values are more aggressive.
            Typical values: 3.5 (conservative), 2.5 (moderate), 1.0 (aggressive)

        Returns:
        --------
        tuple[np.ndarray, np.ndarray]
            phoneme_formant_times: timestamps for the cleaned segment
            cleaned_data: cleaned formant values
        """
        FormantData.outlier_fix_log_print(
            f"Starting z-score+shift outlier fix for phoneme '{phoneme_timestamp.phoneme}' "
            f"in word '{phoneme_timestamp.parent_word_timestamp.word}' "
            f"sentence '{phoneme_timestamp.parent_sentence_timestamp.sentence}' "
            f"(threshold={threshold})"
        )

        # Extract formant data for this phoneme's time range
        target_indices = np.where(
            (self.formant_times >= phoneme_timestamp.start)
            & (self.formant_times <= phoneme_timestamp.end)
        )[0]
        phoneme_formant_times, phoneme_formant_values = (
            self.formant_times[target_indices],
            self.formant_values[target_indices],
        )

        if phoneme_formant_times.shape[0] == 0 or phoneme_formant_values.shape[0] == 0:
            FormantData.outlier_fix_log_print(
                "No formant data found in phoneme time range. Skipping."
            )
            return phoneme_formant_times, phoneme_formant_values

        N_FORMANTS_TO_KEEP = 5
        N_FORMANTS_TO_CLEAN = 4

        cleaned_data = phoneme_formant_values.copy()
        if cleaned_data.shape[1] > N_FORMANTS_TO_KEEP:
            cleaned_data = cleaned_data[:, :N_FORMANTS_TO_KEEP]
            # Set F5 to max frequency if NaN
            cleaned_data[:, N_FORMANTS_TO_KEEP - 1] = np.where(
                np.isnan(cleaned_data[:, N_FORMANTS_TO_KEEP - 1]),
                FormantData.PRAAT_FORMANT_MAX_FREQ,
                cleaned_data[:, N_FORMANTS_TO_KEEP - 1],
            )

        FormantData.outlier_fix_log_print(
            f"Starting modified z-score outlier detection and cascading shift algorithm "
            f"(threshold={threshold})..."
        )

        # Apply the modified z-score + shift algorithm
        cleaned_data = FormantData.formant_clean_shift_outliers(
            formant_values=cleaned_data,
            formant_times=phoneme_formant_times,
            n_formants_to_clean=N_FORMANTS_TO_CLEAN,
            threshold=threshold,
        )

        FormantData.outlier_fix_log_print(
            "Modified z-score + shift formant cleaning completed."
        )

        # Update the main formant data arrays
        self.formant_times[target_indices] = phoneme_formant_times
        self.formant_values[target_indices] = cleaned_data

        return phoneme_formant_times, cleaned_data

    def set_formant_target_for_phoneme(
        self,
        phoneme_timestamp: PhonemeTimestamp,
        cleaning_algo: Literal["viterbi"] | None = "viterbi", # "median" not used
        use_ground_truth: bool = False,
        dynamic_frequency_ceiling: bool = False,
    ):
        if not os.path.exists(self.landmark_info_filepath):
            FormantData.debug_print(
                f"ERROR: Landmark info file not found: {self.landmark_info_filepath}. Cannot set formant target for phoneme '{phoneme_timestamp.phoneme}'"
            )
            return
        if (
            phoneme_timestamp.parent_sentence_timestamp is None
            or phoneme_timestamp.parent_word_timestamp is None
        ):
            FormantData.debug_print(
                f"ERROR: Phoneme '{phoneme_timestamp.phoneme}' is missing parent sentence or word information. Skipping Phoneme"
            )
            return  # Early return to prevent further processing
        if cleaning_algo and cleaning_algo not in ["viterbi"]: # "median" not used
            FormantData.debug_print(
                f"ERROR: Invalid cleaning algorithm '{cleaning_algo}' specified. Must be 'viterbi' or None."
            )
            return

        if dynamic_frequency_ceiling:
            self.generate_formant_data_points_with_dynamic_frequency_ceiling(
                phoneme_timestamp
            )

        formant_times, formant_values = self.find_formant_target_for_phoneme(
            phoneme_timestamp,
            cleaning_algo=cleaning_algo,
            use_ground_truth=use_ground_truth,
        )
        return formant_times, formant_values

    def generate_formant_data_points_with_dynamic_frequency_ceiling(
        self, phoneme_timestamp: PhonemeTimestamp
    ):
        """
        Insert formant data points for a phoneme if dynamic frequency ceiling condition is met.
        Reads ground truth CSV and checks F1/F2 mean and SD for the given phoneme and gender.
        If (F2mean-SD)-(F1mean+SD)<=1000 and F2mean<2000, inserts formant data points using praat_formant_window_length=5000.
        """
        import pandas as pd

        if not os.path.exists(self.ground_truth_filepath):
            FormantData.debug_print(
                f"Ground truth file not found: {self.ground_truth_filepath}"
            )
            return
        try:
            df = pd.read_csv(self.ground_truth_filepath)
            df.columns = df.columns.str.strip()
            # Find row for this gender and phoneme
            row = df[
                (df["Gender"] == self.gender)
                & (df["Phoneme"] == phoneme_timestamp.phoneme)
            ]
            if row.empty:
                FormantData.debug_print(
                    f"No ground truth found for gender={self.gender}, phoneme={phoneme_timestamp.phoneme}"
                )
                return
            # Try Target 1 first
            F1_mean = (
                row["T1_F1_Mean"].values[0]
                if not pd.isna(row["T1_F1_Mean"].values[0])
                else None
            )
            F1_sd = (
                row["T1_F1_SD"].values[0]
                if not pd.isna(row["T1_F1_SD"].values[0])
                else None
            )
            F2_mean = (
                row["T1_F2_Mean"].values[0]
                if not pd.isna(row["T1_F2_Mean"].values[0])
                else None
            )
            F2_sd = (
                row["T1_F2_SD"].values[0]
                if not pd.isna(row["T1_F2_SD"].values[0])
                else None
            )
            # If Target 1 is missing, try Target 2
            if F1_mean is None or F1_sd is None or F2_mean is None or F2_sd is None:
                F1_mean = (
                    row["T2_F1_Mean"].values[0]
                    if not pd.isna(row["T2_F1_Mean"].values[0])
                    else None
                )
                F1_sd = (
                    row["T2_F1_SD"].values[0]
                    if not pd.isna(row["T2_F1_SD"].values[0])
                    else None
                )
                F2_mean = (
                    row["T2_F2_Mean"].values[0]
                    if not pd.isna(row["T2_F2_Mean"].values[0])
                    else None
                )
                F2_sd = (
                    row["T2_F2_SD"].values[0]
                    if not pd.isna(row["T2_F2_SD"].values[0])
                    else None
                )
            # Only proceed if all values are present
            if None in (F1_mean, F1_sd, F2_mean, F2_sd):
                FormantData.debug_print(
                    f"Insufficient ground truth for dynamic ceiling for phoneme={phoneme_timestamp.phoneme}"
                )
                return
            # Check condition
            if (F2_mean - F2_sd) - (F1_mean + F1_sd) <= 1000 and F2_mean < 2000:
                # Insert formant data points using praat_formant_max_freq=5000
                n_points = None
                start_time = np.max(0.0, phoneme_timestamp.start - 0.05)
                end_time = np.min(
                    self.sound.get_total_duration(), phoneme_timestamp.end + 0.05
                )
                FormantData.debug_print(
                    f"Inserting formant data points for phoneme={phoneme_timestamp.phoneme} with dynamic ceiling."
                )
                formant_times, formant_values = self.insert_formant_data_points(
                    sound=self.sound,
                    start_time=start_time,
                    end_time=end_time,
                    n_points=n_points,
                    praat_formant_max_freq=5000,
                )
                if formant_values.size > 0:
                    FormantData.debug_print(
                        f"Inserted {formant_values.shape[0]} formant values for time range {start_time}-{end_time}s."
                    )
                else:
                    FormantData.debug_print(
                        f"No formant data points generated for phoneme={phoneme_timestamp.phoneme} (skipping insertion)"
                    )
            else:
                FormantData.debug_print(
                    f"Dynamic frequency ceiling condition not met for phoneme={phoneme_timestamp.phoneme}"
                )
        except Exception as e:
            FormantData.debug_print(
                f"Error in generate_formant_data_points_with_dynamic_frequency_ceiling: {e}"
            )

    def find_formant_target_for_phoneme(
        self,
        phoneme_timestamp: PhonemeTimestamp,
        cleaning_algo: Literal["viterbi"] | None = "viterbi", # "median" not used
        use_ground_truth: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not os.path.exists(self.landmark_info_filepath):
            raise FileNotFoundError(
                f"Landmark info file not found: {self.landmark_info_filepath}"
            )
        if (
            phoneme_timestamp.parent_sentence_timestamp is None
            or phoneme_timestamp.parent_word_timestamp is None
        ):
            FormantData.debug_print(
                f"ERROR: Phoneme '{phoneme_timestamp.phoneme}' is missing parent sentence or word information."
            )
            return None, None  # Early return to prevent further processing
        else:
            FormantData.debug_print(
                f"DEBUG: Finding formant target for phoneme '{phoneme_timestamp.phoneme}' in word '{phoneme_timestamp.parent_word_timestamp.word}' sentence '{phoneme_timestamp.parent_sentence_timestamp.sentence}'"
            )
        landmark_info_pd = pd.read_csv(self.landmark_info_filepath, header=0)
        landmark_info_pd.columns = (
            landmark_info_pd.columns.str.strip()
        )  # Ensure no leading/trailing spaces in column names
        aus_brit_phoneme_mapping_pd = pd.read_csv(
            self.aus_brit_phoneme_mapping_filepath, header=0
        )
        aus_brit_phoneme_mapping_pd.columns = (
            aus_brit_phoneme_mapping_pd.columns.str.strip()
        )  # Ensure no leading/trailing spaces in column names
        # Fuzzy match for sentence
        all_sentences = landmark_info_pd["Sentence"].astype(str).tolist()
        query_sentence = str(phoneme_timestamp.parent_sentence_timestamp.sentence)
        FormantData.debug_print(f"DEBUG: Looking for sentence: '{query_sentence}'")
        FormantData.debug_print(
            f"DEBUG: Available sentences: {all_sentences[:5]}..."
        )  # Show first 5 sentences
        best_match = difflib.get_close_matches(
            query_sentence, all_sentences, n=1, cutoff=0.7
        )
        if best_match:
            matched_sentence = best_match[0]
            FormantData.debug_print(f"DEBUG: Found fuzzy match: '{matched_sentence}'")
        else:
            matched_sentence = query_sentence  # fallback to original
            FormantData.debug_print(
                f"DEBUG: No fuzzy match found, using original: '{matched_sentence}'"
            )
        # matched_sentence = query_sentence  # Use exact match for now
        vowel_info = (
            landmark_info_pd[
                [
                    "Word",
                    "Vowel",
                    "No. of Targets",
                    "DefaultTarget1 (% of total vowel duration)",
                    "DefaultTarget1Start (% of total vowel duration)",
                    "DefaultTarget1End (% of total vowel duration)",
                    "FormantTarget1",
                    "DefaultTarget2 (% of total vowel duration)",
                    "DefaultTarget2Start (% of total vowel duration)",
                    "DefaultTarget2End (% of total vowel duration)",
                    "FormantTarget2",
                ]
            ]
            .where(landmark_info_pd["Sentence"] == matched_sentence)
            .dropna(axis=0, how="all")
        )
        vowel_info = vowel_info.reset_index(drop=True)
        FormantData.debug_print(
            f"DEBUG: Found {len(vowel_info)} vowel entries for sentence '{matched_sentence}'"
        )
        vowel_mapping = (
            aus_brit_phoneme_mapping_pd[
                ["Word", "Australian_English", "British_English"]
            ]
            .where(aus_brit_phoneme_mapping_pd["Sentence"] == matched_sentence)
            .dropna(axis=0, how="all")
        )
        vowel_mapping = vowel_mapping.reset_index(drop=True)

        # --- Begin new logic for more vowels/words based on CSV ---
        # Helper function for midpoint
        def get_median(start, end):
            search_range_start_idx = np.where(self.formant_times >= start)[0]
            search_range_end_idx = np.where(self.formant_times <= end)[0]

            if search_range_start_idx.size == 0 or search_range_end_idx.size == 0:
                search_range_start_idx = np.array(
                    [np.argmin(np.abs(self.formant_times - start))]
                )
                search_range_end_idx = np.array(
                    [np.argmin(np.abs(self.formant_times - end))]
                )
                if search_range_start_idx.size == 0 or search_range_end_idx.size == 0:
                    return None, None

            # Use the first valid start index and last valid end index
            start_idx = search_range_start_idx[0]
            end_idx = search_range_end_idx[-1]

            mid_idx = (start_idx + end_idx) / 2
            t = np.mean(
                self.formant_times[int(np.floor(mid_idx)) : int(np.ceil(mid_idx)) + 1]
            )
            # Remove NaN values from each row before calculating median
            v = np.zeros(self.formant_values.shape[1])
            for i in range(self.formant_values.shape[1]):
                valid_formant_values = self.formant_values[start_idx : end_idx + 1, i]
                valid_formant_values = valid_formant_values[
                    ~np.isnan(valid_formant_values)
                ]
                if valid_formant_values.size > 0:
                    v[i] = np.median(valid_formant_values)
                else:
                    v[i] = np.nan  # or some other placeholder for no data
            return t, v

        # Helper for max/min formant N
        def get_extreme_formant_time(indices, mode: str, formant_idx: int):
            if indices.size == 0:
                return None, None
            # Ignore NaN values in the specified formant index
            valid_indices = indices[
                ~np.isnan(self.formant_values[indices, formant_idx])
            ]
            if valid_indices.size == 0:
                return None, None
            if mode == "max":
                temp_idx = np.argmax(self.formant_values[valid_indices, formant_idx])
                idx = np.where(indices == valid_indices[temp_idx])[0][0]
            else:
                temp_idx = np.argmin(self.formant_values[valid_indices, formant_idx])
                idx = np.where(indices == valid_indices[temp_idx])[0][0]
            t = self.formant_times[indices[idx]]
            v = self.formant_values[indices[idx]]
            return t, v

        def get_extreme_formant_difference(
            indices, mode: str, formant_idx_1: int, formant_idx_2: int
        ):
            if indices.size == 0:
                return None, None
            valid_indices_idx_1 = indices[
                ~np.isnan(self.formant_values[indices, formant_idx_1])
            ]
            valid_indices_idx_2 = indices[
                ~np.isnan(self.formant_values[indices, formant_idx_2])
            ]
            valid_indices = np.intersect1d(valid_indices_idx_1, valid_indices_idx_2)
            if mode == "max":
                temp_idx = np.argmax(
                    self.formant_values[valid_indices, formant_idx_1]
                    - self.formant_values[valid_indices, formant_idx_2]
                )
                idx = np.where(indices == valid_indices[temp_idx])[0][0]
            else:
                temp_idx = np.argmin(
                    self.formant_values[valid_indices, formant_idx_1]
                    - self.formant_values[valid_indices, formant_idx_2]
                )
                idx = np.where(indices == valid_indices[temp_idx])[0][0]
            t = self.formant_times[indices[idx]]
            v = self.formant_values[indices[idx]]
            return t, v

        def process_target(
            target_type, target_pct, target_pct_start, target_pct_end, phoneme_timestamp
        ):
            target_type = target_type.strip().lower()
            target_pct = FormantData.safe_float_convert(target_pct)
            target_pct_start = FormantData.safe_float_convert(target_pct_start)
            target_pct_end = FormantData.safe_float_convert(target_pct_end)
            search_range_start = phoneme_timestamp.start + target_pct_start / 100.0 * (
                phoneme_timestamp.end - phoneme_timestamp.start
            )
            search_range_end = phoneme_timestamp.start + target_pct_end / 100.0 * (
                phoneme_timestamp.end - phoneme_timestamp.start
            )
            FormantData.debug_print(
                f"DEBUG: Processing target type='{target_type}', pct={target_pct}%, search_range=[{search_range_start:.4f}s, {search_range_end:.4f}s]"
            )

            if target_type == "time":
                t = phoneme_timestamp.start + target_pct / 100.0 * (
                    phoneme_timestamp.end - phoneme_timestamp.start
                )
                _, v = get_median(search_range_start, search_range_end)
                FormantData.debug_print(
                    f"DEBUG: [TIME target] Appending target: timestamp={t:.4f}s, formants={v}"
                )
                phoneme_timestamp.formant_targets.append(
                    FormantTarget(timestamp=t, targets=v, target_line=None)
                )
                return True
            parts = target_type.split("_")
            if parts[0] in ["min", "max"] and len(parts) > 1 and parts[1].isdigit():
                formant_idx = int(parts[1]) - 1
                mode = parts[0]
                search_range_indices = np.where(
                    (self.formant_times >= search_range_start)
                    & (self.formant_times <= search_range_end)
                )[0]
                t, v = get_extreme_formant_time(search_range_indices, mode, formant_idx)
                if t is not None:
                    FormantData.debug_print(
                        f"DEBUG: [{mode.upper()}_F{formant_idx+1} target] Appending target: timestamp={t:.4f}s, formants={v}"
                    )
                    phoneme_timestamp.formant_targets.append(
                        FormantTarget(timestamp=t, targets=v, target_line=None)
                    )
                else:
                    FormantData.debug_print(
                        f"DEBUG: [{mode.upper()}_F{formant_idx+1} target] Could not find target (returned None)"
                    )
                return True
            elif (
                parts[0] in ["min", "max"]
                and len(parts) > 2
                and parts[1] == "diff"
                and parts[2].isdigit()
                and parts[3].isdigit()
            ):
                formant_idx_1 = int(parts[2]) - 1
                formant_idx_2 = int(parts[3]) - 1
                mode = parts[0]
                search_range_indices = np.where(
                    (self.formant_times >= search_range_start)
                    & (self.formant_times <= search_range_end)
                )[0]
                t, v = get_extreme_formant_difference(
                    search_range_indices, mode, formant_idx_1, formant_idx_2
                )
                if t is not None:
                    FormantData.debug_print(
                        f"DEBUG: [{mode.upper()}_DIFF_F{formant_idx_1+1}_F{formant_idx_2+1} target] Appending target: timestamp={t:.4f}s, formants={v}"
                    )
                    phoneme_timestamp.formant_targets.append(
                        FormantTarget(timestamp=t, targets=v, target_line=None)
                    )
                else:
                    FormantData.debug_print(
                        f"DEBUG: [{mode.upper()}_DIFF_F{formant_idx_1+1}_F{formant_idx_2+1} target] Could not find target (returned None)"
                    )
                return True
            elif parts[0] == "mid":
                t, v = get_median(search_range_start, search_range_end)
                if t is not None:
                    FormantData.debug_print(
                        f"DEBUG: [MID target] Appending target: timestamp={t:.4f}s, formants={v}"
                    )
                    phoneme_timestamp.formant_targets.append(
                        FormantTarget(timestamp=t, targets=v, target_line=None)
                    )
                else:
                    FormantData.debug_print(
                        f"DEBUG: [MID target] Could not find target (returned None)"
                    )
                return True
            return False

        # Get word and vowel for this phoneme
        vowel = phoneme_timestamp.phoneme
        word = phoneme_timestamp.parent_word_timestamp.word
        FormantData.debug_print(f"DEBUG: Processing phoneme '{vowel}' in word '{word}'")
        word = word.replace(
            "'", "’"
        )  # Replace single quotes with proper UTF-8 apostrophe
        FormantData.debug_print(f"DEBUG: Single quote replaced in word: '{word}'")
        FormantData.debug_print(
            f"DEBUG: Vowel mapping data has {len(vowel_mapping)} entries"
        )
        FormantData.debug_print("DEBUG: Available vowel mappings:")
        for idx, row in vowel_mapping.iterrows():
            FormantData.debug_print(
                f"  Word: '{row['Word']}', British: '{row['British_English']}', Australian: '{row['Australian_English']}'"
            )
        word_lower = word.lower()
        mapping_mask_british = (vowel_mapping["Word"].str.lower() == word_lower) & (
            vowel_mapping["British_English"] == vowel
        )
        mapping_mask_australian = (vowel_mapping["Word"].str.lower() == word_lower) & (
            vowel_mapping["Australian_English"] == vowel
        )
        vowel_aus_british = vowel_mapping[mapping_mask_british]
        vowel_aus_australian = vowel_mapping[mapping_mask_australian]
        FormantData.debug_print(
            f"DEBUG: Found {len(vowel_aus_british)} British->Australian mappings for word '{word}' and phoneme '{vowel}'"
        )
        FormantData.debug_print(
            f"DEBUG: Found {len(vowel_aus_australian)} Australian mappings for word '{word}' and phoneme '{vowel}'"
        )
        if not vowel_aus_british.empty:
            original_vowel = phoneme_timestamp.phoneme
            vowel = vowel_aus_british["Australian_English"].values[0]
            phoneme_timestamp.phoneme = vowel  # Update phoneme in timestamp
            FormantData.debug_print(
                f"DEBUG: Mapped British phoneme '{original_vowel}' to Australian phoneme: '{vowel}'"
            )
        elif not vowel_aus_australian.empty:
            FormantData.debug_print(
                f"DEBUG: Phoneme '{vowel}' is already in Australian format"
            )
        else:
            FormantData.debug_print(
                f"DEBUG: No phoneme mapping found for '{vowel}' in word '{word}', using as-is"
            )
        while len(phoneme_timestamp.formant_targets) > 0:
            formant_target = phoneme_timestamp.formant_targets.pop()
            # Remove the target line from the plot
            if formant_target.target_line:
                formant_target.target_line.setParent(None)
                target_line = formant_target.target_line
                formant_target.target_line = None
                target_line.deleteLater()
        # Find the row in vowel_info for this word/vowel
        query_word = word
        FormantData.debug_print(f"DEBUG: Original word: '{query_word}'")
        # Try case-insensitive matching first
        query_word_lower = query_word.lower()
        available_words_lower = [
            w.lower() for w in vowel_info["Word"].astype(str).tolist()
        ]
        available_words_orig = vowel_info["Word"].astype(str).tolist()

        # Find case-insensitive match
        case_insensitive_match = None
        for i, word_lower in enumerate(available_words_lower):
            if word_lower == query_word_lower:
                case_insensitive_match = available_words_orig[i]
                break

        if case_insensitive_match:
            query_word = case_insensitive_match
            FormantData.debug_print(
                f"DEBUG: Found case-insensitive match: '{query_word}'"
            )
        else:
            # Fallback to fuzzy matching
            matched_words = difflib.get_close_matches(
                query_word, vowel_info["Word"].astype(str).tolist(), n=1, cutoff=0.7
            )
            if matched_words:
                query_word = matched_words[0]
                FormantData.debug_print(
                    f"DEBUG: Fuzzy matched word '{word}' to '{query_word}'"
                )
            else:
                FormantData.debug_print(f"DEBUG: No match found for word '{word}'")
        row = vowel_info[
            (vowel_info["Vowel"] == vowel)
            & (vowel_info["Word"].str.lower() == query_word.lower())
        ]
        FormantData.debug_print(
            f"DEBUG: Looking for vowel '{vowel}' in word '{query_word}', found {len(row)} matches"
        )
        if len(vowel_info) > 0:
            FormantData.debug_print("DEBUG: Available combinations in vowel_info:")
            for idx, row_debug in vowel_info.iterrows():
                FormantData.debug_print(
                    f"  Word: '{row_debug['Word']}', Vowel: '{row_debug['Vowel']}'"
                )
        if row.empty:
            FormantData.debug_print(
                f"DEBUG: No formant target info found for phoneme '{vowel}' in word '{query_word}' sentence '{matched_sentence}'"
            )
            return None, None  # No info for this vowel/word
        if cleaning_algo == "viterbi":
            formant_times, formant_values = (
                self.fix_formant_values_outliers_for_phoneme_with_viterbi(
                    phoneme_timestamp, use_ground_truth=use_ground_truth
                )
            )
        # elif cleaning_algo == "median": # "median" not used
        #     formant_times, formant_values = (
        #         self.fix_formant_values_outliers_for_phoneme_with_zscore_shift(
        #             phoneme_timestamp, threshold=3.5
        #         )
        #     )
        else:
            target_indices = np.where(
                (self.formant_times >= phoneme_timestamp.start)
                & (self.formant_times <= phoneme_timestamp.end)
            )[0]
            phoneme_formant_times, phoneme_formant_values = (
                self.formant_times[target_indices],
                self.formant_values[target_indices],
            )
            formant_times, formant_values = (
                phoneme_formant_times,
                phoneme_formant_values,
            )
        row = row.iloc[0]
        # Get targets
        target1_type = str(row["FormantTarget1"]).strip()
        target2_type = str(row["FormantTarget2"]).strip()

        # Check for NaN values and convert to empty string
        if target1_type.lower() in ["nan", "none"]:
            target1_type = ""
        if target2_type.lower() in ["nan", "none"]:
            target2_type = ""

        # Only use second value if '/' present
        if "/" in target1_type:
            target1_type = target1_type.split("/")[1].strip()
        if "/" in target2_type:
            target2_type = target2_type.split("/")[1].strip()

        # Helper function to safely convert to float
        def safe_float_convert(value, field_name):
            try:
                if pd.isna(value) or str(value).strip().lower() in ["nan", "none", ""]:
                    return None
                return float(value)
            except (ValueError, TypeError):
                FormantData.debug_print(
                    f"DEBUG: Cannot convert '{value}' to float for field '{field_name}'"
                )
                return None

        # Target 1
        if target1_type and target1_type.lower() not in ["nan", "none", ""]:
            target1_pct = safe_float_convert(
                row["DefaultTarget1 (% of total vowel duration)"], "target1_pct"
            )
            target1_pct_start = safe_float_convert(
                row["DefaultTarget1Start (% of total vowel duration)"],
                "target1_pct_start",
            )
            target1_pct_end = safe_float_convert(
                row["DefaultTarget1End (% of total vowel duration)"], "target1_pct_end"
            )

            if (
                target1_pct is not None
                and target1_pct_start is not None
                and target1_pct_end is not None
            ):
                process_target(
                    target1_type,
                    target1_pct,
                    target1_pct_start,
                    target1_pct_end,
                    phoneme_timestamp,
                )
            else:
                FormantData.debug_print(
                    "DEBUG: Skipping target1 due to missing numeric values"
                )

        # Target 2
        if target2_type and target2_type.lower() not in ["nan", "none", ""]:
            target2_pct = safe_float_convert(
                row["DefaultTarget2 (% of total vowel duration)"], "target2_pct"
            )
            target2_pct_start = safe_float_convert(
                row["DefaultTarget2Start (% of total vowel duration)"],
                "target2_pct_start",
            )
            target2_pct_end = safe_float_convert(
                row["DefaultTarget2End (% of total vowel duration)"], "target2_pct_end"
            )

            if (
                target2_pct is not None
                and target2_pct_start is not None
                and target2_pct_end is not None
            ):
                process_target(
                    target2_type,
                    target2_pct,
                    target2_pct_start,
                    target2_pct_end,
                    phoneme_timestamp,
                )
            else:
                FormantData.debug_print(
                    "DEBUG: Skipping target2 due to missing numeric values"
                )

        # Final summary of all targets
        FormantData.debug_print(
            f"DEBUG: ========== FINAL TARGET SUMMARY for phoneme '{phoneme_timestamp.phoneme}' =========="
        )
        FormantData.debug_print(
            f"DEBUG: Total number of targets: {len(phoneme_timestamp.formant_targets)}"
        )
        for i, target in enumerate(phoneme_timestamp.formant_targets):
            FormantData.debug_print(
                f"DEBUG:   Target {i+1}: timestamp={target.timestamp:.4f}s, formants={target.targets}"
            )
        FormantData.debug_print(
            f"DEBUG: ====================================================================="
        )

        return formant_times, formant_values

    def formant_targets_to_descriptor_text(
        self, phoneme_timestamp: PhonemeTimestamp
    ) -> str | None:
        if not phoneme_timestamp:
            return None
        phoneme = phoneme_timestamp.phoneme
        start_time = phoneme_timestamp.start
        end_time = phoneme_timestamp.end
        targets_list = []
        for i, formant_target in enumerate(phoneme_timestamp.formant_targets):
            if formant_target.timestamp:
                targets_list.append(f"Target {i+1} at {formant_target.timestamp:.3f}s")
            else:
                targets_list.append(f"Target {i+1} timestamp not set")
        if not targets_list:
            FormantData.debug_print(
                f"No targets set for {start_time:.3f} s - {end_time:.3f} s: {phoneme_timestamp.parent_word_timestamp.word if phoneme_timestamp.parent_word_timestamp else ""} | {phoneme}"
            )
            return None
        word_text = ensure_utf8_display(
            phoneme_timestamp.parent_word_timestamp.word
            if phoneme_timestamp.parent_word_timestamp
            else ""
        )
        phoneme_text = ensure_utf8_display(phoneme)
        descriptor_text = f"{start_time:.3f} s - {end_time:.3f} s: {word_text} | {phoneme_text} --- {', '.join(targets_list)}"
        return descriptor_text

    def sync_formant_targets_with_target_lines(
        self, phoneme_timestamp: PhonemeTimestamp
    ):
        if not phoneme_timestamp:
            return
        for formant_target in phoneme_timestamp.formant_targets:
            if formant_target.target_line:
                t = formant_target.target_line.value()
                try:
                    v = self.get_formant_values_at_time(t)
                    closest_time_idx = np.argmin(np.abs(self.formant_times - t))
                    closest_time = self.formant_times[closest_time_idx]
                    formant_target.timestamp = closest_time
                    formant_target.targets = v
                    formant_target.target_line.setValue(
                        closest_time
                    )  # Snap line to closest time
                    FormantData.debug_print(
                        f"DEBUG: Synced target line at {t}s for {phoneme_timestamp.phoneme} to formant values: {v}"
                    )
                except ValueError as e:
                    FormantData.debug_print(
                        f"ERROR: Syncing target line at {t}s for {phoneme_timestamp.phoneme} --- {e}"
                    )
                    continue

    def clear(self):
        self.formant_times = np.array([])
        self.formant_values = np.array([])
        if os.path.exists(FormantData.FORMANT_TARGET_RECORD_FILE):
            os.remove(FormantData.FORMANT_TARGET_RECORD_FILE)
        if os.path.exists(FormantData.DEBUG_LOG_FILE):
            os.remove(FormantData.DEBUG_LOG_FILE)
        if os.path.exists(FormantData.OUTLIER_FIX_LOG_FILE):
            os.remove(FormantData.OUTLIER_FIX_LOG_FILE)
