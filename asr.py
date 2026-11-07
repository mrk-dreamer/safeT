# -*- coding: utf-8 -*-
import parselmouth
from praatio import textgrid
import torch
import whisperx
import os
import jiwer
from rapidfuzz import fuzz
from helper import SentenceTimestamp
import numpy as np
import json

class WhisperX_ASR:
    def __init__(self, whisper_arch:str="tiny", *, language:str="en", timestamp_tolerance:float=0.05):
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if self.device=="cuda" else "int8"
        self.model = whisperx.load_model(whisper_arch=whisper_arch, device=self.device, compute_type=compute_type, language=language)
        self.align_model, self.metadata = whisperx.load_align_model(language_code=language, device=self.device)
        self.timestamp_tolerance = timestamp_tolerance
        self.chunk_size_choices = [x for x in range(20, 31, 3)]

    def set_asr_target(self, audio_path:str, *, stimulus_path:str, textgrid_path:str|None=None):
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
        if not self.stimulus_path or not os.path.exists(self.stimulus_path):
            raise ValueError("Error: No valid stimulus path provided.")
        stimulus_sentences = []
        with open(self.stimulus_path, "r", encoding='utf-8') as f:
            for line in f:
                stimulus_sentences.append(line.strip())
        if len(stimulus_sentences) == 0:
            raise ValueError("Error: No valid sentences found in the stimulus file.")
        return stimulus_sentences

    def calculate_wer(self, *, hypothesis, reference):
        """ Compute Word Error Rate (WER) """
        return jiwer.wer(reference, hypothesis)

    def fuzzy_match_score(self, *, hypothesis, reference):
        """ Compute similarity score using fuzzy matching """
        return fuzz.ratio(hypothesis, reference) / 100.0  # Normalize to [0,1]

    def refine_match_using_shaving(self, segment, target_text, *, wer_threshold=0.6, similarity_threshold=0.6):
        """
        Start with the full segment and iteratively remove words from the front and back 
        until the fuzzy match score stops improving.
        """
        words = segment['words']
        segment_words = [w['word'].strip(",.") for w in words]
        
        best_match = (0, len(segment_words) - 1)
        hypothesis_text = " ".join(segment_words)
        best_score = self.fuzzy_match_score(hypothesis=hypothesis_text, reference=target_text)
        best_wer = self.calculate_wer(hypothesis=hypothesis_text, reference=target_text)

        # if best_score < similarity_threshold and best_wer > wer_threshold:
        #     return None  # No strong match found

        # Try shaving words from front and back
        start_idx, end_idx = best_match
        improved = True

        while improved and end_idx > start_idx:
            improved = False
            
            # Try shaving from front
            hypothesis_text = " ".join(segment_words[start_idx + 1:end_idx + 1])
            new_score = self.fuzzy_match_score(hypothesis=hypothesis_text, reference=target_text)
            new_wer = self.calculate_wer(hypothesis=hypothesis_text, reference=target_text)
            if new_score > best_score and new_wer <= best_wer:  # Keep trimming if score improves
                best_score = new_score
                best_wer = new_wer
                start_idx += 1
                improved = True


            # Try shaving from back
            if start_idx < end_idx:
                hypothesis_text = " ".join(segment_words[start_idx:end_idx])
                new_score = self.fuzzy_match_score(hypothesis=hypothesis_text, reference=target_text)
                new_wer = self.calculate_wer(hypothesis=hypothesis_text, reference=target_text)
                if new_score > best_score and new_wer <= best_wer:  # Keep trimming if score improves
                    best_score = new_score
                    best_wer = new_wer
                    end_idx -= 1
                    improved = True

        shaved_segment = {
            'start': words[start_idx]['start'],
            'end': words[end_idx]['end'],
            'text': " ".join(segment_words[start_idx:end_idx + 1]),
            'words': words[start_idx:end_idx + 1]
        }

        return words[start_idx]['start'], words[end_idx]['end'], best_score, best_wer, shaved_segment

    def find_last_occurrence(self, segments, target_text, *, wer_threshold=0.6, similarity_threshold=0.6):
        """
        Finds the last occurrence of target_text in transcriptions using WER and fuzzy matching.
        """
        if len(segments) == 0:
            raise ValueError("Error: No segments found in the transcription.")
        
        candidate_segments = []

        for seg in segments:
            # Refine the segment using word shaving
            refined_seg_start, refined_seg_end, refined_seg_score, refined_seg_wer, refined_seg = self.refine_match_using_shaving(seg, target_text, wer_threshold=wer_threshold, similarity_threshold=similarity_threshold)
            wer = self.calculate_wer(hypothesis=refined_seg['text'], reference=target_text)
            similarity_score = self.fuzzy_match_score(hypothesis=refined_seg['text'], reference=target_text)

            if wer < wer_threshold or similarity_score > similarity_threshold:
                candidate_segments.append((refined_seg, similarity_score, wer))

        if not candidate_segments:
            return None  # No suitable match found

        # Save candidate_segments to JSON file for debugging (append mode)
        json_filename = f"candidate_segments_{os.path.basename(self.audio_path).replace('.wav', '')}.json"
        json_filepath = os.path.join(os.path.dirname(self.audio_path), json_filename)
        
        # Convert candidate_segments to a serializable format
        new_entry = {
            "target_text": target_text,
            "timestamp": str(np.datetime64('now')),
            "candidate_segments": []
        }
        
        for seg, similarity_score, wer in candidate_segments:
            new_entry["candidate_segments"].append({
                "segment": seg,
                "similarity_score": similarity_score,
                "wer": wer
            })
        
        # Load existing data if file exists, otherwise start with empty list
        existing_data = []
        if os.path.exists(json_filepath):
            try:
                with open(json_filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []
        
        # Append new entry to existing data
        existing_data.append(new_entry)
        
        # Save updated data back to file
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        print(f"Candidate segments appended to: {json_filepath}")

        # # Sort by start time and take the last occurrence
        # candidate_segments.sort(key=lambda x: x[0]['start'])
        # last_segment = candidate_segments[-1]
        best_segment = max(candidate_segments, key=lambda x: (x[1], -x[2], x[0]['start']))  # Maximize similarity score, minimize WER, maximize start time (chronologically later)
        match_start = best_segment[0]['start']
        match_end = best_segment[0]['end']

        # # Refine the match using word shaving
        # refined_result = self.refine_match_using_shaving(last_segment[0], target_text, wer_threshold=wer_threshold, similarity_threshold=similarity_threshold)
        
        # if refined_result:
        #     match_start, match_end, final_score, final_wer, final_segment = refined_result
        # else:
        #     match_start, match_end, final_score, final_wer = last_segment[0]['start'], last_segment[0]['end'], last_segment[1], last_segment[2]

        # audio_duration = parselmouth.Sound(self.audio_path).get_total_duration()
        # match_start = max(0, match_start - self.timestamp_tolerance)
        # match_end = min(audio_duration, match_end + self.timestamp_tolerance)
        return {
            "start": match_start, 
            "end": match_end, 
            "text": best_segment[0]['text'],
            "similarity": best_segment[1],
            "wer": best_segment[2],
        }


    def save_sentences_as_textgrid(self, sentence_timestamps:list[SentenceTimestamp]):
        if not sentence_timestamps:  # Check if results are empty
            raise ValueError("Error: The timestamps list is empty.")
        if not self.audio_path or not os.path.exists(self.audio_path):
            raise ValueError("Error: No valid audio path provided.")
        if not self.textgrid_path:
            raise ValueError("Error: No valid TextGrid path provided.")

        tg = textgrid.Textgrid()  # Create an empty TextGrid
        tg_entries = []

        for timestamp in sentence_timestamps:
            start, end, text = timestamp.start, timestamp.end, timestamp.sentence

            # Ensure timestamps are valid
            if start is not None and end is not None and end > start:
                tg_entries.append((start, end, text))  # Correct order (start, end, label)

        if not tg_entries:  # Check if there are no valid entries
            raise ValueError("Error: No valid timestamps found.")

        audio_duration = parselmouth.Sound(self.audio_path).get_total_duration()
        tier = textgrid.IntervalTier(name="sentences", entries=tg_entries, minT=0, maxT=audio_duration)

        tg.addTier(tier)
        # Force UTF-8 encoding when saving TextGrid files
        try:
            tg.save(self.textgrid_path, format="long_textgrid", includeBlankSpaces=True)
        except UnicodeEncodeError:
            # If there's an encoding error, try to encode the text manually
            print("Warning: Unicode encoding issue detected. Attempting to fix...")
            for tier in tg.tierList:
                for interval in tier.intervalList:
                    if hasattr(interval, 'label') and interval.label:
                        # Ensure the label is properly encoded as UTF-8
                        interval.label = interval.label.encode('utf-8', errors='replace').decode('utf-8')
            tg.save(self.textgrid_path, format="long_textgrid", includeBlankSpaces=True)

    def remove_segment_overlaps(self, segments:list[dict], tolerance:float=0.05):
        """
        Remove overlap from segments.
        """
        if not segments:
            return []

        # Sort segments by start time
        segments.sort(key=lambda x: x['start'])
        non_overlapping_segments = [segments[0]]

        for current in segments[1:]:
            last = non_overlapping_segments[-1]
            if current['start'] > last['end'] + tolerance:
                non_overlapping_segments.append(current)
            else:
                # Boundaries at midpoint of overlap+/-(tolerance / 2)
                midpoint = (last['end'] + current['start']) / 2
                non_overlapping_segments[-1]['end'] = midpoint - (tolerance/2)
                non_overlapping_segments[-1]["words"][-1]['end'] = midpoint - (tolerance/2)  # Adjust last word end time
                current['start'] = midpoint + (tolerance/2)
                current['words'][0]['start'] = midpoint + (tolerance/2)  # Adjust first word start time
                non_overlapping_segments.append(current)

        return non_overlapping_segments

    def get_valid_sentence_timestamps(self, shuffle_chunk_size:bool=False):
        if not self.audio_path or not os.path.exists(self.audio_path):
            raise ValueError("Error: No valid audio path provided.")
        stimulus_sentences = self.get_stimulus_sentences()
        if shuffle_chunk_size:
            chunk_size = np.random.choice(self.chunk_size_choices, size=1, replace=False)[0]
            print(f"Randomly selected chunk size: {chunk_size}")
        else:
            chunk_size = 29
        asr_results = self.model.transcribe(self.audio_path, language="en", task="transcribe", chunk_size=chunk_size, print_progress=True)
        aligned_asr_results = whisperx.align(asr_results["segments"], self.align_model, self.metadata, self.audio_path, self.device, return_char_alignments=False, print_progress=True, combined_progress=True)
        segments = aligned_asr_results["segments"]
        non_overlapping_segments = self.remove_segment_overlaps(segments, tolerance=self.timestamp_tolerance)
        aligned_asr_results["segments"] = non_overlapping_segments

        # Save aligned ASR results to JSON file for analysis
        asr_json_filename = f"aligned_asr_segments_{os.path.basename(self.audio_path).replace('.wav', '')}.json"
        asr_json_filepath = os.path.join(os.path.dirname(self.audio_path), asr_json_filename)
        
        # Create entry with metadata
        asr_entry = {
            "audio_file": os.path.basename(self.audio_path),
            "timestamp": str(np.datetime64('now')),
            "chunk_size": chunk_size,
            "segments": aligned_asr_results["segments"]
        }
        
        # Load existing data if file exists, otherwise start with empty list
        existing_asr_data = []
        if os.path.exists(asr_json_filepath):
            try:
                with open(asr_json_filepath, 'r', encoding='utf-8') as f:
                    existing_asr_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_asr_data = []
        
        # Append new entry to existing data
        existing_asr_data.append(asr_entry)
        
        # Save updated data back to file
        with open(asr_json_filepath, 'w', encoding='utf-8') as f:
            json.dump(existing_asr_data, f, indent=2, ensure_ascii=False)
        print(f"Aligned ASR segments appended to: {asr_json_filepath}")

        valid_timestamps: list[SentenceTimestamp] = []
        for sentence in stimulus_sentences:
            match = self.find_last_occurrence(aligned_asr_results["segments"], sentence)
            if match:
                valid_timestamps.append(SentenceTimestamp(sentence=sentence, start=match['start'], end=match['end']))
            else:
                print(f"Warning: No match found for sentence '{sentence}'")
                valid_timestamps.append(SentenceTimestamp(sentence=sentence, start=None, end=None))
        
        return valid_timestamps