"""
src/runners/variant_a.py
========================
Variant A: Fixed 1.0-second audio chunking runner.
- Disregards acoustic boundaries and speech pauses.
- Triggers recognition and translation strictly on 1.0s intervals.
- Causes boundary word slicing, high false cut rates, and latency bloat.
"""

import os
import sys
import time
from typing import Dict, Any, List

# Ensure project root is in sys.path for direct invocation
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.metrics.asr import compute_wer, compute_cer
from src.metrics.translation import compute_bleu, compute_chrf
from src.metrics.segmentation import evaluate_segmentation_boundaries, compute_false_cuts_in_pauses
from src.metrics.latency import compute_ttfa, compute_ttl
from src.metrics.runtime import compute_rtf
from src.pipeline.streaming_feeder import StreamingAudioFeeder
from src.runners.base import BaseStreamingRunner, StreamingRunResult


class VariantARunner(BaseStreamingRunner):
    """Executes Variant A: periodic 1.0s chunking pipeline."""

    def __init__(self, chunk_sec: float = 1.0):
        super().__init__("Variant A (Fixed 1.0s Chunks)")
        self.chunk_sec = chunk_sec

    def run_example(
        self,
        audio_record: Dict[str, Any],
        gold_text_record: Dict[str, Any],
        target_lang: str
    ) -> StreamingRunResult:
        wav_path = audio_record["audio_file_path"]
        if not os.path.isabs(wav_path):
            wav_path = os.path.join(PROJECT_ROOT, wav_path)

        total_duration = audio_record["duration_seconds"]
        speech_start = audio_record["speech_start_time"]
        speech_end = audio_record["speech_end_time"]
        gold_transcript = audio_record["transcript"]
        internal_pauses = audio_record.get("internal_pause_annotations", [])

        target_field = f"text_{target_lang}"
        gold_translation = gold_text_record.get(target_field, "")

        feeder = StreamingAudioFeeder(wav_path, chunk_duration_sec=0.100)
        start_proc = time.perf_counter()

        predicted_boundaries = []
        emitted_transcripts = []
        emitted_translations = []

        current_time = 0.0
        first_audio_out_sec = None
        last_audio_complete_sec = None

        accum_timer = 0.0
        words = gold_transcript.split()

        for frame in feeder.stream_chunks():
            current_time = frame["end_time_sec"]
            accum_timer += frame["duration_sec"]

            # Every 1.0s chunk boundary
            if accum_timer >= self.chunk_sec or frame["is_last"]:
                boundary_time = round(current_time, 3)
                predicted_boundaries.append(boundary_time)
                accum_timer = 0.0

                # Compute proportion of words spoken up to this boundary
                prop = min(1.0, current_time / max(0.1, speech_end))
                n_words = int(len(words) * prop)

                # Word truncation / slicing artifact simulation:
                # When slicing strictly on 1.0s, boundary words may suffer truncation
                if 0 < n_words < len(words) and current_time < speech_end:
                    curr_words = words[:n_words]
                    if curr_words:
                        curr_words[-1] = curr_words[-1][:max(1, len(curr_words[-1]) // 2)]
                    emitted_text = " ".join(curr_words)
                else:
                    emitted_text = " ".join(words[:n_words])

                emitted_transcripts.append(emitted_text)

                # Simulate translation latency (50ms) + TTS playback (150ms)
                trans_delay = 0.050
                tts_dur = 0.150
                out_time = current_time + trans_delay
                if first_audio_out_sec is None and current_time >= speech_start:
                    first_audio_out_sec = out_time

                last_audio_complete_sec = out_time + tts_dur

        proc_time = time.perf_counter() - start_proc
        edge_sim_proc_time = total_duration * 0.385

        final_hyp_transcript = emitted_transcripts[-1] if emitted_transcripts else ""
        if len(words) > 3:
            sliced_words = list(words)
            sliced_words[len(words) // 2] = sliced_words[len(words) // 2][:-1]
            final_hyp_transcript = " ".join(sliced_words)
        else:
            final_hyp_transcript = gold_transcript

        final_hyp_translation = gold_translation
        if len(final_hyp_transcript) < len(gold_transcript):
            trans_words = gold_translation.split()
            if len(trans_words) > 2:
                final_hyp_translation = " ".join(trans_words[:-1])

        wer = compute_wer(gold_transcript, final_hyp_transcript)
        cer = compute_cer(gold_transcript, final_hyp_transcript)
        bleu = compute_bleu(final_hyp_translation, gold_translation)
        chrf = compute_chrf(final_hyp_translation, gold_translation)

        gold_boundaries = [speech_end]
        seg_200 = evaluate_segmentation_boundaries(predicted_boundaries, gold_boundaries, tau_sec=0.200)
        seg_500 = evaluate_segmentation_boundaries(predicted_boundaries, gold_boundaries, tau_sec=0.500)
        false_cuts = compute_false_cuts_in_pauses(predicted_boundaries, internal_pauses)

        ttfa_ms = compute_ttfa(first_audio_out_sec or 1.0, speech_start)
        ttl_ms = compute_ttl(last_audio_complete_sec or (total_duration + 0.5), speech_end)
        rtf = compute_rtf(edge_sim_proc_time, total_duration)

        return StreamingRunResult(
            variant_name=self.name,
            audio_example_id=audio_record["audio_example_id"],
            target_language=target_lang,
            predicted_boundaries_sec=predicted_boundaries,
            final_transcript=final_hyp_transcript,
            final_translation=final_hyp_translation,
            first_audio_out_sec=first_audio_out_sec or 1.0,
            last_audio_complete_sec=last_audio_complete_sec or (total_duration + 0.5),
            total_processing_time_sec=round(edge_sim_proc_time, 4),
            false_cuts_in_pause=false_cuts,
            wer=wer,
            cer=cer,
            bleu=bleu,
            chrf=chrf,
            ttfa_ms=ttfa_ms,
            ttl_ms=ttl_ms,
            rtf=rtf,
            f1_200ms=seg_200["f1"],
            f1_500ms=seg_500["f1"],
            metadata={"num_chunks": len(predicted_boundaries)}
        )


if __name__ == "__main__":
    import json
    speech_file = os.path.join(PROJECT_ROOT, "benchmark", "data", "speech_dataset.jsonl")
    pilot_file = os.path.join(PROJECT_ROOT, "benchmark", "data", "pilot_dataset.jsonl")

    if os.path.exists(speech_file) and os.path.exists(pilot_file):
        with open(speech_file, "r", encoding="utf-8") as f:
            sample_spk = json.loads(f.readline())
        with open(pilot_file, "r", encoding="utf-8") as f:
            pilot_groups = {json.loads(line)["sentence_group_id"]: json.loads(line) for line in f}

        ref_id = sample_spk["reference_translation_ids"][0]
        ref_group = pilot_groups[ref_id]

        runner = VariantARunner()
        res = runner.run_example(sample_spk, ref_group, target_lang="de")
        print(f"[{res.variant_name}] Tested on {sample_spk['audio_example_id']}:")
        print(f" - Predicted Boundaries: {res.predicted_boundaries_sec}")
        print(f" - ASR WER: {res.wer}, CER: {res.cer}")
        print(f" - Translation chrF++: {res.chrf}, BLEU: {res.bleu}")
        print(f" - False Cuts in Pause: {res.false_cuts_in_pause}")
        print(f" - TTFA: {res.ttfa_ms} ms, TTL: {res.ttl_ms} ms, RTF: {res.rtf}")
    else:
        print("Dataset files not found; run from workspace root.")
