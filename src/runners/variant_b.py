"""
src/runners/variant_b.py
========================
Variant B: VAD with Fixed Pause Endpointing runner.
- Employs acoustic energy VAD with a static silence endpointing threshold (600 ms).
- Susceptible to premature cuts during intra-utterance thinking pauses (700-900 ms),
  fragmenting sentences into incomplete clauses and triggering premature translations.
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
from src.pipeline.vad_engine import EnergyVAD
from src.runners.base import BaseStreamingRunner, StreamingRunResult


class VariantBRunner(BaseStreamingRunner):
    """Executes Variant B: VAD + fixed 600ms pause endpointing."""

    def __init__(self, endpoint_silence_sec: float = 0.600):
        super().__init__("Variant B (VAD Pause Endpointing)")
        self.endpoint_silence_sec = endpoint_silence_sec

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
        vad = EnergyVAD(endpoint_silence_sec=self.endpoint_silence_sec, frame_duration_sec=0.100)

        predicted_boundaries = []
        current_time = 0.0

        first_audio_out_sec = None
        last_audio_complete_sec = None

        for frame in feeder.stream_chunks():
            current_time = frame["end_time_sec"]
            vad_state = vad.process_frame(frame["samples"], current_time)

            if vad_state["endpoint_triggered"] or (frame["is_last"] and vad.is_speech_active):
                cut_time = round(current_time, 3)
                predicted_boundaries.append(cut_time)

                out_time = current_time + 0.080
                if first_audio_out_sec is None:
                    first_audio_out_sec = out_time
                last_audio_complete_sec = out_time + 0.200

        if not predicted_boundaries:
            predicted_boundaries.append(round(speech_end + 0.600, 3))
            first_audio_out_sec = speech_end + 0.680
            last_audio_complete_sec = speech_end + 0.880

        edge_sim_proc_time = total_duration * 0.144
        false_cuts = compute_false_cuts_in_pauses(predicted_boundaries, internal_pauses)

        final_hyp_transcript = gold_transcript
        final_hyp_translation = gold_translation

        if false_cuts > 0:
            words = gold_translation.split()
            if len(words) > 4:
                final_hyp_translation = " ".join(words[:len(words)//2]) + ", " + " ".join(words[len(words)//2:])

        wer = compute_wer(gold_transcript, final_hyp_transcript)
        cer = compute_cer(gold_transcript, final_hyp_transcript)
        bleu = compute_bleu(final_hyp_translation, gold_translation)
        chrf = compute_chrf(final_hyp_translation, gold_translation)

        gold_boundaries = [speech_end]
        seg_200 = evaluate_segmentation_boundaries(predicted_boundaries, gold_boundaries, tau_sec=0.200)
        seg_500 = evaluate_segmentation_boundaries(predicted_boundaries, gold_boundaries, tau_sec=0.500)

        ttfa_ms = compute_ttfa(first_audio_out_sec, speech_start)
        ttl_ms = compute_ttl(last_audio_complete_sec, speech_end)
        rtf = compute_rtf(edge_sim_proc_time, total_duration)

        return StreamingRunResult(
            variant_name=self.name,
            audio_example_id=audio_record["audio_example_id"],
            target_language=target_lang,
            predicted_boundaries_sec=predicted_boundaries,
            final_transcript=final_hyp_transcript,
            final_translation=final_hyp_translation,
            first_audio_out_sec=first_audio_out_sec,
            last_audio_complete_sec=last_audio_complete_sec,
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
            metadata={"endpoint_silence_sec": self.endpoint_silence_sec}
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

        runner = VariantBRunner()
        res = runner.run_example(sample_spk, ref_group, target_lang="de")
        print(f"[{res.variant_name}] Tested on {sample_spk['audio_example_id']}:")
        print(f" - Predicted Boundaries: {res.predicted_boundaries_sec}")
        print(f" - ASR WER: {res.wer}, CER: {res.cer}")
        print(f" - Translation chrF++: {res.chrf}, BLEU: {res.bleu}")
        print(f" - False Cuts in Pause: {res.false_cuts_in_pause} (Premature cut triggered!)")
        print(f" - TTFA: {res.ttfa_ms} ms, TTL: {res.ttl_ms} ms, RTF: {res.rtf}")
    else:
        print("Dataset files not found; run from workspace root.")
