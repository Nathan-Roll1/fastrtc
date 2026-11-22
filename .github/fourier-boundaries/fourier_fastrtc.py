"""Local English utterance recognition for FastRTC."""
from math import gcd
from threading import Lock

import numpy as np
from fastrtc import AdditionalOutputs, AlgoOptions, ReplyOnPause, Stream, audio_to_float32
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly
from transcribe_cpp import Model

MODEL_ID = "oruk/oruk-fourier-2b"
REVISION = "8bcce21695e4ba13569c48de6cedffcb8a7576f5"


class FourierSTT:
    def __init__(self):
        filename = hf_hub_download(MODEL_ID, "Fourier-2B-Q8_0.gguf", revision=REVISION)
        self.model = Model(filename, backend="cpu")
        self.lock = Lock()

    def stt(self, audio: tuple[int, np.ndarray]) -> str:
        rate, samples = audio
        samples = audio_to_float32(samples)
        if samples.ndim == 2:
            samples = samples.mean(axis=0)  # FastRTC is channels first.
        if samples.ndim != 1 or rate <= 0:
            raise ValueError("Expected a positive sample rate and channel-first PCM")
        if not np.any(samples):
            return ""
        if rate != 16000:
            factor = gcd(rate, 16000)
            samples = resample_poly(samples, 16000 // factor, rate // factor)
        with self.lock, self.model.session() as session:
            return session.run(np.ascontiguousarray(samples, dtype=np.float32), language="en").text.strip()


class _TurnWithPreRoll(ReplyOnPause):
    """Keep the last unvoiced VAD window in case speech began near its end."""

    def __init__(self, fn, model=None):
        super().__init__(fn, model=model, algo_options=AlgoOptions(
            audio_chunk_duration=1.0, started_talking_threshold=0.08, speech_threshold=0.03,
        ))

    def copy(self):
        # FastRTC creates a fresh handler (and pre-roll) for each connection.
        return type(self)(self.fn, model=self.model)

    def determine_pause(self, audio, sampling_rate, state):
        pause = super().determine_pause(audio, sampling_rate, state)
        if not state.started_talking and state.buffer is None:
            # The parent consumed this complete window without starting a turn.
            # Replace the previous pre-roll, so idle audio never accumulates.
            state.stream = audio
        return pause


def build_stream(recognizer: FourierSTT):
    import gradio as gr

    def on_pause(audio):
        text = recognizer.stt(audio)
        if text:
            yield AdditionalOutputs(text)

    return Stream(
        modality="audio",
        mode="send-receive",
        handler=_TurnWithPreRoll(on_pause),
        additional_outputs=[gr.Textbox(label="Last completed utterance")],
        additional_outputs_handler=lambda previous, text: text,
    )


if __name__ == "__main__":
    build_stream(FourierSTT()).ui.launch(server_name="127.0.0.1")
