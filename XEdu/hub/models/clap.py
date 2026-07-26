import random

import onnxruntime
import soundfile as sf
import numpy as np


TARGET_SAMPLE_RATE = 44100


def _is_waveform_tuple(audio_input):
    return (
        isinstance(audio_input, tuple)
        and len(audio_input) == 2
        and isinstance(audio_input[1], (int, np.integer))
    )


class CLAP:
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path)

    def _to_mono(self, audio_time_series):
        audio_time_series = np.asarray(audio_time_series, dtype=np.float32)
        if audio_time_series.ndim == 0:
            audio_time_series = audio_time_series.reshape(1)
        elif audio_time_series.ndim == 2:
            if audio_time_series.shape[-1] <= 8:
                audio_time_series = audio_time_series.mean(axis=-1)
            elif audio_time_series.shape[0] <= 8:
                audio_time_series = audio_time_series.mean(axis=0)
            else:
                audio_time_series = audio_time_series.reshape(-1)
        elif audio_time_series.ndim > 2:
            audio_time_series = audio_time_series.reshape(-1)
        return audio_time_series.astype(np.float32)

    def _resample(self, audio_time_series, sample_rate, target_rate=TARGET_SAMPLE_RATE):
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if sample_rate == target_rate or audio_time_series.size == 0:
            return audio_time_series

        target_length = max(1, int(round(audio_time_series.shape[0] * target_rate / sample_rate)))
        old_positions = np.linspace(0.0, 1.0, num=audio_time_series.shape[0], endpoint=False)
        new_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
        return np.interp(new_positions, old_positions, audio_time_series).astype(np.float32)

    def read_audio(self, audio_input):
        if isinstance(audio_input, str):
            audio_time_series, sample_rate = sf.read(audio_input)
        elif isinstance(audio_input, np.ndarray):
            audio_time_series, sample_rate = audio_input, TARGET_SAMPLE_RATE
        elif _is_waveform_tuple(audio_input):
            audio_time_series, sample_rate = audio_input
        elif isinstance(audio_input, dict):
            audio_time_series = (
                audio_input.get("waveform")
                if "waveform" in audio_input
                else audio_input.get("audio", audio_input.get("data"))
            )
            if audio_time_series is None:
                raise ValueError("Audio dict input must contain waveform, audio, or data.")
            sample_rate = audio_input.get("sample_rate", TARGET_SAMPLE_RATE)
        else:
            raise TypeError(
                "Audio input must be a file path, numpy waveform, "
                "(waveform, sample_rate) tuple, or dict."
            )

        audio_time_series = self._to_mono(audio_time_series)
        audio_time_series = self._resample(audio_time_series, sample_rate)
        if audio_time_series.size == 0:
            raise ValueError("Audio input is empty.")
        return audio_time_series.astype(np.float32), TARGET_SAMPLE_RATE

    def load_audio_into_tensor(self,audio_input, audio_duration):
        r"""Loads audio file and returns raw audio."""
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = self.read_audio(audio_input)
        audio_time_series = audio_time_series.reshape(-1)
        target_length = int(audio_duration * sample_rate)

        # audio_time_series is shorter than predefined audio duration,
        # so audio_time_series is extended
        if target_length >= audio_time_series.shape[0]:
            repeat_factor = int(np.ceil(target_length / audio_time_series.shape[0]))
            # Repeat audio_time_series by repeat_factor to match audio_duration
            audio_time_series = audio_time_series.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio_time_series = audio_time_series[0:target_length]
        else:
            # audio_time_series is longer than predefined audio duration,
            # so audio_time_series is trimmed
            start_index = random.randrange(audio_time_series.shape[0] - target_length)
            audio_time_series = audio_time_series[start_index:start_index + target_length]
        # return torch.FloatTensor(audio_time_series)
        return audio_time_series 

    def preprocess_audio(self,audio_inputs):
        r"""Load list of audio files and return raw audio"""
        audio_tensors = []
        duration = 7
        for audio_input in audio_inputs:
            audio_tensor = self.load_audio_into_tensor(audio_input, duration)
            audio_tensor = audio_tensor.reshape(1, -1)
            audio_tensors.append(audio_tensor)
        return audio_tensors
    
    def get_audio_embedding(self,audio_files):
        r"""Compute audio embeddings for a list of audio files"""
        preprocessed_audio = self.preprocess_audio(audio_files)
        ort_session = self.model
        res = []
        for i in preprocessed_audio:
            ort_inputs = {ort_session.get_inputs()[0].name: i}

            ort_outs = ort_session.run(None, ort_inputs)[0]
            res.append(ort_outs[0])
        return np.stack(res)
