import numpy as np
import os
import torch
import torch.nn.functional as F
import torchaudio
from IPython.display import Audio

def load_audio(fraction=1.0):
    train_dataset = torchaudio.datasets.SPEECHCOMMANDS(root='./data', download=True, subset='training')
    test_dataset = torchaudio.datasets.SPEECHCOMMANDS(root='./data', download=True, subset='testing')
    if fraction != 1.0:    
        train_size = int(len(train_dataset) * fraction)
        test_size = int(len(test_dataset) * fraction)
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(train_size)))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(test_size)))
    return train_dataset, test_dataset, train_dataset[0][1]

def play_audio(audio, path=None):
    if len(audio) == 2:
        waveform, sample_rate = audio
    elif len(audio) >= 5:
        waveform, sample_rate, label, speaker_id, utterance_number = audio[:5]
        print(f"Label: {label}, Speaker ID: {speaker_id}, Utterance #: {utterance_number}")
    else:
        raise ValueError("Audio input must be either (waveform, sample_rate) or (waveform, sample_rate, label, speaker_id, utterance_number)")
    _store(waveform, sample_rate, path)
    return Audio(waveform.detach().numpy(), rate=sample_rate)

def _store(waveform, sample_rate, path):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchaudio.save(path, waveform.detach().cpu(), sample_rate)

def get_waveforms(audio_dataset, percentile_fraction=0.95):
    percentile = percentile_fraction * 100
    lengths = [audio[0].shape[-1] for audio in audio_dataset]
    max_length = int(np.percentile(lengths, percentile))

    waveforms = []
    for audio in audio_dataset:
        waveform = audio[0]
        length = waveform.shape[-1]

        if length < max_length:
            pad_amount = max_length - length
            waveform = F.pad(waveform, (0, pad_amount))
        else:
            waveform = waveform[..., :max_length]

        waveforms.append(waveform)

    return torch.stack(waveforms)
