# Speech AI Sandbox

This repository contains implementations of machine learning architectures for audio generation, with experiments performed on the Speech Commands dataset.

The model currently examined is the LSTM-Autoencoder.

All models are implemented using **PyTorch**.

## Installation

To set up the environment, create a Conda environment from the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate speech-sandbox
```

## Showcase

### LSTM-Autoencoder Reconstruction

<div style="display: flex; gap: 20px; align-items: center; margin-bottom: 10px;">
  <div>
    <p style="font-weight: bold;">Original</p>
    <audio controls>
      <source src="audio/original_1.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div>
    <p style="font-weight: bold;">Reconstruction</p>
    <audio controls>
      <source src="audio/lstm_ae/reconstruction_1.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
</div>
<div style="display: flex; gap: 20px; align-items: center; margin-bottom: 10px;">
  <div>
    <audio controls>
      <source src="audio/original_2.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div>
    <audio controls>
      <source src="audio/lstm_ae/reconstruction_2.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
</div>
<div style="display: flex; gap: 20px; align-items: center;">
  <div>
    <audio controls>
      <source src="audio/original_3.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div>
    <audio controls>
      <source src="audio/lstm_ae/reconstruction_3.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
</div>
