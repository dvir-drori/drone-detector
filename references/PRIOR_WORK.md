# Prior Work and Datasets

Open-source datasets, related projects, and design rationale for the
drone detector.

## Datasets

### Primary positives/negatives

- **DroneAudioDataset** (saraalemadi)
  <https://github.com/saraalemadi/DroneAudioDataset>
  Drone vs. non-drone audio clips.  Primary source for Stage-2 training
  positives and negatives, and for validating Stage-1 detection rate.

### Hard negatives

- **Drone-detection-dataset** (DroneDetectionThesis)
  <https://github.com/DroneDetectionThesis/Drone-detection-dataset>
  Includes bird, helicopter, and other confuser classes -- useful for
  evaluating false-alarm rejection on acoustically similar sources.

### Diverse drones and non-drone scenes

- **drone-audio-detection-samples** (geronimobasso, HuggingFace)
  <https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples>
  Multiple drone models and varied non-drone environments.  Good for
  cross-model generalisation testing.

### Low-SNR evaluation

- **DroneAudioSet** (NUS)
  Drone recordings at multiple distances in noisy environments.  Useful
  for evaluating detection performance at low signal-to-noise ratios
  where Stage-1 physics features are stressed.

## Related projects

### SudarshanChakra -- drone audio CNN

A CNN-based drone audio classifier.  Relevant as a **Stage-2 reference
implementation** -- it demonstrates that a small CNN on log-mel
spectrograms can reach high accuracy on clean drone audio.  It is
**not** a replacement for Stage 1: it requires labelled training data,
does not explain *why* a detection fires, and has not been validated on
low-SNR field recordings with real confusers.

## Design philosophy: Stage 1 stays physics-first

Stage 1 of this detector is deliberately **not** a learned model.  It
uses signal-processing features (harmonic comb scoring, spectral
flatness, envelope modulation, f0 jitter) that are grounded in the
physical acoustics of multirotor flight.

Benefits of this approach:

- **No training data required** -- works out of the box.
- **Interpretable** -- every detection can be traced to measured f0,
  harmonicity, tonality, and modulation depth.
- **Robust to distribution shift** -- a new drone model still has a
  harmonic comb; a new environment still has broadband noise.
- **Complementary to Stage 2** -- physics features become inputs to
  the learned classifier, not competitors.

Stage 2 (RandomForest, CNN) is the right place to absorb the nuances
that physics alone cannot capture: specific motor signatures, prop
noise texture, and site-specific confuser patterns.  But Stage 2 is
always *downstream* of Stage 1 -- it refines, not replaces.
