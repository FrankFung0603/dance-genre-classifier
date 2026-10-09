# CPU vs GPU Inference Results

Same model (R(2+1)D-18, FP16 ONNX), same test set (n = 1,414 windows), same preprocessing — only the execution provider differs.

## Overall

| Metric | GPU (DirectML) | CPU |
|---|---|---|
| Macro F1 | 0.879 | 0.880 |
| Accuracy | 0.874 | 0.875 |
| Latency (ms / clip) | 100 | 487 |

## Per-class results

| Class | GPU P | GPU R | GPU F1 | CPU P | CPU R | CPU F1 |
|---|---|---|---|---|---|---|
| ballet | 0.911 | 0.729 | 0.810 | 0.912 | 0.733 | 0.813 |
| breakdance | 0.935 | 0.924 | 0.929 | 0.935 | 0.924 | 0.929 |
| contemporary | 0.702 | 0.820 | 0.756 | 0.704 | 0.820 | 0.757 |
| hiphop | 0.845 | 0.817 | 0.831 | 0.845 | 0.817 | 0.831 |
| salsa | 0.971 | 0.997 | 0.984 | 0.971 | 0.997 | 0.984 |
| unknown | 0.957 | 0.973 | 0.965 | 0.957 | 0.973 | 0.965 |
| macro avg | 0.887 | 0.877 | 0.879 | 0.887 | 0.877 | 0.880 |

P = precision, R = recall. Latency = mean of 20 single-clip inference runs (16 frames, 224x224). Test set: held-out windows, group split by source video.
