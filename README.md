Presentation: [Presentation.pdf](./Presentation(final).pdf)
Thesis:[Thesis.pdf](./Master_Thesis-final.pdf)
# Real-Time Signal Prediction System

## Requirements
- Python 3.9
- Install dependencies: `pip install -r requirements.txt`

## Predict.py (TCN Model)
Model: TCN pt10.04

For 500Hz sampling frequency, recommended parameters: delay=20000, refresh=0.2

**Usage:**
```bash
python Predict.py <filename> <delay> <refresh> <offset>
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| filename | Input CSV file path (file should update continuously) |
| delay | Warmup time in samples. 20000 = 40 seconds at 500Hz. Single wait at startup only |
| refresh | Output refresh interval in seconds. Can be set to 0 for minimal latency |
| offset | Calibration offset. Should give ~-0.302V at zero force |

**Output:** CSV file with processed signals, updated in real-time

**Example:**
```bash
python Predict.py test0105.csv 20000 0.2 -0.302
```

---

## Predict-KF.py (DLKF Model)
Model: DLKF pt0.795

For 500Hz sampling frequency, recommended parameters: delay=550, refresh=0.2

**Usage:**
```bash
python Predict-KF.py <filename> <delay> <refresh> <offset>
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| filename | Input CSV file path (file should update continuously) |
| delay | Warmup time in samples. 550 = ~1.1 seconds at 500Hz |
| refresh | Output refresh interval in seconds |
| offset | Calibration offset for environment and sensor variations |

**Output:** CSV file with processed signals, updated in real-time

**Example:**
```bash
python Predict-KF.py test0105.csv 550 0.2 -0.302
```
