# RobustAnalogNet: Multi-Degradation Aware Deep Learning for Industrial Analog Meter Reading

Deep learning approach for robust analog meter reading in harsh industrial environments with degradation-aware architecture.

## Features

- Multi-degradation simulation (dirt, damage, lighting variations)
- Attention-based pointer detection
- Uncertainty quantification for reliability assessment
- Few-shot domain adaptation for new meter types

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- NumPy

## Dataset

Sample meter image: `meter.jpg`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from robust_analog_net import RobustAnalogNet

model = RobustAnalogNet()
reading, confidence = model.predict("meter.jpg")
```

## Research

This work targets submission to Artificial Intelligence journal focusing on novel contributions in industrial AI applications.