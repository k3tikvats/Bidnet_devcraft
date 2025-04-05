# Adobe DevCraft Bidding System

A sophisticated real-time ad auction bidding system with Python and Java implementations.

## Overview

Adobe DevCraft Bidding System is a comprehensive solution for real-time ad auctions, featuring machine learning integration, efficient lookup systems, and robust data processing capabilities. The system is designed for optimal bidding decisions in high-frequency advertising environments.

## Quick Start

### Installation

Install dependencies using pip:
```bash
pip install -r requirements.txt
```

Or install using the setup script:
```bash
python setup.py install
```

Set up a conda environment:
```bash
# If using conda
conda env create -f environment.yaml
```

### Running Inference

```bash
cd Adobe_Devcraft_PS/bidder/python
python3 Bid.py
```

### Model Weights Location

```
Adobe_Devcraft_PS/bidder/python/model_parameters/epoch3_iter_0_train_5.467432975769043
```

## Key Features

- **Smart Bidding Algorithm**: ML-powered bidding strategies optimized for ROI
- **Data Processing Pipeline**: Comprehensive tools for data preparation and transformation
- **Machine Learning Integration**: Deep learning models for bid optimization
- **Visualization Tools**: Built-in capabilities for data analysis and visualization
- **Efficient Lookup System**: Fast bidding decisions through optimized lookup tables

## Project Structure

```
ADOBE DEVCRAFT/
├── Adobe_Devcraft_PS/
│   ├── bidder/
│   │   └── python/
│   │       ├── __pycache__/
│   │       ├── Code/
│   │       ├── model_parameters/
│   │       ├── advertiser_id.json
│   │       ├── Bid.py
│   │       ├── Bidder.py
│   │       ├── BidRequest.py
│   │       ├── city.txt
│   │       ├── profile.json
│   │       ├── region.txt
│   │       ├── __init__.py
│   │       └── city.txt
│   ├── README
│   ├── region.txt
│   └── user.profile.tags.txt
├── .gitignore
├── environment.yml
├── INSTRUCTION.md
├── requirements.txt
└── setup.py
```

## Data Files

| File | Description |
|------|-------------|
| `city.txt` | City-wise mapping information |
| `region.txt` | Regional classification data |
| `user.profile.tags.txt` | Detailed user profiling information |

## Core Components

### Optimization Tools
- **train.py**: Neural network training pipeline
- **model.py**: Deep learning model architecture
- **loss.py**: Custom loss functions for model optimization
- **create_look_up.py**: Lookup table generation for efficient bidding
- **histograms_and_box.py**: Statistical analysis and visualization tools

## Documentation

For detailed implementation guidance, refer to the files in the Code directory and the instructions in INSTRUCTION.md.

## Requirements

- Python 3.7+
- Dependencies listed in requirements.txt
- Optional: Conda environment manager