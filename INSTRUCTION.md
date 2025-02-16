# Adobe DevCraft Bidding System 🚀

> A sophisticated bidding system implementation for real-time ad auctions, supporting both Python and Java implementations.


## ⚡ Quick Start

### 🔧 Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Or use the setup script:**
   ```bash
   python setup.py install
   ```

2. **Configure environment:**
   ```bash
   # If using conda
   conda env create -f environment.yaml
   ```

### 🚀 Running Inference

```bash
cd Adobe_Devcraft_PS/bidder/python
python3 Bid.py
```
## model weights
```bash
Adobe_Devcraft_PS/bidder/python/model_parameters/epoch3_iter_0_train_5.467432975769043
```


## ✨ Features


- **📊 Data Processing**: Comprehensive tools for data preparation and transformation
- **🤖 Machine Learning Integration**: Deep learning model for optimized bidding strategies
- **📈 Visualization Tools**: Built-in data visualization capabilities
- **⚡ Lookup System**: Efficient lookup table implementation for fast bidding decisions


## 📁 Project Structure

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


## 📂 Data Files

| File | Description |
|------|-------------|
| `city.txt` | Contains city-wise mapping information |
| `region.txt` | Regional classification data |
| `user.profile.tags.txt` | Detailed user profiling information |


## 🛠️ Core Components


### Optimization Tools
- `train.py`: Neural network training pipeline
- `model.py`: Deep learning model architecture
- `loss.py`: Custom loss functions for model optimization
- `create_look_up.py`: Lookup table generation for efficient bidding
- `histograms_and_box.py`: Statistical analysis and visualization tools

