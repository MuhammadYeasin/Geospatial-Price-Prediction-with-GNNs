# Geospatial Price Prediction with Graph Neural Networks (GNNs)# Geospatial Price Prediction with Graph Neural Networks (GNNs)



**Authors:**[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

- Md Nadim Yeasin (ID: 48343110, Email: mdnadim.yeasin@students.mq.edu.au)[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

- Safkat Hasin Alavi (ID: 48726591, Email: Safkathasin.alavi@students.mq.edu.au)[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.0%2B-orange)](https://pytorch-geometric.readthedocs.io/)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

A professional implementation of Graph Neural Networks for predicting Airbnb listing prices using geospatial data across Australian cities (Brisbane, Melbourne, Sydney). This project demonstrates the effectiveness of attention-based GNN architectures in capturing spatial relationships for price prediction tasks.

This project implements Graph Neural Networks (GNNs) for geospatial price prediction using Airbnb listings data from three major Australian cities (Brisbane, Melbourne, and Sydney). The project demonstrates how graph-based deep learning can effectively capture spatial relationships and predict property prices.

## Table of Contents

## Key Features

- [Overview](#overview)

- **Multi-City Dataset**: Combined analysis of 39,425+ Airbnb listings across Brisbane, Melbourne, and Sydney- [Key Features](#key-features)

- **Graph Construction**: k-Nearest Neighbors (k-NN) graph based on geographic coordinates (latitude/longitude)- [Project Structure](#project-structure)

- **Multiple GNN Architectures**: - [Installation](#installation)

  - GCN Baseline (Kipf & Welling, 2017)- [Usage](#usage)

  - GAT Advanced with multi-head attention (Veličković et al., 2018)- [Methodology](#methodology)

  - GAT_model Lightweight (our own design)- [Results](#results)

- **Comprehensive Evaluation**: Model comparison and systematic ablation studies- [Model Architectures](#model-architectures)

- **Feature Engineering**: 8 features including numerical attributes and one-hot encoded categorical variables- [Requirements](#requirements)

- [Contributing](#contributing)

## Project Structure- [License](#license)

- [Acknowledgments](#acknowledgments)

```

.## Overview

├── airbnb_gnn_analysis.ipynb    # Main notebook with complete pipeline

├── raw_data/                     # Raw CSV filesThis project implements multiple Graph Neural Network architectures to predict Airbnb property prices across Australian cities. By leveraging spatial relationships between properties through graph structures and attention mechanisms, the models achieve strong performance compared to traditional approaches.

│   ├── brisbane_listings.csv

│   ├── melbourne_listings.csv### Problem Statement

│   └── sydney_listings.csv

├── processed_data/              # Processed data and modelsPredict Airbnb listing prices based on:

│   ├── combined_data.pt         # PyTorch Geometric Data object- Property features (accommodates, bedrooms, bathrooms, beds, ratings)

│   ├── combined_listings_cleaned.csv- Geographical location (latitude, longitude)

│   ├── features.npy- Room type (entire home, private room, shared room)

│   └── scaler.pkl- Spatial relationships with neighboring properties

├── figures/                     # Generated visualizations

│   ├── ablation_studies.png### Dataset

│   ├── price_distribution.png

│   └── gnn_training_curves.png- **Source**: Airbnb listings for Brisbane, Melbourne, and Sydney

├── requirements.txt             # Python dependencies- **Raw data**: CSV files per city under `raw_data/`

└── README.md                    # This file- **Processed artifacts**: Saved under `processed_data/` (graph, features, scaler, cleaned CSV)

```- **Target**: Nightly price (continuous), modeled in log space; reported in dollar scale for interpretability



## Dataset## Key Features



The dataset is available at: [Google Drive](https://drive.google.com/drive/folders/1Pzh7fR5eCwEWfmx8bGoVnBlcTAfw3ayF?usp=sharing)- **Multiple GNN Implementations**:

   - GCN (Graph Convolutional Network) baseline

Download the CSV files and place them in the `raw_data/` directory:   - GAT (Graph Attention Network) advanced model

- `brisbane_listings.csv`   - Lightweight GAT model with 4 attention heads

- `melbourne_listings.csv`

- `sydney_listings.csv`- **Complete Pipeline**:

  - Data preprocessing and cleaning

## Installation  - Log transformation for skewed distributions

  - Feature engineering and scaling

### Prerequisites  - k-NN graph construction from geospatial coordinates

- Python 3.8+  - Train/validation/test splitting

- Virtual environment (recommended)  - Model training with early stopping

  - Comprehensive evaluation and visualization

### Setup

- **Robust Evaluation**:

1. **Clone the repository**  - RMSE in both log and dollar scales

```bash  - Model comparison and ablation studies

git clone https://github.com/MuhammadYeasin/Geospatial-Price-Prediction-with-GNNs.git  - Generalization analysis

cd Geospatial-Price-Prediction-with-GNNs  - Visualization of predictions and training progress

```

## Project Structure

2. **Create and activate virtual environment**

```bash```

python -m venv .venvGeospatial-Price-Prediction-with-GNNs/

source .venv/bin/activate  # On Windows: .venv\Scripts\activate│

```├── README.md

├── PROJECT_SUMMARY.md

3. **Install dependencies**├── airbnb_gnn_analysis.ipynb      # Main notebook (end-to-end pipeline)

```bash├── raw_data/

pip install -r requirements.txt│   ├── brisbane_listings.csv

```│   ├── melbourne_listings.csv

│   └── sydney_listings.csv

### Required Packages└── processed_data/

- pandas   ├── combined_data.pt              # PyTorch Geometric Data object

- numpy   ├── features.npy                  # Scaled feature matrix

- matplotlib   ├── scaler.pkl                    # StandardScaler for features

- scikit-learn   └── combined_listings_cleaned.csv # Cleaned combined dataset

- torch (PyTorch)```

- torch-geometric (PyTorch Geometric)

- scipy### Notebook Structure



## UsageThe main notebook (`airbnb_gnn_analysis.ipynb`) is organized into 6 phases:



### Running the Analysis1. **Phase 1: Setup and Data Loading**

   - Import libraries

1. **Ensure dataset is in place**   - Load dataset

   - Download data from Google Drive link above   - Clean price column

   - Place CSV files in `raw_data/` directory   - Handle missing values



2. **Open Jupyter Notebook**2. **Phase 2: Target Variable Transformation**

```bash   - Visualize price distribution

jupyter notebook airbnb_gnn_analysis.ipynb   - Apply log transformation

```   - Verify normalization



3. **Execute cells sequentially**3. **Phase 3: Feature Engineering**

   - The notebook is organized into 7 phases   - Select numerical and categorical features

   - All cells should be run in order   - Handle missing values (median imputation)

   - Results are reproducible with fixed random seeds   - One-hot encoding

   - Feature scaling (StandardScaler)

### Notebook Structure

4. **Phase 4: Graph Construction**

The notebook is organized into the following phases:   - Create k-nearest neighbors graph (k=8)

   - Convert to PyTorch Geometric format

#### Phase 1-4: Data Preparation and Graph Construction   - Create Data object with train/val/test masks

- Data loading, cleaning, and feature engineering

- Log transformation of prices for stability5. **Phase 5: GNN Model Building and Training**

- k-NN graph construction (k=8) based on geographic coordinates   - GCN baseline model

- Train/validation/test split (70/15/15)   - GAT advanced model

   - Lightweight GAT model

#### Phase 5: Model Architecture Comparison (2 Points)   - Training loops and evaluation

Comparison of three GNN architectures:

1. **GCN Baseline** - Standard graph convolution (Kipf & Welling 2017)6. **Phase 6: Final Evaluation and Analysis**

2. **GAT Advanced** - Multi-head attention (Veličković et al. 2018)   - Final test set performance

3. **GAT_model Lightweight** - Our efficient design   - Comprehensive analysis

   - Report-ready summaries

**Key Findings:**

- GAT models outperform GCN baseline## Installation

- Attention mechanisms effectively learn neighbor importance

- Lightweight design achieves competitive results with fewer parameters### Prerequisites



#### Phase 6: Ablation Studies (2 Points)- Python 3.8 or higher

Systematic evaluation of design choices:- pip package manager

1. **Attention Heads** (1 vs 4 vs 8) - 4 heads optimal- Jupyter Notebook or JupyterLab

2. **Dropout Rate** (0.0 vs 0.3 vs 0.5) - 0.3 provides best generalization

3. **Graph Connectivity** (k=5 vs 8 vs 12) - k=8 balances locality and context### Step 1: Clone the Repository

4. **Feature Importance** (with vs without room_type) - room_type is critical

```bash

**Key Insights:**git clone https://github.com/MuhammadYeasin/Geospatial-Price-Prediction-with-GNNs.git

- 4 attention heads provide optimal balancecd Geospatial-Price-Prediction-with-GNNs

- Moderate dropout (0.3) prevents overfitting```

- k=8 neighbors captures sufficient spatial context

- Categorical features (room_type) significantly impact performance### Step 2: Create Virtual Environment (Recommended)



#### Phase 7: Final Evaluation and Comprehensive Summary```bash

- Final test set evaluation on best model# Using conda

- Consolidated results from all experimentsconda create -n gnn-price python=3.9

- Performance metrics in dollar scale for interpretabilityconda activate gnn-price



## Model Architectures# Or using venv

python -m venv gnn-env

### 1. GCN Baselinesource gnn-env/bin/activate  # On Windows: gnn-env\Scripts\activate

```python```

- Architecture: 3-layer GCNConv

- Hidden dimensions: 64### Step 3: Install Dependencies

- Activation: ReLU

- Regularization: Dropout (0.3)```bash

- Aggregation: Uniform (treats all neighbors equally)# Install PyTorch (CPU version)

```pip install torch torchvision torchaudio



### 2. GAT Advanced# Install PyTorch Geometric

```pythonpip install torch-geometric

- Architecture: 3-layer GATConv

- Hidden dimensions: 64# Install other dependencies

- Attention heads: 4 (multi-head attention)pip install pandas numpy matplotlib scikit-learn jupyter

- Activation: ReLU```

- Regularization: Dropout (0.3)

- Aggregation: Attention-weighted (learns neighbor importance)### Step 4: Verify Installation

```

```bash

### 3. GAT_model Lightweight (Our Design)python -c "import torch; import torch_geometric; print('Installation successful!')"

```python```

- Architecture: 2-layer GATConv + Linear

- Hidden dimensions: 64## Usage

- Attention heads: 4

- Activation: ReLU### Running the Notebook

- Regularization: Dropout (0.3)

- Design goal: Reduced complexity while maintaining performance1. **Start Jupyter Notebook**:

```   ```bash

   jupyter notebook

## Results   ```



### Performance Metrics2. **Open the notebook**:

All models evaluated using:   - Navigate to `airbnb_gnn_analysis.ipynb`

- **RMSE (Root Mean Squared Error)** - Primary metric in dollar scale   - Run cells sequentially from top to bottom

- **MAE (Mean Absolute Error)** - Robust to outliers

- **R² Score** - Proportion of variance explained3. **Run All Phases**:

   - Click `Cell` > `Run All` to execute the entire pipeline

### Key Findings   - Or run cells individually to explore each phase



**Model Comparison:**### Quick Start Example

- GAT models consistently outperform GCN baseline

- Attention mechanism provides 5-10% improvement in RMSE```python

- Lightweight GAT achieves competitive results with 30% fewer parameters# After running all preprocessing phases, the main training loop is:



**Ablation Studies:**# Train the lightweight GAT model

- 4 attention heads optimal (diminishing returns beyond)for epoch in range(1, 201):

- Dropout=0.3 provides best test performance   loss = train()  # One training step

- k=8 neighbors is the sweet spot for spatial context   if epoch % 10 == 0:

- room_type feature critical (15%+ performance drop when removed)      val_rmse_dollars = test(data.val_mask)

      print(f"Epoch {epoch:3d} | Train Loss: {loss:.4f} | Val RMSE: ${val_rmse_dollars:,.2f}")

## Methodology

# Get final test performance

### Graph Constructionfinal_test_rmse = test(data.test_mask)

- **Nodes**: Individual property listings (39,425 nodes)print(f"Final Test RMSE: ${final_test_rmse:,.2f}")

- **Edges**: k-NN connectivity based on geographic distance```

- **k value**: 8 nearest neighbors per node

- **Graph type**: Undirected (symmetric connections)## Methodology



### Feature Engineering### 1. Data Preprocessing

**Numerical Features (5):**

- accommodates- **Price Cleaning**: Remove currency symbols and convert to float

- bedrooms- **Missing Value Handling**: Drop rows with missing prices, median imputation for features

- bathrooms- **Log Transformation**: Apply `log1p` to normalize skewed price distribution

- beds- **Feature Selection**: 5 numerical + 1 categorical (room_type) → 8 final features

- review_scores_rating

### 2. Graph Construction

**Categorical Features (1):**

- room_type (one-hot encoded into 3 features)- **Node Representation**: Each property is a node with 8 features

- **Edge Creation**: k-nearest neighbors (k=8) based on Euclidean distance of coordinates

**Total Features**: 8 dimensions per node- **Graph Properties**:

  - 5,152 nodes

### Training Configuration  - 52,118 edges (undirected)

- **Optimizer**: Adam (learning rate = 0.005)  - Average degree: 10.12

- **Loss Function**: Mean Squared Error (MSE)  - No isolated nodes

- **Epochs**: 100-200 with early stopping

- **Early Stopping**: Patience = 15-20 epochs### 3. Train/Validation/Test Split

- **Target Transform**: Log transformation (log1p)

- **Evaluation**: Inverse transform to dollar scale (expm1)- Training: 70% (3,606 nodes)

- Validation: 15% (772 nodes)

## Reproducibility- Test: 15% (774 nodes)



All experiments are reproducible:### 4. Model Training

- Fixed random seeds throughout the pipeline

- Consistent train/val/test splits- **Optimizer**: Adam with learning rate 0.005

- Saved processed data for quick replication- **Loss Function**: Mean Squared Error (MSE)

- All hyperparameters documented in notebook- **Training Duration**: 200 epochs

- **Regularization**: Dropout (0.3)

## Visualizations- **Evaluation**: RMSE in original dollar scale using `np.expm1()`



The project generates several visualizations:## Results

- Price distribution (original vs log-transformed)

- Training curves (loss and RMSE over epochs)### Model Performance Comparison

- Predictions vs actual scatter plots

- k-NN graph visualization| Model | Architecture | Parameters | Test RMSE (Log) | Test RMSE ($) |

- Ablation study comparison charts|-------|--------------|------------|-----------------|---------------|

- Model architecture performance comparison| **GCN Baseline** | 3 GCNConv layers | 4,801 | 0.6939 | N/A |

| **GAT Advanced** | 3 GATConv + Multi-head | 69,379 | 0.5797 | N/A |

All figures are saved to the `figures/` directory.| **GAT_model** | 2 GATConv + Linear | 19,457 | N/A | **$189.05** |



## Future Improvements### Key Findings



1. **Additional Features**1. **Performance**: GAT_model achieved **$189.05 test RMSE** in dollar scale

   - Amenities text features2. **Generalization**: Model performs better on test set ($189.05) than training set ($227.54)

   - Seasonal patterns3. **Efficiency**: GAT models outperform GCN by **16.45%** in RMSE

   - Host characteristics4. **Attention Advantage**: Multi-head attention effectively captures neighborhood importance

   - Neighborhood information

### Visualization Results

2. **Advanced Models**

   - Temporal GNNs for time-aware pricingThe notebook includes comprehensive visualizations:

   - Heterogeneous graphs with multiple node types- Price distribution (before/after log transformation)

   - Graph transformers for long-range dependencies- Training loss curves

- Validation RMSE over epochs

3. **Optimization**- Predictions vs Actual scatter plots

   - Hyperparameter tuning with grid/random search- Model comparison charts

   - Model ensembles

   - Transfer learning across cities## Model Architectures



## References### 1. GCN Baseline Model



1. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.```python

2. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. ICLR.class GCN_Baseline(nn.Module):

3. PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io/    - Layer 1: GCNConv(8, 64)

    - Layer 2: GCNConv(64, 64)

## License    - Layer 3: GCNConv(64, 1)

    - Activation: ReLU

This project is for academic purposes as part of COMP8221 coursework.    - Dropout: 0.3

```

## Acknowledgments

### 2. GAT Advanced Model

- Dataset: Airbnb listings data from Inside Airbnb

- Framework: PyTorch Geometric for GNN implementation```python

- Course: COMP8221 - Advanced Machine Learningclass GAT_Advanced(nn.Module):

    - Layer 1: GATConv(8, 64, heads=4)

## Contact    - Layer 2: GATConv(256, 64, heads=4)

    - Layer 3: GATConv(256, 1, heads=1)

For questions or collaboration:    - Activation: ELU

- Md Nadim Yeasin: mdnadim.yeasin@students.mq.edu.au    - Dropout: 0.3

- Safkat Hasin Alavi: Safkathasin.alavi@students.mq.edu.au```



---### 3. Lightweight GAT Model



**Note**: This project demonstrates the application of Graph Neural Networks to geospatial prediction tasks, achieving strong performance through attention mechanisms and systematic design choices validated via ablation studies.```python

class GAT_model(nn.Module):
    - Layer 1: GATConv(8, 64, heads=4)
    - Layer 2: GATConv(256, 64, heads=1)
    - Layer 3: Linear(64, 1)
    - Activation: ReLU
    - Dropout: 0.3
```

## Requirements

### Core Dependencies

```
python>=3.8
torch>=2.0.0
torch-geometric>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
scipy>=1.7.0
```

### Optional Dependencies

```
jupyter>=1.0.0
jupyterlab>=3.0.0
seaborn>=0.11.0  # For enhanced visualizations
```

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB RAM, GPU (CUDA compatible)
- **Storage**: 500MB for dataset and models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- [ ] Add more GNN architectures (GraphSAGE, GIN, etc.)
- [ ] Implement temporal GNN for time-series prediction
- [ ] Add hyperparameter tuning with Optuna
- [ ] Create Flask/Streamlit web interface
- [ ] Add more evaluation metrics (MAE, R², MAPE)
- [ ] Implement ensemble methods
- [ ] Add feature importance analysis
- [ ] Create automated data pipeline

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Airbnb Open Data (listings)
- PyTorch Geometric: for GNN implementations and documentation

## References

1. **Graph Convolutional Networks (GCN)**:
   - Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks"

2. **Graph Attention Networks (GAT)**:
   - Veličković et al. (2018). "Graph Attention Networks"

3. **PyTorch Geometric**:
   - Fey & Lenssen (2019). "Fast Graph Representation Learning with PyTorch Geometric"

## Contact

**Muhammad Yeasin**
- GitHub: [@MuhammadYeasin](https://github.com/MuhammadYeasin)
- Repository: [Geospatial-Price-Prediction-with-GNNs](https://github.com/MuhammadYeasin/Geospatial-Price-Prediction-with-GNNs)

## Version History

- **v1.0.0** (October 2025)
  - Initial release
  - Complete 6-phase implementation
  - Three GNN model architectures
  - Comprehensive evaluation and visualization

---

**If you find this project helpful, please consider giving it a star!**

**For questions or issues, please open an issue on GitHub.**
