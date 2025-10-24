# Geospatial Price Prediction with Graph Neural Networks (GNNs)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.0%2B-orange)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive implementation of Graph Neural Networks for predicting Brisbane Airbnb listing prices using geospatial data. This project demonstrates the effectiveness of attention-based GNN architectures in capturing spatial relationships for price prediction tasks.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Model Architectures](#model-architectures)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project implements multiple Graph Neural Network architectures to predict Airbnb property prices in Brisbane, Australia. By leveraging spatial relationships between properties through graph structures and attention mechanisms, the models achieve superior performance compared to traditional approaches.

### Problem Statement

Predict Airbnb listing prices based on:
- Property features (accommodates, bedrooms, bathrooms, beds, ratings)
- Geographical location (latitude, longitude)
- Room type (entire home, private room, shared room)
- Spatial relationships with neighboring properties

### Dataset

- **Source**: Brisbane Airbnb Listings
- **Size**: 5,774 listings (5,152 after cleaning)
- **Features**: 79 columns including price, location, amenities, and reviews
- **Target**: Price (continuous variable)

## ✨ Key Features

- **Multiple GNN Implementations**:
  - GCN (Graph Convolutional Network) Baseline
  - GAT (Graph Attention Network) Advanced Model
  - Assignment-specific GAT Model with 4 attention heads

- **Complete Pipeline**:
  - Data preprocessing and cleaning
  - Log transformation for skewed distributions
  - Feature engineering and scaling
  - k-NN graph construction from geospatial coordinates
  - Train/validation/test splitting
  - Model training with early stopping
  - Comprehensive evaluation and visualization

- **Robust Evaluation**:
  - RMSE in both log and dollar scales
  - Model comparison and ablation studies
  - Generalization analysis
  - Visualization of predictions and training progress

## 📁 Project Structure

```
Geospatial-Price-Prediction-with-GNNs/
│
├── README.md                      # This file
├── phase1_data_loading.ipynb     # Main Jupyter notebook with all 6 phases
├── brisbane_listings.csv         # Dataset (Brisbane Airbnb listings)
│
└── .git/                          # Git repository
```

### Notebook Structure

The main notebook (`phase1_data_loading.ipynb`) is organized into 6 phases:

1. **Phase 1: Setup and Data Loading**
   - Import libraries
   - Load dataset
   - Clean price column
   - Handle missing values

2. **Phase 2: Target Variable Transformation**
   - Visualize price distribution
   - Apply log transformation
   - Verify normalization

3. **Phase 3: Feature Engineering**
   - Select numerical and categorical features
   - Handle missing values (median imputation)
   - One-hot encoding
   - Feature scaling (StandardScaler)

4. **Phase 4: Graph Construction**
   - Create k-nearest neighbors graph (k=8)
   - Convert to PyTorch Geometric format
   - Create Data object with train/val/test masks

5. **Phase 5: GNN Model Building and Training**
   - GCN Baseline Model
   - GAT Advanced Model
   - Assignment-specific GAT Model
   - Training loops and evaluation

6. **Phase 6: Final Evaluation and Analysis**
   - Final test set performance
   - Comprehensive analysis
   - Report-ready summaries

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Step 1: Clone the Repository

```bash
git clone https://github.com/MuhammadYeasin/Geospatial-Price-Prediction-with-GNNs.git
cd Geospatial-Price-Prediction-with-GNNs
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n gnn-price python=3.9
conda activate gnn-price

# Or using venv
python -m venv gnn-env
source gnn-env/bin/activate  # On Windows: gnn-env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Step 4: Verify Installation

```bash
python -c "import torch; import torch_geometric; print('✅ Installation successful!')"
```

## 💻 Usage

### Running the Notebook

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**:
   - Navigate to `phase1_data_loading.ipynb`
   - Run cells sequentially from top to bottom

3. **Run All Phases**:
   - Click `Cell` > `Run All` to execute the entire pipeline
   - Or run cells individually to explore each phase

### Quick Start Example

```python
# After running all preprocessing phases, the main training loop is:

# Train the assignment-specific GAT model
for epoch in range(1, 201):
    loss = train()  # One training step
    
    if epoch % 10 == 0:
        val_rmse_dollars = test(data.val_mask)
        print(f"Epoch {epoch:3d} | Train Loss: {loss:.4f} | Val RMSE: ${val_rmse_dollars:,.2f}")

# Get final test performance
final_test_rmse = test(data.test_mask)
print(f"Final Test RMSE: ${final_test_rmse:,.2f}")
```

## 🔬 Methodology

### 1. Data Preprocessing

- **Price Cleaning**: Remove currency symbols and convert to float
- **Missing Value Handling**: Drop rows with missing prices, median imputation for features
- **Log Transformation**: Apply `log1p` to normalize skewed price distribution
- **Feature Selection**: 5 numerical + 1 categorical (room_type) → 8 final features

### 2. Graph Construction

- **Node Representation**: Each property is a node with 8 features
- **Edge Creation**: k-nearest neighbors (k=8) based on Euclidean distance of coordinates
- **Graph Properties**:
  - 5,152 nodes
  - 52,118 edges (undirected)
  - Average degree: 10.12
  - No isolated nodes

### 3. Train/Validation/Test Split

- Training: 70% (3,606 nodes)
- Validation: 15% (772 nodes)
- Test: 15% (774 nodes)

### 4. Model Training

- **Optimizer**: Adam with learning rate 0.005
- **Loss Function**: Mean Squared Error (MSE)
- **Training Duration**: 200 epochs
- **Regularization**: Dropout (0.3)
- **Evaluation**: RMSE in original dollar scale using `np.expm1()`

## 📊 Results

### Model Performance Comparison

| Model | Architecture | Parameters | Test RMSE (Log) | Test RMSE ($) |
|-------|--------------|------------|-----------------|---------------|
| **GCN Baseline** | 3 GCNConv layers | 4,801 | 0.6939 | N/A |
| **GAT Advanced** | 3 GATConv + Multi-head | 69,379 | 0.5797 | N/A |
| **GAT_model** | 2 GATConv + Linear | 19,457 | N/A | **$189.05** |

### Key Findings

1. **Performance**: GAT_model achieved **$189.05 test RMSE** in dollar scale
2. **Generalization**: Model performs better on test set ($189.05) than training set ($227.54)
3. **Efficiency**: GAT models outperform GCN by **16.45%** in RMSE
4. **Attention Advantage**: Multi-head attention effectively captures neighborhood importance

### Visualization Results

The notebook includes comprehensive visualizations:
- Price distribution (before/after log transformation)
- Training loss curves
- Validation RMSE over epochs
- Predictions vs Actual scatter plots
- Model comparison charts

## 🏗️ Model Architectures

### 1. GCN Baseline Model

```python
class GCN_Baseline(nn.Module):
    - Layer 1: GCNConv(8, 64)
    - Layer 2: GCNConv(64, 64)
    - Layer 3: GCNConv(64, 1)
    - Activation: ReLU
    - Dropout: 0.3
```

### 2. GAT Advanced Model

```python
class GAT_Advanced(nn.Module):
    - Layer 1: GATConv(8, 64, heads=4)
    - Layer 2: GATConv(256, 64, heads=4)
    - Layer 3: GATConv(256, 1, heads=1)
    - Activation: ELU
    - Dropout: 0.3
```

### 3. Assignment-Specific GAT Model

```python
class GAT_model(nn.Module):
    - Layer 1: GATConv(8, 64, heads=4)
    - Layer 2: GATConv(256, 64, heads=1)
    - Layer 3: Linear(64, 1)
    - Activation: ReLU
    - Dropout: 0.3
```

## 📦 Requirements

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

## 🤝 Contributing

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Airbnb listings data from Brisbane, Australia
- **PyTorch Geometric**: For excellent GNN implementations and documentation
- **Assignment**: COMP8221 - Advanced Topics in Data Science
- **Institution**: Griffith University

## 📚 References

1. **Graph Convolutional Networks (GCN)**:
   - Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks"

2. **Graph Attention Networks (GAT)**:
   - Veličković et al. (2018). "Graph Attention Networks"

3. **PyTorch Geometric**:
   - Fey & Lenssen (2019). "Fast Graph Representation Learning with PyTorch Geometric"

## 📧 Contact

**Muhammad Yeasin**
- GitHub: [@MuhammadYeasin](https://github.com/MuhammadYeasin)
- Repository: [Geospatial-Price-Prediction-with-GNNs](https://github.com/MuhammadYeasin/Geospatial-Price-Prediction-with-GNNs)

## 🔄 Version History

- **v1.0.0** (October 2025)
  - Initial release
  - Complete 6-phase implementation
  - Three GNN model architectures
  - Comprehensive evaluation and visualization

---

**⭐ If you find this project helpful, please consider giving it a star!**

**📝 For questions or issues, please open an issue on GitHub.**
