# Multi-Layer Perceptron Exploratory Data Analysis

This project contains a series of Multi-Layer Perceptron (MLP) models implemented from scratch, designed to explore different architectures and optimization strategies. The models range from a simple two-layer perceptron to more complex five-layer models, with the addition of advanced techniques such as RMSProp and Adam optimization.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Descriptions](#model-descriptions)
- [Results](#results)
- [License](#license)

## Project Overview

This project explores the implementation and performance of various Multi-Layer Perceptron (MLP) architectures on a dataset from the 2017 American Community Survey (ACS). The models are built using basic Python libraries, with custom implementations of feedforward and backpropagation algorithms. The project is intended for educational purposes, particularly for those interested in learning about neural network architectures and optimization strategies.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/minhosong88/mlp-project-for-cencus-data-classification.git
cd mlp-project-for-cencus-data-classification
```

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install the Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the Models

To run the models, execute the main.py file. This script will load the data, preprocess it, and train the specified model.

```bash
python main.py
```

You can modify the main.py script to test different models or configurations

## Project Structure

The project is organized as follows:

```plaintext
MLP_Project_For_Census_Data_Classification/
│
├── scripts/
│   ├── __init__.py
│   ├── load_and_preprocess.py
│   ├── models.py
│   └── visualization.py
│
├── tests/
│   ├── __init__.py
│   └── test_models.py
│
├── data/
│   └── acs2017_census_tract_data.csv  # (This file is expected to be placed here)
│
├── notebooks/
│   └── EDA.ipynb  # (explorative data analysis in Jupyter notebooks)
│
├── main.py
├── requirements.txt
└── README.md
```

## Summary

The study demonstrates the importance of data preprocessing (e.g., normalization and one-hot encoding) and the potential benefits of deepening the network and employing adaptive learning strategies. However, the improvements in this specific task were modest, suggesting that further experimentation with different architectures, learning rates, or even regularization techniques might be necessary to achieve more substantial gains.

## Contact Information

For any questions or inquiries, please contact:

- **GitHub**: [minhosong88](https://github.com/minhosong88)
- **Email**: [hominsong@naver.com](mailto:hominsong@naver.com)
