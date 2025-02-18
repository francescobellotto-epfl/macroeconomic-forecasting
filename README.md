# FORECASTING S&P500 RETURNS WITH EARNINGS CALLS TRANSCRIPTS


## 📝 Project Description

This project focuses on predicting S&P 500 returns by analyzing insights from earnings call transcripts of the companies within the index. Earnings calls are a key way for company management to present financial results and future forecasts, providing valuable information for analysts and investors. The study explores whether the content of these calls can help predict a significant macroeconomic indicator, combining financial data from sources like WRDS (Wharton Research Data Services) with macroeconomic indicators from the FRED-MD system.


## 🚀 Setup Instructions

### 1. Clone the Project

Clone this repository to your local machine using the following command:

```bash
git clone <project_link>
cd <project_repo>
```

### 2. [OPTIONAL] create conda environment
```bash
conda create -n <env_name> python=3.11
conda activate <env_name>
```

### 3. install requirements
```bash
pip install -r requirements.txt
```

### 4. download data

To run the project, you’ll need to download the dataset. You can download it from the following link:

[Download Data](https://epflch-my.sharepoint.com/:f:/g/personal/francesco_bellotto_epfl_ch/ElsxLJLDlIxCrkA_6HAIv28BDc3xdexHJ8cJfZaYHiCYEA?e=had2EZ)

Once the data is downloaded, place it in the data/ directory (see Folder structure in next section).

## 📂 Folder Structure

The main results are summarized in run.ipynb, where all the explored model are run and the main results are displayed.

```bash
Forecasting S&P500 returns with earnings calls transcripts/
├── data/                   # Contains the dataset files (CSV, Parquet, ...) 
├── src/                    # Source code for the project
│   ├── scripts/            # Utility functions for data preprocessing 
│   └── utils/              # Utility functions for data manipulation, ML models and plotting
├── requirements.txt        # List of required libraries and dependencies
├── eda.ipynb               # Exploratory data analysis
├── run.ipynb               # Main notebook to run the prediction model
└── README.md               # Project documentation
```
