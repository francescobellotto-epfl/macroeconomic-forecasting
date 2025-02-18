import numpy as np
import pandas as pd
import datetime 
import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
import random
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
import sys
import os

class BaselineNeuralNetwork(nn.Module):
    """
    Baseline model, made of a single linear 
    regression layer
    """

    def __init__(self, input_size, output_size):
        """
        Constructor

        Args:
            input_size: size of the input
            output_size: size of the output
        """

        super(BaselineNeuralNetwork, self).__init__()
        layers = []
        
        # Output layer
        layers.append(nn.Linear(input_size, output_size))
        
        # Combine layers into a sequential module
        self.network = nn.Sequential(*layers)
    
    def forward(self, X):
        """
        Forward pass

        Args:
            inputs
        
        Returns:
            NN output
        """
        return self.network(X)

class NeuralNetwork(nn.Module):
    """
    Neural Network used in the run. Three hidden layers consisting 
    of Linear, ReLu sequence.
    """

    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        """
        Constructor

        Args:
            input_size/output_size
            hidden_size: size of the first hidden layer
            num_hidden_layers: number of hidden layers (unused)
        """

        super(NeuralNetwork, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        layers.append(nn.Linear(hidden_size, hidden_size//2))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size//2, hidden_size//4))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size//4, hidden_size//8))
        layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size//8, output_size))
        
        # Combine layers into a sequential module
        self.network = nn.Sequential(*layers)
    
    def forward(self, X):
        """
        Forward pass

        Args:
            inputs
        
        Returns:
            NN output
        """
        return self.network(X)


def change_format(row):
    """
    Transform a string with the format in 'embeddings' to a list of numbers.

    Args:
        row

    Returns:
        num_list: list of numerical values
    """

    string = row.replace('[', '').replace(']', '').replace('\n', '')
    numbers = string.split()
    num_list = [float(num) for num in numbers]

    return num_list

def get_quarter(month):
    """
    Compute quarter of the year for a month.
    """
    return (month - 1) // 3 + 1

def apply_tranformations(data, trans_row=1, start_from=2, skip_cols={}):
    """
    Apply needed transformations to Fred-MD Dataset.
    Transformations are specified through a number in row
    trans_row.
    The map for transformations is:
    - 1 -> no transformation
    - 2 -> Delta (of subsequent rows)
    - 3 -> Delta^2
    - 4 -> log
    - 5 -> Delta(log)
    - 6 -> (Delta^2)(log)
    - 7 -> Ratio (of subsequent rows) - 1

    Args:
        data: pandas DataFrame with Fred-MD Data.
        trans_row: number of row where transformations are specified (default 1).
        start_from: number of row from where to begin transformations (default 2).
        skip_cols: Python set where to put names of columns that are not
                conceptually transformable.

    Returns:
        data: the changed dataset, with applied transformations, and 
            trans_row values set to 1 for transformed columns.
    """
    
    for col_idx, col_name in enumerate(data.columns):
            
        if col_name not in skip_cols:

            match data.iloc[trans_row, col_idx]:
                case 2:
                    data.iloc[start_from:, col_idx] = data.iloc[start_from:, col_idx].diff()
                case 3:
                    data.iloc[start_from:, col_idx] = data.iloc[start_from:, col_idx].diff().diff()
                case 4:
                    data.iloc[start_from:, col_idx] = data.iloc[start_from:, col_idx].apply(lambda x: np.log(x) if ~np.isnan(x) else x)
                case 5:
                    data.iloc[start_from:, col_idx] = data.iloc[start_from:, col_idx].apply(lambda x: np.log(x) if ~np.isnan(x) else x)
                    data.iloc[start_from:, col_idx] = data.iloc[start_from:, col_idx].diff()
                case 6:
                    data.iloc[start_from:, col_idx] = data.iloc[start_from:, col_idx].apply(lambda x: np.log(x) if ~np.isnan(x) else x)
                    data.iloc[start_from:, col_idx] = data.iloc[start_from:, col_idx].diff().diff()
                case 7:
                    data.iloc[start_from:, col_idx] = data.iloc[start_from:, col_idx] / data.iloc[start_from:, col_idx].shift(1) - 1
            
            data.iloc[1, col_idx] = 1

    return data

def train_model(model, dataloader, criterion, optimizer, num_epochs, tol, device, epochs_limit):
    """
    Train NN model

    Args:
        model: model to use for training
        dataloader: DataLoader
        criterion: criterion for loss computation
        optimizer: optimizer for training
        num_epochs: maximum number of training epochs
        tol: tolerance for loss stability to stop epochs
        device: device for computations (e.g. 'cpu')
        epochs_limit: max number of epochs 

    Returns:
        losses: computed losses in the training phase
    """

    model.train()
    flag = True
    differences = 1e5*np.ones((epochs_limit,1))
    cont = 0
    new_loss = 0.0
    losses = []
    while(cont < num_epochs and flag == True):
        l = []
        old_loss = new_loss
        for i, X in enumerate(dataloader):
            inputs = X[0].to(dtype=torch.float32).to(device)
            targets = X[1].to(dtype=torch.float32).to(device)  

            # Forward pass  
            predictions = model(inputs) 
            loss = criterion(predictions, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l.append(loss.item())

        # Average loss on the batch
        new_loss = np.mean(l)
        losses.append(new_loss)

        differences = np.append(differences,abs(new_loss-old_loss))

        differences = differences[1:]

        # Check if loss is enough stable
        if(np.all(differences < tol)):
            flag = False

        cont = cont + 1

    return losses

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate NN model

    Args:
        model: model to use for evaluation
        dataloader: DataLoader
        criterion: criterion for loss computation
        device: device for computations (e.g. 'cpu')

    Returns:
        outputs: predicted values
        average loss on batch
    """

    model.eval() 
    total_loss = 0.0
    outputs = torch.tensor([]).to(device)
    with torch.no_grad():  
        for _, X in enumerate(dataloader):
            inputs = X[0].to(dtype=torch.float32).to(device)
            targets = X[1].to(dtype=torch.float32).to(device)
            output = model(inputs)
            loss = criterion(output, targets)
            total_loss += loss.item()
            outputs = torch.cat((outputs, output), dim=0)
    return outputs, (total_loss / len(dataloader))

def get_by_dates(X, month_col, year_col, start_month, start_year, end_month, end_year):
    """
    Filters a DataFrame based on a date range defined by start and end months and years.

    Args:
        X (pd.DataFrame): The input DataFrame.
        month_col (str): The name of the column containing months.
        year_col (str): The name of the column containing years.
        start_month (int): The starting month of the range.
        start_year (int): The starting year of the range.
        end_month (int): The ending month of the range.
        end_year (int): The ending year of the range.

    Returns:
        pd.DataFrame: A filtered DataFrame within the specified date range.
    """
    # Check if columns exist
    if month_col not in X.columns or year_col not in X.columns:
        raise ValueError('Columns not found')
    if start_year == end_year and start_month < end_month:
        filtered_X = X[
        ((X[year_col] == start_year) & (X[month_col] >= start_month) & (X[month_col] <= end_month)) 
        ]
    if start_year < end_year:
        filtered_X = X[
        ((X[year_col] == start_year) & (X[month_col] >= start_month)) |
        ((X[year_col] == end_year) & (X[month_col] <= end_month)) |
        ((X[year_col] > start_year) & (X[year_col] < end_year))
        ]    
    
    return filtered_X

def get_end_train(start_month, start_year):
    """
    Calculates the end month and year of the training period, 
    given the start month and year.

    Args:
        start_month (int): The starting month of the training period.
        start_year (int): The starting year of the training period.

    Returns:
        tuple: (end_month, end_year) for the training period.
    """
    start_month = int(start_month)
    start_year = int(start_year)
    if start_month==1:
        return 12, start_year+5
    elif start_month==4:
        return 3, start_year+6
    elif start_month==7:
        return 6, start_year+6
    elif start_month==10:
        return 9, start_year+6
    else:
        raise ValueError('Invalid start month') 
    
def get_largest_N_mktcap(dataset, N, start_month, start_year, end_month, end_year):
    """
    Extract rows only corresponding to N mostly capitalized companies 
    (on average) within the required time period.

    Args:
        dataset: Pandas DataFrame
        N: positive integer
        start_month: start month of the period
        start_year: start year of the period
        end_month: end month of the period
        end_year: end year of the period

    Returns:
        Required Pandas DataFrame
    """

    dataset = get_by_dates(dataset, 'month', 'year', start_month, start_year, end_month, end_year)
    avg_mkt= dataset.groupby(['gvkey'])['mktcap'].mean().reset_index()
    selected_gvkey = avg_mkt.nlargest(N, 'mktcap')['gvkey'].reset_index(drop=True)
    return dataset[dataset['gvkey'].isin(selected_gvkey)]

def elementwise_mean(lists):
    """
    Compute elementwise mean for input list

    Args:
        list: list of numerical values
    
    Returns:
        element-wise mean    
    """
    
    return np.mean(lists, axis=0)

def get_end_train(start_month, start_year):
    """
    Calculates the end month and year of the training period, 
    given the start month and year.

    Args:
        start_month (int): The starting month of the training period.
        start_year (int): The starting year of the training period.

    Returns:
        tuple: (end_month, end_year) for the training period.
    """
    start_month = int(start_month)
    start_year = int(start_year)
    if start_month==1:
        return 9, start_year+5
    elif start_month==4:
        return 12, start_year+5
    elif start_month==7:
        return 3, start_year+6
    elif start_month==10:
        return 6, start_year+6
    else:
        raise ValueError('Invalid start month') 
    
    
def get_next_quarter(end_month, end_year):
    """
    Calculates the next quarter's start and end month along with the corresponding year.

    Args:
        end_month (int): The ending month of the current quarter.
        end_year (int): The year of the current quarter.

    Returns:
        tuple: (next_start_month, next_start_year, next_end_month, next_end_year)
    """
    end_month = int(end_month)
    end_year = int(end_year)
    if end_month==1:
        return 2, end_year, 4, end_year
    elif end_month==2:
        return 3, end_year, 5, end_year
    elif end_month==3:
        return 4, end_year, 6, end_year
    elif end_month==4:
        return 5, end_year, 7, end_year
    elif end_month==5:
        return 6, end_year, 8, end_year
    elif end_month==6:
        return 7, end_year, 9, end_year
    elif end_month==7:
        return 8, end_year, 10, end_year
    elif end_month==8:
        return 9, end_year, 11, end_year
    elif end_month==9:
        return 10, end_year, 12, end_year
    elif end_month==10:
        return 11, end_year, 1, end_year+1
    elif end_month==11:
        return 12, end_year, 2, end_year+1
    elif end_month==12:
        return 1, end_year+1, 3, end_year+1
    else:
        raise ValueError('Invalid start month')

    
def get_next_iteration(start_month,start_year):
    """
    Get quarter just after required time

    Args:
        start_month, start_year

    Returns:
        required quarter month and year
    """
    
    # Create a timestamp for the end of the current quarter
    current_date = pd.Timestamp(year=start_year, month=start_month, day=1)
    
    # Move to the start of the next quarter
    next_start_date = current_date + pd.DateOffset(months=3)

    return next_start_date.month, next_start_date.year
    

def create_by_mean(top_N_data, Y, output_columns):
    """
    Build inputs and outputs tensors aggregating by elementwise mean 
    embeddings in each quarter.

    Args:
        top_N_data: Pandas DataFrame of inputs (should contain 
                    columns: 'embeddings', 'quarter and year')
        Y: Pandas DataFrame of outputs (should contain columns: 
                    'quarter and year')
        output_columns: list of valid column names for Y

    Returns:
        X: Torch Tensor of inputs (shape (n,768), n=number of 
                    quarters in top_N_data)
        Y_: Torch Tensor of outputs (shape (n,out_dim), out_dim=
                    number of output columns)
    """
    
    # Select only needed columns for inputs
    mean_embeddings = top_N_data[['embeddings', 'quarter and year']]
    # Standardize embeddings
    mean_embeddings['embeddings'].apply(lambda x: np.array(x))
    mean_embeddings['embeddings'].apply(lambda x: x/(np.linalg.norm(x)+1e-10))
    # Apply elemntwise mean
    mean_embeddings = mean_embeddings.groupby(by='quarter and year').agg(lambda x: elementwise_mean(list(x))).reset_index()
    # Select only needed columns for outputs
    mean_Y = Y[output_columns+['quarter and year']]
    # Consider only quarters for which we have transcripts
    mean_dataset = pd.merge(mean_embeddings, mean_Y, right_on='quarter and year', left_on='quarter and year', how='left')
    # Select texts as a list
    X = [row['embeddings'] for _,row in mean_dataset.iterrows()]
    # Create outputs Torch tensor
    Y_ = mean_dataset[output_columns]

    return X, Y_

def create_by_PCA(top_N_data_train, top_N_data_test, Y, output_columns, n_comp, training=True):
    """
    Build inputs and outputs tensors aggregating by PCA 
    embeddings in each quarter.

    Args:
        top_N_data_train: Pandas DataFrame of train inputs (should contain 
                    columns: 'embeddings', 'quarter and year')
        top_N_data_test: Pandas DataFrame of test inputs (should contain 
                    columns: 'embeddings', 'quarter and year')
        Y: Pandas DataFrame of outputs (should contain columns: 
                    'quarter and year')
        output_columns: list of valid column names for Y
        n_comp: number of components to consider in PCA decomposition
        training: bool indicating whether creation is required in training 
                    phase or not (default True)

    Returns:
        X_train: Torch Tensor of train inputs (shape (n,768), n=number of 
                    quarters in top_N_data)
        X_test: Torch Tensor of test inputs (shape (n,768), n=number of 
                    quarters in top_N_data)
        Y_train: Torch Tensor of train outputs (shape (n,out_dim), out_dim=
                    number of output columns)
        Y_test: Torch Tensor of test outputs (shape (n,out_dim), out_dim=
                    number of output columns)
        eig_val: fraction of explained variance by considered PCA 
        components
    """
    
    X_train = []
    X_train = torch.tensor(X_train)
    # Loop over quarters in top_N_data_train
    for qay in top_N_data_train['quarter and year'].unique():
        emb=[]
        # Create a list of all embeddings for a specific quarter
        for _, s in top_N_data_train[top_N_data_train['quarter and year']==qay].iterrows():
            emb.append(s['embeddings'])
        emb = np.array(emb)
        # Center embeddings on 0
        data_centred = np.array(emb - emb.mean(axis=0))
        # Compute covariance matrix
        covariance_mat = np.dot(data_centred.T,data_centred)
        # Apply PCA to get the first n_comp components
        pca = PCA(n_components = n_comp)
        principal_scores = pca.fit_transform(covariance_mat) 
        principal_vectors = pca.components_  # Shape: (n_comp, 768)
        # Compute explained variance ratio 
        eig_val = pca.explained_variance_ratio_
        # Add result to inputs tensor
        out = torch.tensor(principal_scores.flatten().reshape(1, -1))
        X_train = torch.cat([X_train, out], dim=0)

    # Select only needed columns for outputs
    mean_Y_train = Y[output_columns+['quarter and year']]
    # Consider only quarters for which we have transcripts
    mean_Y_train = mean_Y_train[mean_Y_train['quarter and year'].isin(top_N_data_train['quarter and year'].unique())]
    # Create output Torch tensor
    Y_train = mean_Y_train[output_columns]

    X_test = []
    X_test = torch.tensor(X_test)
    Y_test = 0
    if not training:
        # Now we can create X_test exploiting last computed pca
        X_test = []
        X_test = torch.tensor(X_test)
        emb=[]
        for _, s in top_N_data_test[top_N_data_test['quarter and year']==top_N_data_test['quarter and year'].unique()[0]].iterrows():
            emb.append(s['embeddings'])
        emb = np.array(emb)
        data_centred = np.array(emb - emb.mean(axis=0))
        covariance_mat = np.dot(data_centred.T,data_centred)
        principal_scores = pca.transform(covariance_mat) 
        principal_vectors = pca.components_  # Shape: (n_comp, 768)
        eig_val = pca.explained_variance_ratio_
        out = torch.tensor(principal_scores.flatten().reshape(1, -1))
        X_test = torch.cat([X_test, out], dim=0)

        # Select only needed columns for outputs
        mean_Y_test = Y[output_columns+['quarter and year']]
        # Consider only quarters for which we have transcripts
        mean_Y_test = mean_Y_test[mean_Y_test['quarter and year'].isin(top_N_data_test['quarter and year'].unique())]
        # Create output Torch tensor
        Y_test = mean_Y_test[output_columns]

    return X_train.detach().numpy(), X_test.detach().numpy(), Y_train, Y_test, eig_val

def compute_similarity(top_N_data):
    """
    Compute average pairwise cosine similarity among texts in top_N_data.

    Args:
        top_N_data: Pandas DataFrame (should contain columns: 'embeddings')

    Returns:
        average_pairwise_cosine_similarity
    """

    text_list = [row['embeddings'] for _,row in top_N_data.iterrows()]
    text_list = torch.tensor(text_list)
    pre_saved_similarity = 1.0
    if len(text_list) == 1:
        # If only one embedding, use the pre-saved similarity value
        average_pairwise_cosine_similarity = pre_saved_similarity
    else:
        # Convert list of embeddings to a tensor
        text_list = torch.tensor(text_list)
        
        # Normalize the embeddings
        normalized_embeddings = torch.nn.functional.normalize(text_list.to(dtype=torch.float32), p=2, dim=1)
        
        # Compute cosine similarity matrix
        cosine_similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)
        
        # Get pairwise cosine similarities from the upper triangle of the matrix
        num_embeddings = len(text_list)
        triu_indices = torch.triu_indices(num_embeddings, num_embeddings, offset=1)
        pairwise_cosine_similarities = cosine_similarity_matrix[triu_indices[0], triu_indices[1]]
        
        # Compute the average pairwise cosine similarity
        average_pairwise_cosine_similarity = pairwise_cosine_similarities.mean().item()

    return average_pairwise_cosine_similarity


def get_window(dataset, Y, N, start_month, start_year, end_month, end_year, output_columns, by='mean', n_comp=10):
    """
    Extract training, validation and test sets from dataset corresponding 
    to the required period of time.

    Args:
        dataset: Pandas DataFrame of covariates
        Y: Pandas DataFrame of output variable
        N: number of top capitalized companies to consider
        start_month: starting month for training
        start_year: starting year for training
        end_month: end month for training
        end_year: end year for training
        output_columns: list of valid columns in Y
        by: set 'mean' if aggregation in quarter is by averaging 
                    the embeddings, PCA for PCA (default 'mean')
        n_comp: number of directions for PCA (necessary only if by='PCA', 
                    default 10)

    Returns:
        X_train: Torch Tensor of training covariates (shape (n_train,768), 
                                n_train=number of transcripts in train set)
        X_val: Torch Tensor of validation covariates (shape (n_val,768))
        X_test: Torch Tensor of test covariates (shape (n_test,768))
        Y_train: Torch Tensor of training output (shape (n_train,out_dim), 
                                        out_dim=number of output columns)
        Y_val: Torch Tensor of validation output (shape (n_val,out_dim))
        Y_test: Torch Tensor of test output (shape (n_test,out_dim))
        average_pairwise_cosine_similarity: cosine similarity for covariates (texts) 
                                        in test set (real number)
    """

    # Select only data about top N companies
    top_N_data_train = get_largest_N_mktcap(dataset, N, start_month, start_year, end_month, end_year)
    eig_val = 0
    if by == 'mean':
        X_train, Y_train = create_by_mean(top_N_data_train, Y, output_columns)
        X_train = torch.tensor(X_train)
        Y_train = torch.tensor(Y_train.values, dtype=torch.float32)
    elif by == 'PCA':
        X_train, _, Y_train, _, eig_val = create_by_PCA(top_N_data_train, top_N_data_train, Y, output_columns, n_comp)
        X_train = torch.tensor(X_train)
        Y_train = torch.tensor(Y_train.values, dtype=torch.float32)

    # Build validation sets
    val_start_month, val_start_year, val_end_month, val_end_year = get_next_quarter(end_month, end_year)
    print(f"Validation period: {val_start_month}/{val_start_year} - {val_end_month}/{val_end_year}")
    top_N_data_val = get_largest_N_mktcap(dataset, N, val_start_month, val_start_year, val_end_month, val_end_year)
    if by == 'mean':
        X_val, Y_val = create_by_mean(top_N_data_val, Y, output_columns)
        X_val = torch.tensor(X_val)
        Y_val = torch.tensor(Y_val.values, dtype=torch.float32)
    elif by == 'PCA':
        X_train, X_val, Y_train, Y_val, _ = create_by_PCA(top_N_data_train, top_N_data_val, Y, output_columns, n_comp, training=False)
        X_val = torch.tensor(X_val)
        Y_val = torch.tensor(Y_val.values, dtype=torch.float32)

    # Built test sets
    test_start_month, test_start_year, test_end_month, test_end_year = get_next_quarter(val_end_month, val_end_year)
    print(f"Test period: {test_start_month}/{test_start_year} - {test_end_month}/{test_end_year}")
    top_N_data_test = get_largest_N_mktcap(dataset, N, test_start_month, test_start_year, test_end_month, test_end_year)
    if by == 'mean':
        X_test, Y_test = create_by_mean(top_N_data_test, Y, output_columns)
        X_test = torch.tensor(X_test)
        Y_test = torch.tensor(Y_test.values, dtype=torch.float32)
    elif by == 'PCA':
        X_train, X_test, Y_train, Y_test, _ = create_by_PCA(top_N_data_train, top_N_data_test, Y, output_columns, n_comp, training=False)
        X_test = torch.tensor(X_test)
        Y_test = torch.tensor(Y_test.values, dtype=torch.float32)
        
    # Similarity computation
    average_pairwise_cosine_similarity = compute_similarity(top_N_data_test)
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, average_pairwise_cosine_similarity, eig_val


def validation(dataloader_train, dataloader_val, criterion, num_epochs, 
               num_hidden_layers, hidden_sizes, learning_rates, tol, 
               input_size, output_size, device, epochs_limit, model_type='NN'):
    """
    Perform validation on number of hidden layers, hidden size 
    of the NN and learning rates.

    Args:
        dataloader_train: DataLoader of the train set
        dataloader_val: DataLoader of the validation set
        criterion: criterion for the loss
        num_epochs: number of epochs
        num_hidden_layers: list of hidden layers to validate on
        hidden_sizes: list of hidden sizes to validate on
        learning_rates: list of learning rates to validate on
        tol: tolerance of loss decrease to stop epochs
        input_size: input size of the NN
        output_size: output size of the NN
        device: device for computations (e.g. 'cpu')
        epochs_limit: max number of epochs
        model_type: type of model to adopt (choose between 'NN' and 
                    'linear', default 'NN')

    Returns:
        best_nl_idx: index of the best number of hidden layers
        best_hs_idx: index of the best hidden size
        best_lr_idx: index of the best learning rate
        losses: all losses computed during validation
    """

    # Initialize array containing losses
    losses = np.empty((len(num_hidden_layers),len(hidden_sizes), len(learning_rates)))
    loss = 1e4
    wd = 0.01
    # Loop over validation parameters
    for i, nl in enumerate(num_hidden_layers):
        for j, hs in enumerate(hidden_sizes):
            for k, lr in enumerate(learning_rates):
                if model_type=='NN':
                    current_model = NeuralNetwork(input_size,hs,output_size,nl).to(device)
                elif model_type=='linear':
                    current_model = BaselineNeuralNetwork(input_size,output_size).to(device)
                optimizer = AdamW(current_model.parameters(), lr, weight_decay=wd)
                # Train model
                train_model(current_model, dataloader_train, criterion, optimizer, num_epochs, tol, device, epochs_limit)
                # Evaluate on validation set
                _, loss = evaluate_model(current_model, dataloader_val, criterion, device)
                # Update losses
                losses[i][j][k] = loss

    # Find best indexes
    argmin_flat = np.argmin(losses)
    best_nl_idx, best_hs_idx, best_lr_idx = np.unravel_index(argmin_flat, losses.shape)
    print("Loss for validation:", np.min(losses), "obtained with:", )

    return best_nl_idx, best_hs_idx, best_lr_idx, losses


def rolling_window(dataset, Y, N, start_month, start_year, end_month, end_year, num_epochs, 
                   bs, output_columns, learning_rates, hidden_sizes, num_hidden_layers, tol, 
                   input_size, output_size, criterion, device, epochs_limit, by='mean', n_comp=10, model_type='NN'):
    """
    Perform rolling window on dataset from start time to end time.

    Args:
        dataset: Pandas DataFrame of inputs
        Y: Pandas DataFrame of outputs
        N: number of top capitalized companies to consider
        start_month: start month for rolling window
        start_year: start year for rolling window
        end_month: end month for rolling window
        end_year: end year for rolling window
        num_epochs: number of minimum epochs to have 
                    stable losses
        bs: batch size
        output_columns: valid columns for Y
        learning_rates: list of learning rates to validate on
        hidden_sizes: list of hidden sizes to validate on
        num_hidden_layers: list of hidden layers to validate on
        tol: tolerance of loss decrease to stop epochs
        input_size: input size of the NN
        output_size: output size of the NN
        criterion: criterion for the loss
        device: device for computations (e.g. 'cpu')
        epochs_limit: max number of epochs
        by: set 'mean' if aggregation in quarter is by averaging 
                    the embeddings, PCA for PCA (default 'mean')
        n_comp: number of directions for PCA (necessary only if by='PCA', 
                    default 10)
        model_type: type of model to adopt (choose between 'NN' and 
                    'linear', default 'NN')

    Returns: predictions, test_losses, test_values, similarities, failed_idxs, eigenvalues
        predictions: predicted values
        test_losses: losses obtained with evaluation on test set
        train_losses: losses obtained during training
        val_losses: losses obtained during validation
        test_values: values of test outputs
        similarities: list of computed similarities
        eigenvalues: list of explained variability with PCA (0 if by='mean')
    """
    
    # Initialize training local start month and year
    local_start_month = start_month
    local_start_year = start_year
    
    test_values = []
    predictions = []
    train_losses = []
    val_losses = []
    test_losses = []
    similarities = []
    eigenvalues = []
    
    # Loop over all quarters
    for i in range(24):

        # Get needed data structures for inputs and outputs
        local_end_month, local_end_year = get_end_train(local_start_month, local_start_year)
        print(f"Training period: {local_start_month}/{local_start_year} - {local_end_month}/{local_end_year}")
        X_train, X_val, X_test, Y_train, Y_val, Y_test, similarity, eig_val = get_window(dataset, Y, N, local_start_month, local_start_year, local_end_month, local_end_year, output_columns, by, n_comp)
        X_train, Y_train = X_test.detach(), Y_test.detach()
        X_val, Y_val = X_val.detach(), Y_val.detach()
        X_test, Y_test = X_test.detach(), Y_test.detach()

        # Training DataLoader
        dataset_db_train = TensorDataset(X_train, Y_train)
        dataloader_train = DataLoader(dataset_db_train, batch_size=bs, shuffle=False)

        # Validation DataLoader
        dataset_db_val = TensorDataset(X_val, Y_val)
        dataloader_val = DataLoader(dataset_db_val, batch_size=bs, shuffle=False)
        
        # Test DataLoader
        dataset_db_test = TensorDataset(X_test, Y_test)
        dataloader_test = DataLoader(dataset_db_test, batch_size=bs, shuffle=False)

        # Perform validation
        best_nl_idx, best_hs_idx, best_lr_idx, val_loss = validation(dataloader_train, dataloader_val, criterion, 
                                                            num_epochs, num_hidden_layers, hidden_sizes, 
                                                            learning_rates, tol, input_size, output_size,
                                                            device, epochs_limit, model_type)
        print("Best number of layers:", num_hidden_layers[best_nl_idx])
        print("Best hidden size:", hidden_sizes[best_hs_idx])
        print("Best learning rate:", learning_rates[best_lr_idx])

        # Train model
        if model_type=='NN':
            final_model = NeuralNetwork(input_size, hidden_sizes[best_hs_idx], output_size, num_hidden_layers[best_nl_idx]).to(device)
        elif model_type=='linear':
            final_model = BaselineNeuralNetwork(input_size, output_size).to(device)
        final_optimizer = AdamW(final_model.parameters(), learning_rates[best_lr_idx], weight_decay=0.01)
        window_losses = train_model(final_model, dataloader_train, criterion, final_optimizer, num_epochs, tol, device, epochs_limit)

        # Evaluate model
        outputs, test_loss = evaluate_model(final_model, dataloader_test, criterion, device)

        # Update variables
        predictions.append(outputs.item())
        train_losses.append(window_losses)
        test_losses.append(test_loss)
        val_losses.append(val_loss)
        similarities.append(similarity)
        test_values.append(Y_test.item())
        eigenvalues.append(eig_val)

        try:
            # Get next local start month and year for training
            local_start_month, local_start_year = get_next_iteration(local_start_month, local_start_year)
        except:
            print("Could not retrieve local_start_month and local_start_year for iteration:", i)
            break

    return predictions, test_losses, train_losses, val_losses, test_values, similarities, eigenvalues