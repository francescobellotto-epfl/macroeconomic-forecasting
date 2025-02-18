import numpy as np
import pandas as pd

def get_useful_transcripts(texts_df, texts_df_col, useful_transcripts_df, transcript_id_col):
    """
    Among transcripts'texts in texts_df, consider only the ones for 
    which we have data in useful_transcripts_df.

    Arguments:
        texts_df: Pandas DataFrame containing transcripts texts
        texts_df_col: valid column name for transcript IDs in texts_df
        useful_transcripts_df: Pandas DataFrame containing data only 
        about useful transcripts
        transcript_id_col: valid column name for transcript IDs in 
        useful_transcripts_df

    Output:
        texts_df: modified DataFrame of texts
    """

    if texts_df_col not in texts_df.columns or \
    transcript_id_col not in useful_transcripts_df.columns:
        raise Exception("Please enter a valid column names")
    
    try:
        trancript_id_list = useful_transcripts_df[transcript_id_col].unique()
        texts_df = texts_df[texts_df[texts_df_col].isin(trancript_id_list)]
    except:
        print("Something went wrong: could not select useful transcripts.")
    
    return texts_df


def concat_transcripts(texts_df, transcript_id_col, text_col):
    """
    Concatenate transcripts that are split on different rows.

    Arguments:
        texts_df: Pandas DataFrame containing transcripts texts
        transcript_id_col: valid column name for transcript IDs in texts_df
        text_col: valid column name for texts in texts_df

    Output:
        texts_df: modified DataFrame of texts
    """

    try:
        texts_df = (
        texts_df.groupby(transcript_id_col, as_index=False)
        .agg({text_col: lambda x: ' '.join(filter(None, x))})
        )
    except:
        print("Something went wrong: could not concatenate transcripts.")

    return texts_df


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

def get_latest_version(data, company_ids, event_ids, version_ids, mapping):
    """
    Get most updated version of transcripts for each company and event.

    Args:
        data: pandas DataFrame 
        company_ids: valid column name of company identifier
        event_ids: valid column name of event identifier
        version_ids: valid column name of version identifier
        mapping: dictionary mapping versions in ascending order 
        (1 is the most updated version)

    Returns:
        new_data: pandas DataFrame with most updated version of transcripts

    """
    new_data = data.groupby([company_ids, event_ids, version_ids]).agg(lambda x: x.iloc[0])
    new_data = new_data.reset_index()
    new_data[version_ids] = new_data[version_ids].map(mapping)
    new_data = new_data.sort_values(by=version_ids)
    new_data = new_data.groupby([company_ids, event_ids]).first()
    return new_data.reset_index()