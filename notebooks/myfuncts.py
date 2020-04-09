import pandas as pd
import numpy as np
import re
from fractions import Fraction

# import plot libraries
import seaborn as sns
sns.set_palette('Set2')
import matplotlib.pyplot as plt


# list number of files
import os, os.path
from pathlib import Path
import datetime



# General functions defined

def fn_drop_columns(cols_todrop, df):
    """drop unwanted columns. inplace=False for raw data"""
    df = df.drop(df.columns[cols_todrop], axis=1, inplace=False)
    return df

def fn_check_missing_data(df):
    """check for any missing data in the df (display in descending order)"""
    return df.isnull().sum().sort_values(ascending=False)

def fn_remove_col_white_space(df,col):
    """remove white space at the beginning of string""" 
    df[col] = df[col].str.lstrip()
    
def fn_convert_str_datetime(df, str_date_col_name, converted_name_column):
    """
    Convert a string based date column to a datetime64[ns] timestamp without the timezone.
    To include the UTC timezone remove the .apply section. 
    """
    df[converted_name_column] = pd.to_datetime(df[str_date_col_name], 
                                               format='%Y-%m-%d %H:%M:%S').apply(lambda x: x.tz_localize(None))
    
def fn_col_mapping(df):
    """Return a list of column names and their associated column index number."""
    return [f'{c[0]}:{c[1]}' for c in enumerate(df.columns)]

def fn_col_mapping_dict(df):
    """Return a column mapping dictionary with column name and associated column index number.
    This can be used to assign new column names"""
    return {c[0]:c[1] for c in enumerate(df.columns)}

def fn_col_mapping_dict_for_rename(df):
    """Return a column mapping dictionary with column name and associated column index number.
    This can be used to assign new column names"""
    return {c[1]:c[1] for c in enumerate(df.columns)}


def fn_modification_date(filename):
    """Return the filename and the modification data. 
    Parameters
    ---------
    filename: Pass in Path(filepath variable)
    """
    fn = os.path.split(filename)[1]
    t = os.path.getmtime(filename)
    t = datetime.datetime.fromtimestamp(t)
    print(f'Source file: {fn} was modified on: {t}')


def fn_floatformat(decimals):
    dec = '{:.'+str(decimals)+'f}'
    pd.options.display.float_format = dec.format

def fn_col_value_from_key(df, cols):
    """
    Provide a dataframe and a list of column interger(s) to display. fn_col_mapping_dict(df) is called to get a dictionary
    of the dataframe columns. A list of column keys are iterated and their values are return. These values can be used 
    in iloc slicing.
    """
    df_cols_dict = fn_col_mapping_dict(df)
    col_names = [df_cols_dict[x] for x in cols]
    return col_names


def fn_df_unique_2cols(df, columns):
    """Returns a unique list in the last column of the dataframe based on the first 2 columns.
    Inputs:
    df: Dataframe of interest
    columns: List of 2 column names. First for grouping and second generate the unique values for.
    Notes: 2 columns must be based in. The unique values are seperated by '/' """
    # Rename to unique column so not to confuse with orginal name
    rename_unique_col = columns[1]+'_uq'
    df_unique = df[[columns[0], columns[1]]]\
                                            .drop_duplicates()\
                                            .groupby([columns[0]])[columns[1]]\
                                            .apply(list)\
                                            .reset_index()\
                                            .rename(columns={columns[1]: rename_unique_col})

    df_unique[rename_unique_col] = df_unique.apply(lambda x:(' / '.join([str(s) for s in x[rename_unique_col]])), axis=1)
    return df_unique

   
def fn_df_unique_3cols(df, columns):
    """Returns a unique list in the last column of the dataframe based on the first 2 columns.
    Inputs:
    df: Dataframe of interest
    columns: List of 3 column names. First 2 for grouping and last the column for unique values
    Notes: 3 columns must be based in. The unique values are seperated by '/' """
    # Rename to unique column so not to confuse with orginal name
    rename_unique_col = columns[2]+''
    df_unique = df[[columns[0], columns[1], columns[2]]]\
                                            .drop_duplicates()\
                                            .groupby([columns[0],columns[1]])[columns[2]]\
                                            .apply(list)\
                                            .reset_index()\
                                            .rename(columns={columns[2]: rename_unique_col})

    df_unique[rename_unique_col] = df_unique.apply(lambda x:(' / '.join([str(s) for s in x[rename_unique_col]])),
                                                   axis=1)
    return df_unique

def fn_pd_options_display(max_columns = 20, max_rows=50, max_colwidth=200):
    """Set Pandas Display options"""
    pd.options.display.max_columns = max_columns  # None -> No Restrictions
    pd.options.display.max_rows = max_rows    # None -> Be careful with this 
    pd.options.display.max_colwidth = max_colwidth

    
def fn_filter_column_all_search_words(df, column_to_search, search_words, cols_to_display):
    """Search a specified column for ALL search words passed in as a list. 
    Inputs:
    df: dataframe
    column_to_search: 1 column name in the dataframe as a string
    search_words: words as string within a list
    cols_to_display: pass in list of intergers for columns from the dataframe to display
    """
    return df.loc[df[column_to_search].apply(lambda lookin: all(word.upper() in lookin for word in search_words))].iloc[
            :, cols_to_display]


def fn_df_drop_blank_cols_row(df):
    """Input a dataframe and drop all null columns / rows and return a copy"""
    return df.dropna(how='all').dropna(axis=1, how='all').copy()


def fn_clean_header(df):
    """Pass in df and amend replacments accordingly"""
    # Note the order of replacment is not sequence hence '__' has to be added at the end to clean up double underscores
    header_replace = { (' ', '_'), ('\n', ''), ('/', ''), ('.', ''), ('(', '_'), (')', ''), ('#', 'num')}
    for k, v in header_replace:
        df.columns = df.columns.str.strip().str.lower().str.replace(k, v).str.replace('__', '_')
        
def fn_filter_words_in_column(df, col_to_search, search_words, any_words=True):
    """Filter for any (default) or all words in a dataframe column. Case insensitive.
    df: dataframe to search
    col_to_search: column name to search as a str
    search_words: list of search words
    any_words: Default True to search for any word. False will search to match all words.
    """
    if any_words == True:
        df_f = df.loc[df[col_to_search].str.lower().apply(
                lambda x: any(word.lower() in x for word in search_words))]
    else: 
        df_f = df.loc[df[col_to_search].str.lower().apply(
                lambda x: all(word.lower() in x for word in search_words))]
    return df_f

def fn_pivot_table(data, index_cols, agg_dict, margins=True, margins_name='TOTAL QTY'):
    """Pivot a dataframe. 
    data: dataframe
    index_cols = list of columns names
    agg_dict: dictionary of aggregate values
    margins: True or False
    margins_name: str
    values: set as keys of the agg_dict
    """
    return pd.pivot_table(data, index=index_cols, aggfunc=agg_dict, values=agg_dict.keys(),
              margins=margins, margins_name=margins_name)

def fn_keep_columns(df, cols_tokeep):
    """Return dataframe with columns passed in as a list of string names. """
    return df.filter(df.columns[cols_tokeep])

def fn_local_time_zone():
    """Return the local timezone"""
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(0))).astimezone().tzinfo