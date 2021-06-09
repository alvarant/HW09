### knn_fxns.py
import numpy as np
import pandas as pd

def row_distance(row1, row2):
    '''
    Returns the distance between two input rows, extracted from a Pandas dataframe
    INPUT: two rows which should be Pandas series or array-type, not data frame 
    OUTPUT: Euclidean disstance
    '''
    

def calc_distance_to_all_rows(df,example_row ):
    '''
    Computes distance between every row in input df (Pandas dataframe) and example_row (Pandas series or array type)
    Calls 'row_distance'
    INPUT: df, Pandas dataframe; example_row
    OUTPUT:Pandas dataframe with additional column 'distance_to_ex' added to input dataframe df
    '''
    
    

def find_k_closest(df, example_row, k):
    """
    Finds the k closest neighbors to example, excluding the example itself.
    Calls 'calc_distance_to_all_rows'
    IF there is a tie for kth closest, choose the final k to include via random choice.
    INPUT: df, Pandas dataframe; example_row, Pandas series or array type; k, integer number of nearest neighbors.
    OUTPUT: dataframe in same format as input df but with k rows and sorted by 'distance_to_ex.'
    """
    
    
def classify(df, example_row, k):
    """
    Return the majority class from the k nearest neighbors of example
    Calls 'find_k_closest'
    INPUT: df, Pandas dataframe; example_row, Pandas series or array type; k, integer number of nearest neighbors
    OUTPUT: string referring to closest class.
    """
    
