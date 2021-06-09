### knn_fxns.py
import numpy as np
import pandas as pd

def row_distance(row1, row2):
    '''
    Returns the distance between two input rows, extracted from a Pandas dataframe
    INPUT: two rows which should be Pandas series or array-type, not data frame 
    OUTPUT: Euclidean disstance
    '''
    pt1=np.array(row1)
    pt2=np.array(row2)
    dist=np.sqrt(sum((pt1-pt2)**2))
    return dist

def calc_distance_to_all_rows(df,example_row ):
    '''
    Computes distance between every row in input df (Pandas dataframe) and example_row (Pandas series or array type)
    Calls 'row_distance'
    INPUT: df, Pandas dataframe; example_row
    OUTPUT:Pandas dataframe with additional column 'distance_to_ex' added to input dataframe df
    '''
    distances=[]
    attribute_df=df.drop(['class'],axis=1)
    num_rows=df.shape[0]
    for row_num in np.arange(num_rows):
        current_row=attribute_df.iloc[row_num,:]
        current_distance=row_distance(current_row,example_row)
        distances.append(current_distance)
    return_df=df.assign(distance_to_ex=distances)
    return return_df
        

def find_k_closest(df, example_row, k):
    """
    Finds the k closest neighbors to example, excluding the example itself.
    Calls 'calc_distance_to_all_rows'
    IF there is a tie for kth closest, choose the final k to include via random choice.
    INPUT: df, Pandas dataframe; example_row, Pandas series or array type; k, integer number of nearest neighbors.
    OUTPUT: dataframe in same format as input df but with k rows and sorted by 'distance_to_ex.'
    """
    ret_df=calc_distance_to_all_rows(df,example_row)
    sort_df=ret_df.sort_values(by=['distance_to_ex'])
    return sort_df[0:k]

def get_sorted_k_counts(ksorted_df):
    grouped_df=ksorted_df.groupby(['class']).size().reset_index(name='counts')
    sorted_df=grouped_df.sort_values(by=['counts'],ascending=False)
    return sorted_df

def majority_class(sorted_df):
    return sorted_df['class'].iloc[0]

def classify(df, example_row, k):
    """
    Return the majority class from the k nearest neighbors of example
    Calls 'find_k_closest'
    INPUT: df, Pandas dataframe; example_row, Pandas series or array type; k, integer number of nearest neighbors
    OUTPUT: string referring to closest class.
    """
    k_df= find_k_closest(df,example_row,k)
    sorted_df=get_sorted_k_counts(k_df)
    return majority_class(sorted_df)

def evaluate_accuracy(training_df, test_df,k):
    '''Returns the proportion of the test_df that was correctly classified'''
    test_attribute_df=test_df.drop(['class'],axis=1)
    num_correct=0
    num_rows=test_df.shape[0]
    
    for row_num in np.arange(num_rows):
        example_row=test_attribute_df.iloc[row_num,:]
        c=classify(training_df,example_row,k)
        if c==test_df['class'].iloc[row_num]:
            num_correct=num_correct+1
    return num_correct/num_rows