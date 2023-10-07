#bigquery
from google.cloud import bigquery

#standard libraries
import re
from typing import List
import pandas as pd


def read_data_from_bq(bq_uri: str) -> pd.DataFrame:
    '''
    A function to read data from bigquery
    
    INPUT
    :bq_uri: bigquery URI, has to start with bq://
    
    OUTPUT
    :df: pandas dataframe of the table from bigquery
    '''
    
    #check if the URI is correct
    if not bq_uri.startswith('bq://'):
        raise Exception("uri is not a BQ uri. It should be bq://project_id.dataset.table")
        
    #getting project, dataset, and table
    project,dataset,table =  bq_uri.split(".")
    project_id = project[5:]
    
    #initializing bigquery client
    bq_client = bigquery.Client(project=project[5:])
    
    #query -> consider making it a variable (?)
    sql_query = f'SELECT * FROM `{project_id}.{dataset}.{table}`'
    
    #read the query and save it into a variable
    df = bq_client.query(sql_query).to_dataframe()
    return df

def save_to_gcs(project_id: str, bucket_name: str, blob_path: str, variable):
    """
    A function to save a variable to GCS
    
    INPUT
    :project_id: string, project id for current project
    :bucket_name: string, GCS bucket name
    :blob_path: string, path to file that will be saved
    :variable: the variable to be saved
    """
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob_bytes = pickle.dumps(variable)
    blob.upload_from_string(blob_bytes)
    print(f'Variable uploaded to gs://{bucket_name}/{tokens_path}')

def load_from_gcs(project_id: str, bucket_name: str, blob_path: str):
    """
    A function to load pickle file from GCS
    
    INPUT
    :project_id: string, project id for current project
    :bucket_name: string, GCS bucket name
    :blob_path: string, path to file that will be loaded
    
    OUTPUT
    :loaded_var: loaded pickle file saved in a variable
    """
    client = storage.Client(project=project_id)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)
    pickle_bytes = blob.download_as_bytes()
    loaded_var = pickle.loads(pickle_bytes)
    return loaded_var

def replace_pattern(pattern, text:str):
    """
    Funtion to replace text containing specific regex pattern with space
    """
    return re.sub(pattern, ' ', text)

def clean_text(df: pd.DataFrame, text_col: str, patterns: List):
    """
    Function to clean the text column. 
    
    INPUT
    :df: pandas dataframe
    :text_col: the name of the column containing text to be processed, datatype string
    :patterns: a list containing regex patterns used to do various things to the text column
    
    OUTPUT
    :df: pandas dataframe with the text column already cleaned
    """
    
    #initializing tqdm for pandas
    
    #converting the text column into lowercase
    df[text_col] = df[text_col].str.lower()
    
    #clean the text column using regex
    for i, pattern in enumerate(patterns):
        df[text_col] = df[text_col].apply(lambda text: replace_pattern(pattern, text))
    return df

def preprocess_data(df: pd.DataFrame, text_col: str, target_col: str) -> pd.DataFrame:
    '''
    A function to preprocess the text and target column using clean_text and replace_pattern functions
    
    INPUT
    :df: pandas dataframe containing text and label
    :text_col: a string, the column containing the text in the dataframe
    :target_col: a string, the column containing the label (preprocessed) in the dataframe
    
    OUTPUT
    :df: pandas dataframe containing preprocessed and cleaned text and label
    '''
    #import libraries
    import re
    import pandas as pd
    
    #preprocessing variables
    product_mapping = {
        "Credit reporting, credit repair services, or other personal consumer reports": "credit_reporting",
        "Debt collection": "debt_collection",
        "Mortgage": "mortgage",
        "Credit reporting": "credit_reporting",
        "Credit card or prepaid card": "credit_prepaid_card",
        "Checking or savings account": "bank_account",
        "Credit card": "credit_prepaid_card",
        "Bank account or service": "bank_account",
        "Student loan": "personal_loan",
        "Money transfer, virtual currency, or money service": "money_transfer",
        "Consumer Loan": "consumer_loan",
        "Vehicle loan or lease": "personal_loan",
        "Payday loan, title loan, or personal loan": "personal_loan",
        "Payday loan": "personal_loan",
        "Money transfers": "money_transfer",
        "Prepaid card": "credit_prepaid_card",
        "Other financial service": "others",
        "Virtual currency": "money_transfer"
    }
    patterns = [r'[Xx]{2,}', r"[^\w\d'\s]+", "\d+", ' +']
    
    #cleaning data
    df = df[df['product']!='product']
    
    #mapping target column using product mapping
    df.replace({target_col:product_mapping}, inplace=True)
    
    #cleaning text column
    df = clean_text(df, text_col, patterns)
    
    return df

def after_subplot(ax, group_name, x_label):
    """Add title xlabel and legend to single chart"""
    ax.set_title(group_name)
    ax.set_xlabel(x_label)
    ax.legend(loc="center right")

    if group_name.lower() == "loss":
        ax.set_ylim([None, None])