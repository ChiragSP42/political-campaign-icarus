import requests
import pandas as pd
import io

DATASET_DOI = "doi:10.7910/DVN/NT66Z3" 
DATAVERSE_API_URL = "https://dataverse.harvard.edu/api"

def pull_virginia_precinct_data():
    print("1. Querying the Dataverse digital catalog...")
    
    metadata_url = f"{DATAVERSE_API_URL}/datasets/:persistentId/?persistentId={DATASET_DOI}"
    response = requests.get(metadata_url)
    response.raise_for_status() 
    
    dataset_info = response.json()
    files = dataset_info['data']['latestVersion']['files']
    
    va_file_id = None
    file_name = ""
    
    # THE FIX: Robust, case-insensitive string matching
    for file in files:
        # Convert the file name to lowercase immediately
        current_name = file['dataFile']['filename'].lower()
        
        # Look for explicit bounds around 'va' to avoid matching 'Nevada'
        if '-va-' in current_name or '_va_' in current_name or 'virginia' in current_name:
            va_file_id = file['dataFile']['id']
            # Save the original casing for later logging
            file_name = file['dataFile']['filename'] 
            print(f"2. Found Virginia data! File ID: {va_file_id} ({file_name})")
            break
            
    if not va_file_id:
        raise ValueError("Could not locate Virginia data in this dataset.")

    print("3. Downloading the raw returns...")
    download_url = f"{DATAVERSE_API_URL}/access/datafile/{va_file_id}"
    csv_response = requests.get(download_url)
    csv_response.raise_for_status()
    
    # THE FIX: Handle both .tsv and .tab file extensions correctly
    separator = '\t' if 'tab' in file_name.lower() or 'tsv' in file_name.lower() else ','
    
    df = pd.read_csv(io.StringIO(csv_response.text), sep=separator, low_memory=False)
    
    print(f"\nSuccess! Loaded {len(df)} precinct records for Virginia.")
    return df

if __name__ == "__main__":
    va_df = pull_virginia_precinct_data()
    # print(va_df[['precinct', 'office', 'party_simplified', 'votes']].head())
    print(va_df.head())