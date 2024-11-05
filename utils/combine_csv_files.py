import os
import pandas as pd

def combine_csv_files(directory, output_file):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    combined_df = pd.DataFrame()

    for i, file in enumerate(csv_files):
        file_path = os.path.join(directory, file)
        if i == 0:
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path, header=0)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.to_csv(output_file, index=False)


directory = 'data/labeled'
output_file = 'labeled_messages.csv'
combine_csv_files(directory, output_file)
