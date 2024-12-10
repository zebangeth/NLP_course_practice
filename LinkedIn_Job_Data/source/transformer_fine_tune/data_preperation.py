import pandas as pd
import json
import math

def split_csv_to_jsonl(input_csv, training_jsonl, validation_jsonl, split_ratio=0.8, is_real_data=False):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv)

    # Select only the first 5000 rows for faster training
    df = df.head(5000)

    # Shuffle the data to ensure randomness
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    med_salary = "annual_med_salary" if is_real_data else "med_salary"

    # Calculate split index
    split_index = math.ceil(len(df) * split_ratio)

    # Split the data into training and validation sets
    training_data = df.iloc[:split_index]
    validation_data = df.iloc[split_index:]

    # Helper function to write data to JSONL
    def write_to_jsonl(data, file_path):
        with open(file_path, 'w') as jsonl_file:
            for _, row in data.iterrows():
                # Prepare the data in the required format
                training_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an assistant that predicts the median salary for a job based on its description."
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Job Title: {row['title']}\n"
                                f"Location: {row['location']}\n"
                                f"Description: {row['description']}"
                            )
                        },
                        {
                            "role": "assistant",
                            "content": f"{row[med_salary]}"
                        }
                    ]
                }
                # Write the training example to the JSONL file
                jsonl_file.write(json.dumps(training_example) + '\n')

    # Write training and validation data to JSONL files
    write_to_jsonl(training_data, training_jsonl)
    write_to_jsonl(validation_data, validation_jsonl)

    print(f"Training data saved to {training_jsonl}")
    print(f"Validation data saved to {validation_jsonl}")

input_csv = "../synthetic_data/synthetic_job_data.csv"
training_jsonl = "syn_job_data_training.jsonl"
validation_jsonl = "syn_job_data_validation.jsonl"
split_csv_to_jsonl(input_csv, training_jsonl, validation_jsonl, is_real_data=False)

input_csv = "../real_data/linkedin_jobs_preprocessed.csv"
training_jsonl = "real_job_data_training.jsonl"
validation_jsonl = "real_job_data_validation.jsonl"
split_csv_to_jsonl(input_csv, training_jsonl, validation_jsonl, is_real_data=True)