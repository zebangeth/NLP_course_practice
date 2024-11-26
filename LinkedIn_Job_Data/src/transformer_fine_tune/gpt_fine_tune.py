import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

def upload_file(file_path, purpose):
    response = openai.File.create(
        file=open(file_path, "rb"),
        purpose=purpose
    )
    print(f"Uploaded {file_path}. File ID: {response['id']}")
    return response['id']

training_file_id = upload_file("training_data.jsonl", "fine-tune")
validation_file_id = upload_file("validation_data.jsonl", "fine-tune")

response = openai.FineTuningJob.create(
    model="gpt-4o-mini-2024-07-18",
    training_file=training_file_id,
    validation_file=validation_file_id,
    hyperparameters={
        "n_epochs": 3
    }
)

# Output fine-tuning job details
print(f"Fine-tuning job created with ID: {response['id']}")
print(f"https://platform.openai.com/fine-tune/jobs/{response['id']}")
