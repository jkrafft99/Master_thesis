import requests
import json

# ==== USER SETTINGS ====
API_KEY = "I7O4xUw0JI1WxGJVjgIsyFMIoQXJIfqC"
AUDIO_FILE_PATH = "consult10.wav"
LEXICON_FILE = "medical_lexicon_original.txt"
LANGUAGE = "nl"  # Dutch
# =======================

# Load terms from file
with open(LEXICON_FILE, "r", encoding="utf-8") as f:
    terms = [line.strip() for line in f if line.strip()]

# Construct job config
config = {
    "type": "transcription",
    "transcription_config": {
        "language": LANGUAGE,
        "additional_vocab": terms
    }
}

# Submit transcription job
url = "https://asr.api.speechmatics.com/v2/jobs/"
headers = {"Authorization": f"Bearer {API_KEY}"}
files = {
    "config": (None, json.dumps(config), "application/json"),
    "data_file": open(AUDIO_FILE_PATH, "rb")  # ✅ Correct key
}

print("Submitting job...")
response = requests.post(url, headers=headers, files=files)
job = response.json()
print("Job submitted:", job)

job_id = job["id"]

# Polling for completion
import time

print("Waiting for transcription...")
status_url = f"https://asr.api.speechmatics.com/v2/jobs/{job_id}"
while True:
    status_resp = requests.get(status_url, headers=headers).json()
    if status_resp["job"]["status"] == "done":
        break
    elif status_resp["job"]["status"] == "failed":
        raise Exception("Job failed.")
    time.sleep(5)

# Get transcription result
transcript_url = f"{status_url}/transcript?format=txt"
transcript = requests.get(transcript_url, headers=headers).text

# Save output
with open("transcription.txt", "w", encoding="utf-8") as f:
    f.write(transcript)

print("✅ Transcription complete! Check transcription.txt")