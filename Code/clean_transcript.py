# fix_encoding.py

from ftfy import fix_text

# Input and output file paths
input_file = "transcription.txt"
output_file = "consult10.txt"

# Read corrupted text
with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
    corrupted_text = f.read()

# Fix encoding issues
fixed_text = fix_text(corrupted_text)

# Save the cleaned text
with open(output_file, "w", encoding="utf-8") as f:
    f.write(fixed_text)

print("✅ Fixed text saved to", output_file)
