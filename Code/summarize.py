import argparse
import openai
import google.generativeai as genai

# ==== SETTINGS ====
GEMINI_API_KEY = "AIzaSyDk88GIO5tvwPzgPQz2SDr_bRl24q09_9Y"
OPENAI_API_KEY = "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"
INPUT_FILE = "transcription2.txt"
OUTPUT_FILE = "summary.txt"
# ===================

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Summarize medical consultation with Gemini or GPT.")
parser.add_argument("--model", choices=["gemini", "gpt"], required=True, help="Choose model: 'gemini' or 'gpt'")
args = parser.parse_args()

# Load the transcription
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    transcription = f.read()

# Build the structured prompt
prompt = f"""
Je bent een medische assistent.
Maak een samenvatting van dit medische consult.
Gebruik de volgende structuur:

1. Symptomen (klachten, duur, ernst)
2. Voorgeschiedenis (bestaande aandoeningen, medicatiegebruik)
3. Diagnose (voorlopige of definitieve diagnose)
4. Advies (behandeling, medicijnen, leefstijladvies)
5. Follow-up (wanneer terugkomen of verdere onderzoeken)

Houd het kort, feitelijk, en in het Nederlands.
Gebruik opsommingen waar nodig.

Consulttekst:
{transcription}
"""

# === Summarize based on selected model ===
summary = ""

if args.model == "gemini":
    # Setup Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro')

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=800
        )
    )
    summary = response.text

elif args.model == "gpt":
    # Setup GPT
    openai.api_key = OPENAI_API_KEY

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Or "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "Je bent een medische samenvatter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=800
    )
    summary = response.choices[0].message.content

# Save the summary to file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"✅ Summary saved to {OUTPUT_FILE} using {args.model.upper()}")
