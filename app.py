from flask import Flask, render_template, request
from pathlib import Path
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import torch

app = Flask(__name__)

# Path to your saved model and tokenizer
model_path = Path("./model")

# Load the model and tokenizer
model = MBartForConditionalGeneration.from_pretrained(model_path)
tokenizer = MBart50Tokenizer.from_pretrained(model_path)

# Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        text = request.form["text"]
        language = request.form["language"]

        # Set the source and target language codes
        lang_code = "fr_XX" if language == "fr" else "en_XX"
        tokenizer.src_lang = lang_code
        tokenizer.tgt_lang = lang_code

        # Prepare the input for the model
        inputs = tokenizer(
            text, return_tensors="pt", max_length=512, truncation=True
        ).to(device)

        # Generate the summary
        summary_ids = model.generate(
            **inputs, max_length=128, num_beams=5, early_stopping=True
        )

        # Decode the generated tokens back to text
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return render_template("index.html", summary=summary)


if __name__ == "__main__":
    app.run(debug=True)
