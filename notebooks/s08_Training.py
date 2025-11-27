from Imports import *
from s07_Limit_frequent_repetitions import (
    ds_train, ds_val, ds_test,
    balance_dataset,
)


def tokenize_example(batch: bool) -> BatchEncoding:
    """
    Tokenizes a batch of examples.

    Assumes that the input is a dictionary containing the key "translation".
    Two possible cases:
      - If "translation" is a dictionary with keys "en" and "fr", then
        the corresponding values are lists of texts.
      - Otherwise, if "translation" is a list, we assume each element is
        a pair [en_text, fr_text].

    If a text is empty, it is replaced by a single space (" ")
    to avoid returning an empty list and causing an error.
    """
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    translations = batch["translation"]

    # Case 1: translation is a dictionary
    if isinstance(translations[0], dict):
        en_texts = [ex.get("en", "").strip() for ex in translations]
        fr_texts = [ex.get("fr", "").strip() for ex in translations]
    # Case 2: translation is a list
    elif isinstance(translations[0], list):
        en_texts = [
            pair[0].strip()
            for pair in translations
            if isinstance(pair, list) and len(pair) >= 2
        ]
        fr_texts = [
            pair[1].strip()
            for pair in translations
            if isinstance(pair, list) and len(pair) >= 2
        ]
    else:
        raise ValueError("Unknown structure in 'translation'. Expected a dict or a list of pairs.")

    # Replace empty texts with a space
    en_texts = [text if text != "" else " " for text in en_texts]
    fr_texts = [text if text != "" else " " for text in fr_texts]

    # Tokenize source text (English)
    model_inputs = tokenizer(
        en_texts,
        max_length=128,
        truncation=True,
        padding="longest",
    )

    # Tokenize target text (French) in target mode
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            text_target=fr_texts,
            max_length=128,
            truncation=True,
            padding="longest",
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

####################################################################


ds_train = balance_dataset(ds_train)

# Apply batched tokenization to the datasets
tokenized_train = ds_train.map(tokenize_example, batched=True)
tokenized_val = ds_val.map(tokenize_example, batched=True)
tokenized_test = ds_test.map(tokenize_example, batched=True)



# Check if the model has already been trained
model_save_path = Path("../model")
if (model_save_path / "model.safetensors").exists():
    print(f"Model found at {model_save_path.resolve()}")
    print("No training needed.")
    print("If you want to train a new model, please delete this folder content"
          " before launching."
         )
else:
    # Load the pre-trained model and tokenizer for English → French translation
    model_name = "Helsinki-NLP/opus-mt-en-fr"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Create the DataCollator to handle dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Define improved training hyperparameters
    training_args = TrainingArguments(
        output_dir="../results",
        num_train_epochs=7,                    # Increased number of epochs
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,         # To simulate a larger batch size if needed
        per_device_eval_batch_size=4,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=500,
        save_steps=1000,
        save_total_limit=2,
        learning_rate=3e-5,                    # Lower learning rate for fine-tuning
        warmup_steps=1000,                     # Longer warmup period
        weight_decay=0.01,
        max_grad_norm=1.0,                     # Gradient clipping to stabilize training
        report_to="none",
        seed=42,
        fp16=True,
        load_best_model_at_end=True,           # Load the best model based on the metric
        metric_for_best_model="eval_loss",     # Use loss to determine the best model
        greater_is_better=False,               # Lower loss is better
    )

    # model.generation_config.update(  # because of a previous warning
    #     max_length=512,
    #     num_beams=4,
    #     bad_words_ids=[[59513]],
    # )

    # Create the Trainer with EarlyStoppingCallback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,   # 67,923 training examples
        eval_dataset=tokenized_val,      # 6,088 validation examples
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train the model
    start = time.time()
    trainer.train()
    print(
        "Elapsed time :",
        # time.ctime(time.time() - start)[11:19],
        time.strftime(
            '%Hh%Mm%Ss',
            time.gmtime(time.time() - start)),
    )

    # Save the model
    trainer.save_model(model_save_path)
    # Optional: Also save the tokenizer for future use
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path.resolve()}")