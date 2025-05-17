## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
The challenge is to build an NER system capable of identifying named entities (e.g., people, organizations, locations) in text, using a pre-trained BART model fine-tuned for this task. The system should be interactive, allowing users to input text and see the recognized entities in real-time.
### DESIGN STEPS:

#### STEP 1: Fine-tune the BART model
Start by fine-tuning the BART model for NER tasks. This involves training the model on a labeled NER dataset with text data that contains named entities (e.g., people, places, organizations).

#### STEP 2: Create an API for NER model inference
Develop an API endpoint that takes input text and returns the recognized entities using the fine-tuned BART model.

#### STEP 3: Integrate the API with Gradio
Build a Gradio interface that takes user input, passes it to the NER model via the API, and displays the results as highlighted text with identified entities.

### PROGRAM:
```py
# Required imports
import os
import io
import json
import requests
from PIL import Image
from dotenv import load_dotenv, find_dotenv
from IPython.display import Image, display, HTML
import gradio as gr

# Load API keys from .env file
_ = load_dotenv(find_dotenv())  # read local .env file
hf_api_key = os.environ['HF_API_KEY']

# Helper function for calling Hugging Face inference API
def get_completion(inputs, parameters=None, ENDPOINT_URL=os.environ['HF_API_SUMMARY_BASE']): 
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(data))
    return json.loads(response.content.decode("utf-8"))

# -------------------------------
# TEXT SUMMARIZATION SECTION
# -------------------------------

# Sample input text
text = (
    '''Founded in 2001, Saveetha Engineering College is a co-educational institution located in Thandalam, Chennai, just 8 km (5.0 mi) from the bustling township of Poonamallee. 
    The campus overlooks the scenic Chembarambakkam Lake and sits along the Chennai–Bangalore National Highway (NH4). Affiliated with Anna University, the largest technical university in India, 
    the college is approved by the AICTE and recognized by the Government of Tamil Nadu. It was established by the Saveetha Medical and Educational Trust, a registered charitable society committed to academic excellence. 
    In recognition of its academic growth, the college was granted autonomous status by the University Grants Commission (UGC). 
    In 2024, Saveetha Engineering College was ranked in the 201–300 band among engineering colleges in India by the National Institutional Ranking Framework (NIRF), 
    reinforcing its position as one of the emerging hubs of technical education in the country.'''
)

# Run summarization
get_completion(text)

# Define summarization function
def summarize(input):
    output = get_completion(input)
    return output[0]['summary_text']

# Gradio interface for summarization
gr.close_all()
demo = gr.Interface(
    fn=summarize,
    inputs=[gr.Textbox(label="Text to summarize", lines=6)],
    outputs=[gr.Textbox(label="Result", lines=3)],
    title="Text summarization with distilbart-cnn",
    description="Summarize any text using the `shleifer/distilbart-cnn-12-6` model under the hood!"
)
demo.launch(share=True)

# -------------------------------
# NAMED ENTITY RECOGNITION (NER)
# -------------------------------

# Update endpoint for NER
API_URL = os.environ['HF_API_NER_BASE']

# Sample NER call
text = "My name is Srisaran Karthik and I study at Saveetha Engineering College"
get_completion(text, parameters=None, ENDPOINT_URL=API_URL)

# Function to merge I-XXX tokens into full entities
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            merged_tokens.append(token)
    return merged_tokens

# Final NER function
def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

# Gradio interface for NER
gr.close_all()
demo = gr.Interface(
    fn=ner,
    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
    outputs=[gr.HighlightedText(label="Text with entities")],
    title="NER with dslim/bert-base-NER",
    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
    allow_flagging="never",
    examples=[
        "My name is K. Srisaran Karthik and I study Artificial Intelligence and Data Science at Saveetha Engineering College",
        "Saveetha Engineering College is located in Thandalam, Chennai near Chembarambakkam Lake"
    ]
)
demo.launch()

# Optional: Close all previous Gradio apps
gr.close_all()
```
### OUTPUT:

![EXP(GenAI) 5 SS](https://github.com/user-attachments/assets/c86e8927-2c8c-4a4a-9bb3-7974eeb9a123)

### RESULT:
Thus, the developed an NER prototype application with user interaction and evaluation features, using a fine-tuned BART model deployed through the Gradio framework.
