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
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
import requests
import json
import gradio as gr

# Load environment variables
_ = load_dotenv(find_dotenv())  # Read local .env file
hf_api_key = os.environ['HF_API_KEY']
API_URL = os.environ['HF_API_SUMMARY_BASE']  # Ensure the API URL is correct

# Helper function to send API requests
def get_completion(inputs, parameters=None, ENDPOINT_URL=API_URL): 
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(data))
    return json.loads(response.content.decode("utf-8"))

# Function to merge subword tokens (e.g., "San" and "Francisco" into "San Francisco")
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # Merge the token if it's part of the same entity
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Add new token to the list
            merged_tokens.append(token)
    return merged_tokens

# NER function to process input and call the API
def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

# Initialize Gradio interface
gr.close_all()
demo = gr.Interface(
    fn=ner,
    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
    outputs=[gr.HighlightedText(label="Text with entities")],
    title="NER with dslim/bert-base-NER",
    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
    allow_flagging="never",
    examples=[
        "My name is Andrew, I'm building DeeplearningAI and I live in California",
        "My name is Poli, I live in Vienna and work at HuggingFace"
    ]
)

# Launch the Gradio interface
demo.launch()

```
### OUTPUT:
![alt text](image-1.png)
### RESULT:
Thus, the developed an NER prototype application with user interaction and evaluation features, using a fine-tuned BART model deployed through the Gradio framework.
