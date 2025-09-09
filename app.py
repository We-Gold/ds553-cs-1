from dotenv import load_dotenv

# Load the HF token from .env
load_dotenv()

import os
from pathlib import Path

import gradio as gr
from huggingface_hub import InferenceClient

pipe = None
stop_inference = False

# Fancy styling
fancy_css = """
#main-container {
    background-color: #f0f0f0;
    font-family: 'Arial', sans-serif;
    width: 700px;
}
.gradio-container {
    width: 700px;
    margin: 0 auto;
    padding: 20px;
    background: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}
.gr-button {
    background: #8e44ad;
    color: white;
    border-radius: 50px;
    padding: 12px 24px;
    font-size: 1.1em;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
}
.gr-button:hover {
    background: #732d91;
    transform: translateY(-2px);
}
.gr-slider input {
    color: #4CAF50;
}
.gr-chat {
    font-size: 16px;
}
#title {
    text-align: center;
    font-size: 2em;
    margin-bottom: 20px;
    color: #333;
}
"""


def upload_file(filepath):
    name = Path(filepath).name
    return [
        gr.UploadButton(visible=False),
        gr.DownloadButton(label=f"Download {name}", value=filepath, visible=True),
    ]


def download_file():
    return [gr.UploadButton(visible=True), gr.DownloadButton(visible=False)]


def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
    use_local_model: bool,
):
    global pipe

    # Build messages from history
    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = ""

    if use_local_model:
        print("[MODE] local")
        import torch
        from transformers import pipeline

        if pipe is None:
            pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")

        # Build prompt as plain text
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        outputs = pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        response = outputs[0]["generated_text"][len(prompt) :]
        yield response.strip()

    else:
        print("[MODE] api")

        if hf_token is None or not getattr(hf_token, "token", None):
            yield "⚠️ Please log in with your Hugging Face account first."
            return

        client = InferenceClient(token=hf_token.token, model="openai/gpt-oss-20b")

        for chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            choices = chunk.choices
            token = ""
            if len(choices) and choices[0].delta.content:
                token = choices[0].delta.content
            response += token
            yield response


with gr.Blocks(css=fancy_css) as demo:
    with gr.Row(): 
        gr.Markdown("<h1 style='text-align: center; color: black'>㊗️ HaikuAI ㊗️</h1>") 
        gr.LoginButton()
    with gr.Row():
        upload = gr.UploadButton(label="📂 Upload Audio File", file_types=[".mp3", ".wav"])
    with gr.Row():
        submit = gr.Button("Submit") 
    with gr.Row():
        output = gr.Textbox(label="Output", lines=5, interactive=False)
    with gr.Row():
        checkbox = gr.Checkbox(label="Use Local Model", value=False)

    submit.click(fn=respond, inputs=[upload, checkbox], outputs=output)

if __name__ == "__main__":
    demo.launch()
