from dotenv import load_dotenv
from pydub import AudioSegment
from transformers import pipeline

# Load the HF token from .env
load_dotenv()

import gradio as gr
from huggingface_hub import InferenceClient

LOCAL_AUDIO_FILE = "./input.wav"


def load_whisper_model():
    return pipeline(
        "automatic-speech-recognition", model="openai/whisper-tiny", 
    )

pipe = load_whisper_model()

def load_audio_file(file):
    try: 
        return AudioSegment.from_file(file)
    except:
        raise gr.Error("Make sure a valid file is already uploaded.")
    
def build_message_prompt(text):
    system_message = f"""Generate a haiku based on the given text.
        A haiku is a short, Japanese poem typically with three lines. 
        It follows a structure of 5 syllables in the first line, 7 in the second, and 5 in the third, 
        totaling 17 syllables. 
        Please respond with only the haiku and no additional text. 
        """

    messages = [{"role": "system", "content": system_message}]
    messages.append({"role": "user", "content": text})

    return messages

def respond(file, hf_token: gr.OAuthToken):
    global pipe

    input_sound = load_audio_file(file)

    # Save audio file in wav format (which is compatible with whisper)
    input_sound.export(LOCAL_AUDIO_FILE, format="wav")

    if pipe is None:
        pipe = load_whisper_model()

    # Convert the audio to text with the whisper tiny model
    response = pipe(LOCAL_AUDIO_FILE)
    text_result = response["text"]

    messages = build_message_prompt(text_result)

    # Convert text to haiku
    if hf_token is None or not getattr(hf_token, "token", None):
        yield "⚠️ Please log in with your Hugging Face account first."
        return
    
    client = InferenceClient(token=hf_token.token, model="openai/gpt-oss-20b")

    response = ""

    for chunk in client.chat_completion(
        messages,
        stream=True
    ):
        choices = chunk.choices
        token = ""
        if len(choices) and choices[0].delta.content:
            token = choices[0].delta.content
        response += token

        yield response
    
# Fancy styling
CSS = """
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

with gr.Blocks(css=CSS) as demo:
    with gr.Row():
        gr.Markdown("<h1 style='text-align: center; color: black'>⛩️ HaikuAI ⛩️</h1>")
        gr.LoginButton()
    with gr.Row():
        upload = gr.UploadButton(
            label="📂 Upload Audio File (< 30 seconds)", file_types=[".mp3", ".wav", ".flac", ".mp4", ".m4a"], 
        )
    with gr.Row():
        submit = gr.Button("Submit")
    with gr.Row():
        output = gr.Textbox(label="Output", lines=5, interactive=False)

    upload.upload(lambda: gr.Info("File uploaded."))
    submit.click(fn=respond, inputs=[upload], outputs=output)

if __name__ == "__main__":
    demo.launch()
