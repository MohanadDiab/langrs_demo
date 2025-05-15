import openai
import gradio as gr
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # store key in env var

def chat(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

with gr.Blocks() as demo:
    gr.Markdown("## 💬 OpenAI ChatBot")
    inp = gr.Textbox(label="Your message")
    out = gr.Textbox(label="Response")
    inp.submit(chat, inputs=inp, outputs=out)

demo.launch()
