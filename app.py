import gradio as gr
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path

BASE_MODEL = 'EleutherAI/gpt-neo-125M'
LORA_PATH = Path(__file__).resolve().parent / "results"

print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
model = model.merge_and_unload()
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
print('Ready!')

def generate_story(prompt, max_tokens, temperature, top_p):
    grounded_prompt = (
        "You explain digital circuits in very simple terms. "
        "Stay factual. If unsure, say you are unsure.\n"
        f"Q: {prompt.strip()}\nA:"
    )
    out = generator(
        grounded_prompt,
        max_new_tokens=int(max_tokens),
        do_sample=False,
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return out[0]['generated_text'].split('A:')[-1].strip()

with gr.Blocks(title='Circuit Story LLM', theme=gr.themes.Soft(primary_hue='indigo')) as demo:
    gr.Markdown('# ?? Circuit Story LLM\n### Explain digital circuits as creative stories\nBuilt by Sudhindra')
    prompt_box = gr.Textbox(label='Prompt', placeholder='Explain a flip-flop in a story')
    with gr.Row():
        max_tokens = gr.Slider(50, 300, value=120, step=10, label='Max tokens')
        temperature = gr.Slider(0.1, 1.5, value=0.3, step=0.05, label='Temperature')
        top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label='Top-p')
    submit_btn = gr.Button('? Generate Story', variant='primary')
    output_box = gr.Textbox(label='?? Story', lines=6, interactive=False)
    submit_btn.click(fn=generate_story, inputs=[prompt_box, max_tokens, temperature, top_p], outputs=output_box)

demo.launch(server_name='0.0.0.0', server_port=7860, inbrowser=True)
