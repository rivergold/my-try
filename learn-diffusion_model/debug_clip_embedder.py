import sys

sys.path.append('../../')

# from ldm.modules.encoders.modules import FrozenCLIPEmbedder
import torch
from transformers import CLIPTokenizer, CLIPTextModel

if __name__ == '__main__':
    model_id = 'openai/clip-vit-large-patch14'
    max_length = 77

    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        cache_dir='/home/hejing/data/opensource/huggingface/hub',
        local_files_only=True)
    clip_text_model = CLIPTextModel.from_pretrained(
        model_id,
        cache_dir='/home/hejing/data/opensource/huggingface/hub',
        local_files_only=True)
    clip_text_model.to('cuda')

    text = 'a car.'
    batch_encoding = tokenizer([text],
                               truncation=True,
                               max_length=max_length,
                               return_length=True,
                               return_overflowing_tokens=False,
                               padding="max_length",
                               return_tensors="pt")
    tokens = batch_encoding['input_ids'].to('cuda')
    outputs = clip_text_model(input_ids=tokens)

    z = outputs.last_hidden_state
    print(z.shape)