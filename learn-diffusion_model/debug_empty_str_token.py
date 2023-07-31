import sys

sys.path.append('../../')

# from ldm.modules.encoders.modules import FrozenCLIPEmbedder
import torch
from transformers import CLIPTokenizer, CLIPTextModel

if __name__ == '__main__':
    model_path = '/home/hejing/data/opensource/huggingface/model/stable-diffusion-v1-5'
    max_length = 77

    tokenizer = CLIPTokenizer.from_pretrained(model_path,
                                              subfolder='tokenizer')
    clip_text_model = CLIPTextModel.from_pretrained(model_path,
                                                    subfolder='text_encoder')
    clip_text_model.to('cuda')

    text = ''
    assert max_length == tokenizer.model_max_length
    print(f"model_max_length: {tokenizer.model_max_length}")
    batch_encoding = tokenizer([text],
                               truncation=True,
                               max_length=max_length,
                               return_length=True,
                               return_overflowing_tokens=False,
                               padding="max_length",
                               return_tensors="pt")
    print(batch_encoding)
    tokens = batch_encoding['input_ids'].to('cuda')
    print(tokens)

    outputs = clip_text_model(input_ids=tokens)

    z = outputs.last_hidden_state
    print(z.shape)