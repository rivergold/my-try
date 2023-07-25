from transformers import CLIPTokenizer

if __name__ == '__main__':
    tokenizer = CLIPTokenizer.from_pretrained(
        '/home/hejing/data/opensource/huggingface/model/model--CompVis--stable-diffusion-v1-4',
        subfolder='tokenizer')

    res = tokenizer.add_tokens(['new_tok1'])
    print(res)

    res = tokenizer.encode('new_tok1', add_special_tokens=False)
    print(res)