#プロンプトを打ち込むと解説してくれるAI(低精度)
!pip install transformers sentencepiece
!pip install -q -U git+https://github.com/kuramitsulab/papertown.git

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from papertown import load_tokenizer

USE_CUDA = torch.cuda.is_available()

def cuda(x):
    return x.cuda() if USE_CUDA else x

# 事前学習済みモデルとトークナイザーをロード
##　最新のパスはSlackで送信します
tokenizer = load_tokenizer("myst72/mejiro2023-rinna-e4")
model = cuda(AutoModelForCausalLM.from_pretrained("myst72/mejiro2023-rinna-e4"))

def complete(prompt):
    '''
    prompt:入力->generated_text:出力
    '''
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt")
    output = model.generate(
        input_ids=cuda(inputs.input_ids),
        attention_mask=cuda(inputs.attention_mask),
        max_length=512,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# 事前学習済みモデルとトークナイザーをロード
##　最新のパスはSlackで送信します
tokenizer = load_tokenizer("myst72/mejiro2023-rinna-e4")
model = cuda(AutoModelForCausalLM.from_pretrained("myst72/mejiro2023-rinna-e4"))

prompt = "チャイコフスキー　シンフォニー　６番　悲愴"
generated_text = complete(prompt)
print(generated_text)

prompt = "メロスは激怒した。"
generated_text = complete(prompt)
print(generated_text)

prompt = "　吾輩わがはいは猫である。名前はまだ無い。どこで生れたか"
generated_text = complete(prompt)
print(generated_text)

from papertown.new_model import print_model
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-small", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-small")

print_model(model)

prompt = "本日"
generated_text = complete(prompt)
print(generated_text)
