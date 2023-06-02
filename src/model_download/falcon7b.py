from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_name = "tiiuae/falcon-7b"
model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

save_path = '/home/julian.laue/models/falcon/'

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)