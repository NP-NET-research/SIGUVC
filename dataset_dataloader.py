from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead
from peft import LoraConfig, TaskType, get_peft_model
import torch
import json

class MyDataset(Dataset):
    def __init__(self, prompt_path, data_path, tokenizer_path, device):
        self.data_path = data_path
        self.prompt_path = prompt_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,trust_remote_code=True)
        self.device = device
        with open(data_path, 'r', encoding='utf-8') as fp:
            self.dataset = json.load(fp)
        with open(prompt_path, 'r', encoding='utf-8') as fp:
            self.paraphrase_prompt = fp.read()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        chat_list = [
            {"role": "system", "content": self.paraphrase_prompt},
            {"role": "user", "content": self.dataset[index]["original_sentence"]}
        ]
        tokenized = torch.tensor(self.tokenizer.apply_chat_template(chat_list, add_generation_prompt=True, tokenize=True, max_length=512), device=self.device)
        return {
            "original_sentence": self.dataset[index]["original_sentence"],
            "input_ids": tokenized
        }


def Mycollater(samples):

    dct = {
        "batch_original_sentence": [sample['original_sentence'] for sample in samples],
        "batch_input_ids": [sample['input_ids'] for sample in samples],
    }
    return dct

if __name__ == "__main__":
    my_dataset = MyDataset(
        prompt_path="./paraphrased_system_prompt.txt",
        data_path="../data/Idiom_1000_ori.json",
        tokenizer_path="../glm-4-9b-chat-1m",
        device='cuda:0')
    
    dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True, collate_fn=Mycollater)
    tokenizer = AutoTokenizer.from_pretrained("../glm-4-9b-chat-1m",trust_remote_code=True)
    print(tokenizer.decode(tokenizer.eos_token_id))
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="../glm-4-9b-chat-1m",
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,   
            bnb_4bit_quant_type="fp4"            
        ),
        device_map="auto",
        max_memory={0: "20GiB"}
    )

    if hasattr(model.transformer, 'output_layer'):
        model.lm_head = model.transformer.output_layer
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        peft_model,
        trust_remote_code=True,
        load_in_8bit=True, 
    )
    model.is_peft_model = True

    for batch in dataloader:
        output = model.generate(torch.stack(batch["batch_input_ids"]).to("cuda:0"))
        print(batch["batch_original_sentence"])
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        print("=" * 16)
        
