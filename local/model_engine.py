import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class ModelEngine:

    def __init__(self, log_callback=None):
        self.tokenizer = None
        self.model = None
        self.log_callback = log_callback

    def log(self, message):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def load_model(self, base_model_path, lora_path):
        self.log("开始加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path
        )
        self.log("Tokenizer 加载完成")
        self.log("开始加载基础 AI 模型...")
        self.log(f"模型路径：{base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        self.log("基础 AI 模型加载完成")
        self.log("开始加载 LoRA 增量模型...")
        self.log(f"LoRA 路径：{lora_path}")
        self.model = PeftModel.from_pretrained(
            base_model,
            lora_path
        )
        self.model.eval()
        if torch.cuda.is_available():
            self.log(
                f"GPU：{torch.cuda.get_device_name(0)}"
            )
        else:
            self.log("当前使用 CPU")
        self.log("========== 模型加载完成 ==========")

    def chat(
        self,
        prompt,
        max_new_tokens=200,
        temperature=0.7
    ):
        if self.model is None:
            raise RuntimeError("模型尚未加载")
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )