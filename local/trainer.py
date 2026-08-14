import json
import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

class TrainingLogCallback(TrainerCallback):

    def __init__(self, log_callback):
        self.log_callback = log_callback

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        message = " | ".join(
            f"{key}: {value}"
            for key, value in logs.items()
        )
        self.log_callback(message)


class TrainerEngine:

    def __init__(
        self,
        model_path,
        dataset_path,
        output_path,
        device="GPU",
        epochs=10,
        log_callback=None
    ):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.device = device
        self.epochs = epochs
        self.log_callback = log_callback

    def log(self, message):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def train(self):
        self.log("========== 开始训练 ==========")
        # ===============================
        # Device
        # ===============================
        if self.device == "GPU":
            self.log("检查 GPU...")
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "用户选择了 GPU，但当前环境没有可用的 CUDA GPU。"
                )
            device = "cuda"
            self.log(
                f"使用 GPU：{torch.cuda.get_device_name(0)}"
            )
        else:
            device = "cpu"
            self.log("使用 CPU 训练")
        # ===============================
        # Tokenizer
        # ===============================
        self.log("加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.log("Tokenizer 加载完成")
        # ===============================
        # Model
        # ===============================
        self.log("加载基准模型...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
        )
        model.to(device)
        self.log("基准模型加载完成")
        # ===============================
        # LoRA
        # ===============================
        self.log("配置 LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj"
            ],
        )
        model = get_peft_model(
            model,
            lora_config
        )
        self.log("LoRA 配置完成")
        # ===============================
        # Dataset
        # ===============================
        self.log("加载训练数据...")
        dataset = load_dataset(
            "json",
            data_files=self.dataset_path,
            split="train",
        )
        self.log(
            f"训练数据加载完成，共 {len(dataset)} 条"
        )
        # ===============================
        # Tokenize
        # ===============================
        self.log("开始 Tokenize...")
        max_prompt_len = 128
        max_target_len = 64
        max_length = 256

        def tokenize(example):
            prompt = (
                f"{example['instruction']}\n\n"
                f"用户输入：{example['input']}\n\n"
                f"输出："
            )
            target = json.dumps(
                example["output"],
                ensure_ascii=False
            )
            prompt_ids = tokenizer(
                prompt,
                truncation=True,
                max_length=max_prompt_len
            )["input_ids"]
            target_ids = tokenizer(
                target,
                truncation=True,
                max_length=max_target_len,
                add_special_tokens=True
            )["input_ids"]
            input_ids = (
                prompt_ids + target_ids
            )[:max_length]
            labels = (
                [-100] * len(prompt_ids)
                + target_ids
            )[:max_length]
            return {
                "input_ids": input_ids,
                "labels": labels,
            }
        dataset = dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
        )
        self.log("Tokenize 完成")
        # ===============================
        # Training Arguments
        # ===============================
        training_args = TrainingArguments(
            output_dir=self.output_path,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=self.epochs,
            learning_rate=1e-4,
            fp16=False,
            logging_steps=5,
            save_steps=100,
            save_total_limit=2,
            report_to="none",
        )
        # ===============================
        # Trainer
        # ===============================
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[
                TrainingLogCallback(
                    self.log
                )
            ],
        )
        # ===============================
        # Train
        # ===============================
        self.log("开始训练...")
        trainer.train()
        # ===============================
        # Save
        # ===============================
        self.log("训练完成，正在保存模型...")
        model.save_pretrained(
            self.output_path
        )
        tokenizer.save_pretrained(
            self.output_path
        )
        self.log(
            f"模型已保存到：{self.output_path}"
        )
        self.log("========== 训练完成 ==========")