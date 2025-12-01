import os
import json
import pandas as pd
import torch
import ast
import random
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, Trainer, TrainingArguments, \
    DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

# --- 配置 ---
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"  # 2B模型，完全合规，无需量化
DATA_DIR = "custom_dataset"
TRAIN_CSV = os.path.join(DATA_DIR, "train_labels.csv")
IMG_DIR = os.path.join(DATA_DIR, "train")
OUTPUT_DIR = "teacher_checkpoint"


# --- 数据集类 ---
class VQADataset(Dataset):
    def __init__(self, csv_file, img_dir, processor):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. 处理解释 (随机选一个)
        explanation = ""
        try:
            if isinstance(row['explanation'], str) and row['explanation'].startswith('['):
                expl_list = ast.literal_eval(row['explanation'])
                explanation = random.choice(expl_list)
            else:
                explanation = str(row['explanation'])
        except:
            explanation = "No explanation."

        # 2. 构建对话
        # 我们训练模型按照固定格式输出
        answer_text = f"Answer: {row['answer']}\nExplanation: {explanation}"

        image_path = os.path.join(self.img_dir, row['file'])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": row['question']}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text}]
            }
        ]

        # 3. 预处理
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            truncation=True,
            max_length=512,  # 控制长度省显存
            return_tensors="pt",
        )

        # 移除batch维度
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # 设置 Labels (忽略 padding)
        inputs["labels"] = inputs["input_ids"].clone()
        inputs["labels"][inputs["labels"] == self.processor.tokenizer.pad_token_id] = -100

        return inputs


def train_teacher():
    print("🚀 正在启动 Teacher 训练...")

    # 1. 加载模型 (BF16, 16GB显存毫无压力)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = Qwen2VLProcessor.from_pretrained(MODEL_ID, min_pixels=256 * 256, max_pixels=512 * 512)

    # 2. 添加 LoRA (微调)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. 准备数据
    dataset = VQADataset(TRAIN_CSV, IMG_DIR, processor)

    # 4. 训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,  # 16GB 可以尝试 4 或 8
        gradient_accumulation_steps=4,  # 等效 batch size 16
        num_train_epochs=5,  # 教师模型不用跑太久
        learning_rate=2e-4,
        bf16=True,  # 开启 BF16 加速
        logging_steps=10,
        save_strategy="epoch",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True),
    )

    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    print("✅ Teacher 模型训练完成！")


if __name__ == "__main__":
    train_teacher()