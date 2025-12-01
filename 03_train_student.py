import os
import pandas as pd
import torch
import ast
import random
from torch.utils.data import Dataset, ConcatDataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, Trainer, TrainingArguments, \
    DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

# --- 配置 ---
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"  # 学生也用这个，从头训练(或基于Teacher继续练)
DATA_DIR = "custom_dataset"
REAL_CSV = os.path.join(DATA_DIR, "train_labels.csv")
PSEUDO_CSV = os.path.join(DATA_DIR, "train_pseudo.csv")
REAL_IMG_DIR = os.path.join(DATA_DIR, "train")
PSEUDO_IMG_DIR = os.path.join(DATA_DIR, "train_non_labels")
OUTPUT_DIR = "student_checkpoint"


# 复用 Dataset 类 (稍微修改以适应两种CSV格式)
class MixedVQADataset(Dataset):
    def __init__(self, df, img_dir, processor):
        self.df = df
        self.img_dir = img_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 处理解释
        explanation = str(row['explanation'])
        # 只有真实数据的解释是列表字符串，需要解析
        if explanation.startswith('['):
            try:
                explanation = random.choice(ast.literal_eval(explanation))
            except:
                pass

        answer_text = f"Answer: {row['answer']}\nExplanation: {explanation}"

        # 兼容列名
        fname = row['file'] if 'file' in row else row['image']
        image_path = os.path.join(self.img_dir, fname)

        messages = [
            {"role": "user",
             "content": [{"type": "image", "image": image_path}, {"type": "text", "text": row['question']}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding="max_length", truncation=True, max_length=512, return_tensors="pt",
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        inputs["labels"][inputs["labels"] == self.processor.tokenizer.pad_token_id] = -100
        return inputs


def train_student():
    print("🎓 正在启动 Student 混合训练...")

    # 1. 加载全新的基础模型 (Student)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = Qwen2VLProcessor.from_pretrained(MODEL_ID, min_pixels=256 * 256, max_pixels=512 * 512)

    # 添加 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64,  # 学生模型可以用更大的Rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)

    # 2. 准备混合数据
    df_real = pd.read_csv(REAL_CSV)
    df_pseudo = pd.read_csv(PSEUDO_CSV)

    ds_real = MixedVQADataset(df_real, REAL_IMG_DIR, processor)
    ds_pseudo = MixedVQADataset(df_pseudo, PSEUDO_IMG_DIR, processor)

    # 拼接数据集
    mixed_dataset = ConcatDataset([ds_real, ds_pseudo])
    print(f"真实数据: {len(ds_real)}, 伪数据: {len(ds_pseudo)}, 总计: {len(mixed_dataset)}")

    # 3. 训练参数 (学生跑更久一点)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=1e-4,  # 学习率稍小
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=mixed_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True),
    )

    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    print("🎉 Student 模型训练完成！")


if __name__ == "__main__":
    train_student()