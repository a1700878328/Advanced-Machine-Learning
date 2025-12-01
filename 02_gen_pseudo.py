import os
import torch
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# --- Configuration ---
TEACHER_PATH = "teacher_checkpoint/final"
DATA_DIR = "custom_dataset"
UNLABELED_CSV = os.path.join(DATA_DIR, "train_non_labels.csv")
IMG_DIR = os.path.join(DATA_DIR, "train_non_labels")
OUTPUT_CSV = os.path.join(DATA_DIR, "train_pseudo.csv")


def generate_pseudo():
    print("Loading Teacher Model to generate pseudo labels...")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        TEACHER_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = Qwen2VLProcessor.from_pretrained(TEACHER_PATH)

    df = pd.read_csv(UNLABELED_CSV)
    # Column name compatibility
    if 'file' in df.columns: df.rename(columns={'file': 'image'}, inplace=True)

    results = []

    model.eval()
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_path = os.path.join(IMG_DIR, row['image'])

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": row['question']}
                ]
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Parse "Answer: ... Explanation: ..."
            answer = "no"
            explanation = "No explanation."
            try:
                if "Answer:" in output_text:
                    parts = output_text.split("Answer:", 1)[1]
                    if "Explanation:" in parts:
                        ans, exp = parts.split("Explanation:", 1)
                        answer = ans.strip()
                        explanation = exp.strip()
                    else:
                        answer = parts.strip()
            except:
                pass

            # Save all original columns + pseudo labels
            results.append({
                "id": row['id'],
                "file": row['image'],
                "question": row['question'],
                "answer": answer,
                "explanation": explanation,
                "is_pseudo": True  # Mark as pseudo-label
            })

    # Save pseudo-label file
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"Pseudo labels generated: {OUTPUT_CSV}")


if __name__ == "__main__":
    generate_pseudo()
