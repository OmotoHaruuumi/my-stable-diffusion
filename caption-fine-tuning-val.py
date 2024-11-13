from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch

# デバイスの設定（GPUが利用可能な場合はGPUを使用）
device = "cuda" if torch.cuda.is_available() else "cpu"

# 事前学習済みモデルのロード
pretrained_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
pretrained_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
pretrained_model.to(device)

# ファインチューニング済みモデルのロード（ファインチューニング済みモデルが存在する場合）
finetuned_processor = BlipProcessor.from_pretrained("path/to/your/finetuned-model")  # 適切なパスに変更
finetuned_model = BlipForConditionalGeneration.from_pretrained("path/to/your/finetuned-model")
finetuned_model.to(device)

# テスト画像のロード
image_url = 'https://i.gyazo.com/0301ae28236d7a50abedb0f2670bf170.jpg'  # ここにテスト画像のURLを指定
raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# 事前学習済みモデルでキャプション生成
def generate_caption(processor, model, image):
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# 事前学習済みモデルの結果
pretrained_caption = generate_caption(pretrained_processor, pretrained_model, raw_image)
print("Pretrained Model Caption:", pretrained_caption)

# ファインチューニング済みモデルの結果
finetuned_caption = generate_caption(finetuned_processor, finetuned_model, raw_image)
print("Fine-tuned Model Caption:", finetuned_caption)

# 出力を比較
print("\n--- Comparison ---")
print(f"Pretrained Model Caption:\n{pretrained_caption}")
print(f"Fine-tuned Model Caption:\n{finetuned_caption}")