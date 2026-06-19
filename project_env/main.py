import torch
import gc
from pdf2image import convert_from_path
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# 1. Memory-Optimized Config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model_id = "Qwen/Qwen2-VL-7B-Instruct"

# Use Flash Attention 2 if your GPU is RTX 30/40 series (Ampere+)
# It dramatically reduces VRAM usage during the "generation" phase.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="sdpa", 
)

# 2. Set strict pixel limits to protect VRAM
# 1280x28x28 is roughly 1M pixels; 784x28x28 (~600k pixels) is usually 
# plenty for clear handwriting while staying under 70% VRAM.
min_pixels = 256 * 28 * 28
max_pixels = 784 * 28 * 28 
processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)

pdf_path = r"D:\chandra_ocr\project_env\student_answers.pdf"
images = convert_from_path(pdf_path)

full_transcript = []

for i, page_image in enumerate(images):
    print(f"Processing Page {i+1}...")
    
    messages = [{"role": "user", "content": [
        {"type": "image", "image": page_image}, 
        {"type": "text", "text": "Transcribe the handwritten text exactly."}
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # 3. Controlled Generation
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=1000, # Cap this to prevent VRAM growth
            do_sample=False,
            use_cache=True # Essential for speed
        )
    
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    page_content = output_text[0].split("assistant\n")[-1]
    full_transcript.append(page_content)

    # --- Aggressive Cleanup ---
    del inputs, generated_ids, image_inputs
    torch.cuda.empty_cache()
    gc.collect()

print("Processing complete.")