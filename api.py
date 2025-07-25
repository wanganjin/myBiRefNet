
import os
import uuid
import requests
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import torch
from torchvision import transforms
from models.birefnet import BiRefNet
from io import BytesIO

# --- 全局设置 ---
# 如果静态文件目录不存在，则创建它
STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# 加载模型
print("正在加载 BiRefNet 模型...")
birefnet = BiRefNet.from_pretrained('ZhengPeng7/BiRefNet')
torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to('cpu')
birefnet.eval()
print("模型加载完成。")

# --- FastAPI应用 ---
app = FastAPI()
app.mount(f"/{STATIC_DIR}", StaticFiles(directory=STATIC_DIR), name="static")

class ImageRequest(BaseModel):
    image_url: str

# --- 图像处理函数 (源自 x.py) ---
def extract_object(birefnet, image: Image.Image):
    """从给定的PIL图像中提取主要对象"""
    # 数据设置
    image_size = (960, 960)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 模型需要RGB图像
    input_images = transform_image(image.convert("RGB")).unsqueeze(0).to('cpu')

    # 预测
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    
    # 将蒙版调整回原始尺寸并应用
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    return image, mask

def crop_to_square(image_with_alpha: Image.Image):
    """将带透明背景的图像剪裁成包含所有非透明像素的最小正方形"""
    alpha = image_with_alpha.split()[-1]
    
    bbox = alpha.getbbox()
    
    if bbox is None:
        return image_with_alpha
    
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    
    square_size = max(width, height)
    
    center_x = left + width // 2
    center_y = top + height // 2
    
    square_left = center_x - square_size // 2
    square_top = center_y - square_size // 2
    square_right = square_left + square_size
    square_bottom = square_top + square_size
    
    img_width, img_height = image_with_alpha.size
    square_left = max(0, square_left)
    square_top = max(0, square_top)
    square_right = min(img_width, square_right)
    square_bottom = min(img_height, square_bottom)
    
    actual_width = square_right - square_left
    actual_height = square_bottom - square_top
    
    if actual_width != actual_height:
        actual_size = min(actual_width, actual_height)
        
        square_left = center_x - actual_size // 2
        square_top = center_y - actual_size // 2
        square_right = square_left + actual_size
        square_bottom = square_top + actual_size
        
        if square_left < 0:
            square_left = 0
            square_right = actual_size
        elif square_right > img_width:
            square_right = img_width
            square_left = img_width - actual_size
            
        if square_top < 0:
            square_top = 0
            square_bottom = actual_size
        elif square_bottom > img_height:
            square_bottom = img_height
            square_top = img_height - actual_size
    
    cropped_image = image_with_alpha.crop((square_left, square_top, square_right, square_bottom))
    
    return cropped_image

# --- API 端点 ---
@app.post("/process-image/")
async def process_image_from_url(request_data: ImageRequest, http_request: Request):
    try:
        # 下载图片
        response = requests.get(request_data.image_url, stream=True)
        response.raise_for_status()
        
        # 使用BytesIO从内存中读取图片数据
        image_data = BytesIO(response.content)
        original_image = Image.open(image_data)

    except requests.exceptions.RequestException as e:
        return {"error": f"下载图片失败: {e}"}
    except IOError:
        return {"error": "无法打开图片。请确认URL指向的是有效的图片文件。"}

    # 确保图片是RGBA模式，以便应用透明蒙版
    if original_image.mode != 'RGBA':
        original_image = original_image.convert('RGBA')

    # 处理图片
    processed_image, _ = extract_object(birefnet, original_image)
    cropped_image = crop_to_square(processed_image)

    # 保存处理后的图片
    filename = f"{uuid.uuid4()}.png"
    save_path = os.path.join(STATIC_DIR, filename)
    cropped_image.save(save_path, 'PNG')

    # 生成结果URL
    base_url = str(http_request.base_url)
    result_url = f"{base_url}{STATIC_DIR}/{filename}"

    return {"result_url": result_url}

if __name__ == "__main__":
    import uvicorn
    # 允许从任何IP访问，端口为8000
    uvicorn.run(app, host="0.0.0.0", port=8000) 