import os
import argparse
from PIL import Image
import torch
import torch.onnx

import cn_clip.clip as clip

import sys
sys.path.append("/data1/zhn/macdata/code/github/python/Chinese-CLIP/cn_clip")

from clip.utils import _MODELS, _MODEL_INFO, _download, available_models, create_model, image_transform

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-arch", 
        default="ViT-B-16",
        choices=["ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
        help="Specify the architecture (model scale) of Chinese-CLIP model to be converted."
    )
    
    parser.add_argument(
        "--pytorch-ckpt-path", 
        default="/data1/zhn/macdata/code/github/python/modelData/experiments/muge_finetune_vit-b-16_roberta-base_bs128_8gpu/checkpoints/epoch3.pt", 
        type=str, 
        help="Path of the input PyTorch Chinese-CLIP checkpoint. Default to None which will automatically download the pretrained checkpoint."
    )
    
    # parser.add_argument(
    #     "--context-length", type=int, default=52, help="The padded length of input text (include [CLS] & [SEP] tokens). Default to 52."
    # )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘", "人", "超人", "小超人", "一张皮卡丘图片"]).to(device)
    
    # text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "人", "超人", "小超人"]).to(device)
    
    if os.path.isfile(args.pytorch_ckpt_path):
        input_ckpt_path = args.pytorch_ckpt_path
    elif args.model_arch in _MODELS:
        input_ckpt_path = _download(_MODELS[args.model_arch], args.download_root or os.path.expanduser("./cache/clip"))
    else:
        raise RuntimeError(f"Model {args.model_arch} not found; available models = {available_models()}")
    # print("")
    with open(input_ckpt_path, 'rb') as opened_file:
        checkpoint = torch.load(opened_file, map_location="cpu")
    
    model = create_model(_MODEL_INFO[args.model_arch]['struct'], checkpoint).float().eval()
    model = model.to(device)
    
    resolution = _MODEL_INFO[args.model_arch]['input_resolution']
    preprocess = image_transform(resolution)
    # image = preprocess(Image.new('RGB', (resolution, resolution))).unsqueeze(0)
    
    image = preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0).to(device)
    
    
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
        image_features /= image_features.norm(dim=-1, keepdim=True) 
        text_features /= text_features.norm(dim=-1, keepdim=True)    

        logits_per_image, logits_per_text = model.get_similarity(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs) 
    # text = clip.tokenize([""], context_length=args.context_length)
