from models.blip import create_vit
import torch
import requests

visual_encoder, vision_width = create_vit("base",224,False, 0, 0)
checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
state_dict = checkpoint["model"]     
msg = visual_encoder.load_state_dict(state_dict,strict=False)
dummy_image = torch.randn(2, 3, 224, 224)
image_embeds = visual_encoder(dummy_image) 
print(f"vision width:{vision_width}")
print(f"image embeddings shape :{image_embeds.shape}")