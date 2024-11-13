from stable_diffusion_pytorch import pipeline
import torch
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts)
images[0].save('output.jpg')