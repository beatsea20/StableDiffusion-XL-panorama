!pip install diffusers transformers accelerate safetensors huggingface_hub
!git clone https://github.com/replicate/cog-sdxl cog_sdxl


#Generate coherent panoramic images from text prompts
import torch
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline
from cog_sdxl.dataset_and_utils import TokenEmbeddingsHandler
from diffusers.models import AutoencoderKL
import PIL.Image

#load pretrained Stable Diffusion XL from Hugging face
pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
).to('cuda') # uses the gpu for accelerated operations

#load model weights
pipe.load_lora_weights("jbilcke-hf/sdxl-panorama", weight_name="lora.safetensors")

#preparing text_encoders and tokenizers for embedding handling
text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

#downloading pre-trained embeddings
embedding_path = hf_hub_download(repo_id="jbilcke-hf/sdxl-panorama", filename="embeddings.pti", repo_type="model")
embhandler = TokenEmbeddingsHandler(text_encoders, tokenizers)
embhandler.load_embeddings(embedding_path)

#prompts to generate images from the pipeline
prompt1= "coherent 360-degree panorama image, The generated panorama’s ends should meet i.e. seam needs to be tiled. An enclosed kids playroom with 4 walls, fun toys, green walls and cool lighting."
image1 = pipe(
    prompt1,
    cross_attention_kwargs={"scale": 0.8},
    num_inference_steps = 50,
    height = 512,
    width = 1024,
).images[0]
image1.save("/content/drive/MyDrive/Avataar.ai/image1.png")

prompt2= "coherent 360-degree panorama image, The generated panorama’s ends should meet i.e. seam needs to be tiled. A small 4-walled salon with multiple mirrors, salon chairs, white walls, warm lighting, daylight."
image2 = pipe(
    prompt2,
    cross_attention_kwargs={"scale": 0.8},
    num_inference_steps = 50,
    height = 512,
    width = 1024,
).images[0]
image2.save("/content/drive/MyDrive/Avataar.ai/image2.png")

prompt3 = "a high resolution, hd, coherent 360 degree panorama image with seam tiling i.e. panorama's ends must meet, an empty court room resembling those in movies, typical lighting, old architechture"
image3 = pipe(
    prompt3,
    cross_attention_kwargs={"scale": 0.9},
    num_inference_steps = 50,
    height = 512,
    width = 1024,
).images[0]
image3.save("/content/drive/MyDrive/Avataar.ai/image3.png")

prompt4= " a high resolution, coherent, 360-degree panoramic view of a gaming room with monitors displaying modern warfare games and a gamer sitting in front of one of the seats.The room is lit up by warm lights. ensure seam tiling"
image4 = pipe(
    prompt4,
    cross_attention_kwargs={"scale": 0.0},
    num_inference_steps = 50,
    height = 512,
    width = 1024,
).images[0]
image4.save("/content/drive/MyDrive/Avataar.ai/image4.png")

prompt5= "ancient room for keeping weapons, 360-degree panoramic image, detailed, 4k, old architechture, well preserved weapons, spectacular details"
image5 = pipe(
    prompt5,
    cross_attention_kwargs={"scale": 0.8},
    num_inference_steps = 50,
    height = 512,
    width = 1024,
).images[0]
image5.save("/content/drive/MyDrive/Avataar.ai/image5.png")
