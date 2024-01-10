# **StableDiffusion-XL-panorama**
**Stable Diffusion XL:**

   Using the LoRA weights from ‘huggingface.com/jbilcke-hf/sdxl-panorama’, fine-tuned the stable diffusion xl base 1.0 model to generate coherent 360-degree panoramic images. The generated images satisfied the following criteria:
   * The scene matched the user’s prompt in terms of content and style
   * The ends of the panorama were seamlessly joined to form a continuous loop

     Some images that closely follow the conditions are demonstrated below with their corresponding prompts.

*Prompt 1:* "a high resolution, coherent, 360-degree panoramic view of a gaming room with monitors displaying modern warfare games and a gamer sitting in front of one of the seats.The room is lit up by warm lights, coherent panorama with seam tiling"

![image4](https://github.com/beatsea20/StableDiffusion-XL-panorama/assets/108799982/ae188af9-1a6b-4f90-9e0c-219ba80c7bff)

*Prompt 2:* "coherent 360-degree panorama image, The generated panorama’s ends should meet i.e. seam needs to be tiled. A small 4-walled salon with multiple mirrors, salon chairs, white walls, warm lighting, daylight."

![image13_n](https://github.com/beatsea20/StableDiffusion-XL-panorama/assets/108799982/64733a47-cdd8-4359-b7b5-762299ca4631)

*Prompt 3:* "coherent 360-degree panorama image, The generated panorama’s ends should meet i.e. seam needs to be tiled. An enclosed kids playroom with 4 walls, fun toys, green walls, cool lighting"

![image12_myn](https://github.com/beatsea20/StableDiffusion-XL-panorama/assets/108799982/583668b4-a350-4503-8712-b281f822ecbc)

*Prompt 4:* "a high resolution, coherent 360 degree panorama image with seam tiling i.e. panorama's ends must meet, an empty court room resembling those in movies, typical lighting, old architechture"

![image10](https://github.com/beatsea20/StableDiffusion-XL-panorama/assets/108799982/29b856ae-0c18-41a8-8ba8-8cb2ce4b4e78)

*Prompt 5:* "ancient room for keeping weapons, 360-degree panoramic image, detailed, 4k, old architechture, well preserved weapons, spectacular details"

![image14](https://github.com/beatsea20/StableDiffusion-XL-panorama/assets/108799982/3b9d708a-34d2-44de-a51d-f663fe3701a4)


 

**Observations:** Analyzing the generated images reveals that while they exhibit a high degree of internal coherence in most cases, they do not consistently adhere to the input prompts on every occasion. Specifically, we observed instances where the images, while maintaining a strong thematic connection to the prompt, displayed minor deviations from the exact details specified. This phenomenon is particularly noticeable when dealing with prompts involving concepts like "extended rooms," "multiple objects," or "closed spaces." In these scenarios, the model appears to engage in "hallucination" introducing elements not explicitly mentioned in the prompt but potentially related to its overall meaning or the model's own internal understanding of the scene.



# **ControlNet with Stable Diffusion XL** 

  ControlNet is a neural network structure to control diffusion models by adding extra conditions.
  The same prompts are given to the model along with the depth map of the image generated in the above section. The depth map is extracted from the image using the Dense Prediction Transformer (DPT) model trained on 1.4 million images for monocular depth estimation by Intel. DPT uses the Vision Transformer (ViT) as backbone and adds a neck + head on top for monocular depth estimation.

Following images are generated using the same prompt and depth map of the previously generated images:
  For *prompt1*-
  
![image4_depth1](https://github.com/beatsea20/StableDiffusion-XL-panorama/assets/108799982/1bdf67a2-0317-45e7-b12f-9d37ea9abbc0)

For *prompt2*-
![image13_n_depth](https://github.com/beatsea20/StableDiffusion-XL-panorama/assets/108799982/cc9f7cdb-3a0c-473c-b093-04dbd1883597)

For *prompt3*-
![image12_myn_depth](https://github.com/beatsea20/StableDiffusion-XL-panorama/assets/108799982/ef78ba1c-5698-4071-8382-3f0b74a9d824)

For *prompt4*-
![image10_depth](https://github.com/beatsea20/StableDiffusion-XL-panorama/assets/108799982/f2099ba4-44b3-4067-a841-27ba32a3677b)

For *prompt5*-
![image14_depth](https://github.com/beatsea20/StableDiffusion-XL-panorama/assets/108799982/d684bc99-56d1-4438-8357-75642bcc61e5)




PS - Following are some images that did not fulfill the conditions, just putting them up.

![image7](https://github.com/beatsea20/StableDiffusion-XL-panorama/assets/108799982/7b3d372e-efab-4590-aa1f-c7eb5385fc24)

![image16](https://github.com/beatsea20/StableDiffusion-XL-panorama/assets/108799982/b5b84cc0-5b94-42cc-8ad0-94aa03fbed3a)

