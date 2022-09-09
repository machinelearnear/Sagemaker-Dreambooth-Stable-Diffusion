# Running Dreambooth Stable Diffusion on Sagemaker
This repo will show you a step by step tutorial on how to run @XavierXiao's [Dreambooth-Stable-Diffusion](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion) implementation with SageMaker Training Jobs. If time permits, we will also create a pipeline where you just upload photos to s3 and run the whole process automatically.

Overview by @XavierXiao:
> This is an implementaion of Google's [Dreambooth](https://arxiv.org/abs/2208.12242) with [Stable Diffusion](https://github.com/CompVis/stable-diffusion). The original Dreambooth is based on [Imagen](https://imagen.research.google/) text-to-image model. However, neither the model nor the pre-trained weights of Imagen is available. To enable people to fine-tune a text-to-image model with a few examples, I implemented the idea of Dreambooth on Stable diffusion.

> This code repository is based on that of [Textual Inversion](https://github.com/rinongal/textual_inversion). Note that Textual Inversion only optimizes word ebedding, while dreambooth fine-tunes the whole diffusion model.

> The implementation makes minimum changes over the official codebase of Textual Inversion. In fact, due to lazyness, some components in Textual Inversion, such as the embedding manager, are not deleted, although they will never be used here.

## Getting started
- [Dreambooth Stable Diffusion](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion)
- [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)
- [StabilityAI/Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://github.com/rinongal/textual_inversion)

## Prepare environment

You can follow this step by step tutorial locally either through Amazon SageMaker StudioLab, Google Colab, or on your own hardware. Please click in any of the links below to get started without using Sagemaker (no AWS credentials)

[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/machinelearnear/Sagemaker-Dreambooth-Stable-Diffusion/blob/main/local-test.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machinelearnear/Sagemaker-Dreambooth-Stable-Diffusion/blob/main/local-test.ipynb)

In order to setup your SMSL environment, you will need to get a [SageMaker Studio Lab](https://studiolab.sagemaker.aws/) account. This is completely free and you don't need an AWS account. Because this is still in Preview and AWS is looking to reduce fraud (e.g., crypto mining), you will need to wait 1-3 days for your account to be approved. You can see [this video](https://www.youtube.com/watch?v=FUEIwAsrMP4&ab_channel=machinelearnear) for more information. Otherwise, you can also use [Google Colab](https://colab.research.google.com/) which provides free GPU compute (NVIDIA T4/K80).

## Prepare your own custom data

> To fine-tune a stable diffusion model, you need to obtain the pre-trained stable diffusion models following their [instructions](https://github.com/CompVis/stable-diffusion#stable-diffusion-v1). Weights can be downloaded on [HuggingFace](https://huggingface.co/CompVis). You can decide which version of checkpoint to use, but I use ```sd-v1-4-full-ema.ckpt```.

> We also need to create a set of images for regularization, as the fine-tuning algorithm of Dreambooth requires that. Details of the algorithm can be found in the paper. Note that in the original paper, the regularization images seem to be generated on-the-fly. However, here I generated a set of regularization images before the training. The text prompt for generating regularization images can be ```photo of a <class>```, where ```<class>``` is a word that describes the class of your object, such as ```dog```. The command is:

```python
class_word = "yerba mate"
ddim_eta = 0.0
n_samples = 8
n_iter = 1
scale = 10.0
ddim_steps = 50
ckpt = '/path/to/original/stable-diffusion/sd-v1-4-full-ema.ckpt'
prompt = f"a photo of a {class_word}"

# run image generation from text prompts
os.system(f"python scripts/stable_txt2img.py --ddim_eta {ddim_eta} " + 
          f"--n_samples {n_samples} --n_iter {n_iter} --scale {scale} " + 
          f"--ddim_steps {ddim_steps} --ckpt {ckpt} --prompt {prompt}")
```

## Re-train SD with your own images 

> Training can be done by running the following command

```bash
python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml 
                -t 
                --actual_resume /path/to/original/stable-diffusion/sd-v1-4-full-ema.ckpt  
                -n <job name> 
                --gpus 0, 
                --data_root /root/to/training/images 
                --reg_data_root /root/to/regularization/images 
                --class_word <xxx>
```

> Detailed configuration can be found in `configs/stable-diffusion/v1-finetune_unfrozen.yaml`. In particular, the default learning rate is `1.0e-6` as I found the `1.0e-5` in the Dreambooth paper leads to poor editability. The parameter `reg_weight` corresponds to the weight of regularization in the Dreambooth paper, and the default is set to `1.0`.

> Dreambooth requires a placeholder word [V], called identifier, as in the paper. This identifier needs to be a relatively rare tokens in the vocabulary. The original paper approaches this by using a rare word in `T5-XXL` tokenizer. For simplicity, here I just use a random word sks and hard coded it.. If you want to change that, simply make a change in this file.

> Training will be run for 800 steps, and two checkpoints will be saved at `./logs/<job_name>/checkpoints`, one at 500 steps and one at final step. Typically the one at 500 steps works well enough. I train the model use two A6000 GPUs and it takes ~15 mins.


```python
base = "configs/stable-diffusion/v1-finetune_unfrozen.yaml"
n = 'test'
gpus = 0,
data_root = "/root/to/training/images"
reg_data_root = '/path/to/regularization/images'

# train your model
os.system(f"python main.py --base {base} -t --actual_resume {ckpt} -n {n} " + 
          f"--gpus {gpus} --data_root {data_root} --reg_data_root {reg_data_root} --class_word {class_word}")
```

## Generate images using your saved checkpoints

> After training, personalized samples can be obtained by running the command

```python
saved_ckpt = "/path/to/saved/checkpoint/from/training"
new_prompt = f"photo of a wooden {class_word}"

# generate new images from a text prompt based on your saved checkpoints
os.system(f"python scripts/stable_txt2img.py --ddim_eta {ddim_eta} " + 
          f"--n_samples {n_samples} --n_iter {n_iter} --scale {scale} " + 
          f"--ddim_steps {ddim_steps} --ckpt {saved_ckpt} --prompt {new_prompt}")
```

## References

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{gal2022textual,
      doi = {10.48550/ARXIV.2208.01618},
      url = {https://arxiv.org/abs/2208.01618},
      author = {Gal, Rinon and Alaluf, Yuval and Atzmon, Yuval and Patashnik, Or and Bermano, Amit H. and Chechik, Gal and Cohen-Or, Daniel},
      title = {An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion},
      publisher = {arXiv},
      year = {2022},
      primaryClass={cs.CV}
}

@article{ruiz2022dreambooth,
  title   = {DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation},
  author  = {Nataniel Ruiz and Yuanzhen Li and Varun Jampani and Yael Pritch and Michael Rubinstein and Kfir Aberman},
  year    = {2022},
  journal = {arXiv preprint arXiv: Arxiv-2208.12242}
}
```

## Disclaimer
- The content provided in this repository is for demonstration purposes and not meant for production. You should use your own discretion when using the content.
- The ideas and opinions outlined in these examples are my own and do not represent the opinions of AWS.