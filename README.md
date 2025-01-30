<p align="center"> <a target="_blank"> <img src="https://github.com/adgiz05/graph-neural-networks-dlmasterupm/blob/main/utils/upm_logo.png?raw=true" width="600" alt="UPM Logo"> </a> </p> 
<h1 align="center">
    Generative Models
    <br>
    <small>Master in Deep Learning - Universidad Politécnica de Madrid</small>
</h1> 
<p align="center">Welcome to the official repository for the <strong>Generative Models</strong> course of the <a href="https://masterdeeplearning.etsisi.upm.es/">Master in Deep Learning</a> at the Universidad Politécnica de Madrid (UPM).</p>
<p align="center">
<strong>Coordinator:</strong> Javier Huertas-Tato
<br> 
<strong>Teachers:</strong> Javier Huertas-Tato & Pablo Miralles-González
</p>

## 📖 Lectures
### Lecture 1 - Introduction to Variational Autoencoders ([📈 Slides](session_1/MP_DL-GEN-1-AE-es.pptx) | [📎 Material](session_1))

In this lecture, we introduce Variational Autoencoders (VAEs), a generative deep
learning model that extends classical autoencoders by incorporating
probabilistic latent space modeling. We discuss the motivation behind VAEs,
their mathematical foundations, and their applications in data generation. The
lecture covers key concepts such as the reparameterization trick and the
mathematical formulation with the Kullback-Leibler divergence.

**Main learning objectives**

- Understand the motivation behind generative models and why VAEs are useful.
- Learn the mathematical formulation of VAEs.
- Understand the reparameterization trick and why it is necessary for backpropagation.  Gain an intuition for how VAEs learn meaningful latent representations for data generation.
- Implement a basic VAE using PyTorch.

**Secondary learning objectives**

- Visualize latent space representations and their effects on generated samples.

### Lecture 2 - Variational Autoencoders, further topics ([📈 Slides](https://upm365-my.sharepoint.com/:p:/g/personal/pablo_miralles_upm_es/EXRC90YW1ghFtacgPEZGPZoBjbKo0KU3gE4QRpy3hriSuw?e=f8nB8v) | [📎 Material](session_2))

In this lecture we continue learning about Variational Autoencoders. We learn
the necessary layers to build a Convolutional VAEs, and discuss VAE variations
such as the β-VAE, which introduces a controllable balance between
reconstruction quality and latent space disentanglement, and the Conditional
VAE, designed for generating data conditioned on specific attributes.

**Main learning objectives**
- Understand convolutional architectures that map from latent space to 2D structures such as images.
- Understand the effects of the $\beta$ parameter in $\beta$-VAEs.
- Understand the concept of Conditional VAE, and how it changes the way we generate samples from specific classes.
- Being able to adapt the VAE code to other problems and datasets.


**Secondary learning objectives**
- Being able to code VAEs and cVAEs from scratch.

### Lecture 3 - Introduction to Generative Adversarial Networks ([📈 Slides](https://upm365-my.sharepoint.com/:p:/g/personal/pablo_miralles_upm_es/EXkiiMBTVf5GhpO3hy8Ug1gBigGxEFF0lOnIpYjO3dRp-w?e=jy6oIi) | [📎 Material](session_3))

This lecture introduces Generative Adversarial Networks (GANs). We will cover
the training procedure and put it into practice with the MNIST dataset. We will
also treat common issues like mode collapse and training instability, and cover
practical tips to solve these problems and improve performance.

**Main learning objectives**
- Intuitively understand the architecture of GANs, as well as the training process.
- Being able to adapt the GAN code to other problems and datasets.
- Being able to tweak model sizes, learning rates and other parameters to help the training process converge.
- Understand some of the problems that can arise during the training of GANs.
- Understand the advantages and disadvantages of using GANs, as opposed to VAEs and Diffusion models.

**Secondary learning objectives**
- Being able to code GANs from scratch.
- Being able to code WGANs from scratch.
- Understand the probabilistic intuition behind the generator's loss, with the probabilistic divergence metric, and how that relates GANs and WGANs.

### Lecture 4 - Generative Adversarial Networks assignment ([📎 Material](session_4))

In the fourth session we continue with Generative Adversarial Networks by
programming a variant called CycleGAN, solidifying the main concepts and showing how GANs
can be used for much more than just generating new examples. In particular, CycleGANs are
applied for Image2Image translation. We will be using them to transform horses to zebras
in images, and viceversa.

**Main learning objectives**
- Understand how GANs can be applied to other tasks such as Image2Image translation.
- Understand CycleGANs conceptually.


### Lecture 5 - Autoregressive text generation ([📈 Slides](https://upm365-my.sharepoint.com/:p:/g/personal/pablo_miralles_upm_es/EXOIHKUonItKnHnnh-G513oBvyJQOJ4PQ1VCSzzJDX8UFA?e=XgW8x7) | [📎 Material](session_5))

This lecture covers the fundamentals of autoregressive text generation, focusing
on essential steps for building and training effective models. We’ll begin with
tokenization techniques and problem formulation, framing text generation as a
sequence prediction task. We will explore the training procedure and finalize
with  decoding strategies—such as greedy decoding, beam search, and sampling
methods—to enhance the quality and coherence of generated text. As practical
assignment, we will build a character level language model with both an LSTM
and a Transformer.

**Main learning objectives**
- Understand how autoregressive text generation work
- Understand tokenization, its trade-offs and caveats
- Understand how sequences of tokens are modeled for autoregression: the embedding layer, the sequence model (we will see only the Transformer) and the prediction head
- Understand the next-token prediction training objective
- Understand the main decoding strategies for text generation once the model is trained

**Secondary learning objectives**
- Know the different training stages of modern LLMs
- Know some techniques that are being used to augment LLM capabilities (RAG, tools, feedback loops and self-critique...)

### Lecture 6 - Diffusion models ([📈 Slides](https://docs.google.com/presentation/d/1A145e7MkpcmgIen9mG1r_nj3nkhuQAdMiP6NyQazkKQ/edit?usp=sharing) | [📎 Material](session_6))

This lecture focuses on diffusion models, the current state of the art in image
content generation. We will explore the fundamentals of diffusion, this includes
the main learning objective (noise prediction), noise samplers and noise
schedulers. We will explore the strategy named Denoising Diffusion Probabilistic
Models, or DDPM as the base for all our explorations. We will observe how a basic
DDPM example on a rudimentary dataset to illustrate the principles of noise prediction
and inference with denoising. Furthermore we will have a brief demo of using an actual
diffusion model locally (or in collab).

**Main learning objectives**
- Understand the core ideas that diffusion brings
- Understand the noise prediction objective
- Understand the denoising inference process
- Introducing the concepts of stable diffusion
- Basic understanding of text guidance

**Secondary learning objectives**
- Running a local diffusion model
- Code a diffusion minimalistic model from scratch

## ✏️ Assignment: Implementing a Conditional Variational Autoencoder (cVAE)

**Objective**:  
Develop a Conditional Variational Autoencoder (cVAE) to understand its structure, functionality, and application in generating conditioned outputs.  

**Deliverables**:  
A Jupyter notebook containing:  
- Data preprocessing steps.  
- Full implementation of the cVAE model.  
- Training and evaluation results.  
- Visualizations of generated samples.  
- A brief written explanation of your results and key observations.  

**Deadline**: 16 Feb 2025

Good luck, and happy coding!  

## Preparing the environment

### Local setup
With `conda` installed (or `mamba` or `micromamba`), run in your command line:
```bash
# if you have a gpu
conda env create --file conda-env-gpu.yml
# if you don't
conda env create --file conda-env-no_gpu.yml

# to activate the environment
conda activate mdl_gen
```

### Google Colab

To create the environment, add the following cells to the beginning of the notebook, after copying the `conda-env-gpu.yml` file to the directory.

**Step 1**: Install conda
```python
!pip install -q condacolab
import condacolab
condacolab.install()
```

**Step 2**: Create the environment
```python
!conda env create --file conda-env-gpu.yml
```

**Step 3**: Activate the environment
```python
!source activate mdl_gen
```