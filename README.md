# Master in Deep Learning - Generative Models

Welcome to the base repository of the "Generative Models" course! In this README you will find:
- **Overview of the lectures**. An overview of each of the lectures, with a brief summary, the learning objectives and links to the slides.
- **Assignment**. A brief description of the assignment.
- **Environment setup**. Some simple steps to recreate our conda environment, which should have everything you need for the course!

You will also find one directory for each class. Inside, you will find an accompanying notebook to the lectures, as well as notebooks with exercises and their solutions.

## Lectures
### Lecture 1 - Introduction to Variational Autoencoders ([Slides]())

### Lecture 2 - Variational Autoencoders, further topics ([Slides](https://upm365-my.sharepoint.com/:p:/g/personal/pablo_miralles_upm_es/EXRC90YW1ghFtacgPEZGPZoBjbKo0KU3gE4QRpy3hriSuw?e=f8nB8v))

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

### Lecture 3 - Introduction to Generative Adversarial Networks ([Slides](https://upm365-my.sharepoint.com/:p:/g/personal/pablo_miralles_upm_es/EXkiiMBTVf5GhpO3hy8Ug1gBigGxEFF0lOnIpYjO3dRp-w?e=jy6oIi)

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

### Lecture 4 - Generative Adversarial Networks assignment ([Slides]())

### Lecture 5 - Autoregressive text generation ([Slides](https://upm365-my.sharepoint.com/:p:/g/personal/pablo_miralles_upm_es/EXOIHKUonItKnHnnh-G513oBvyJQOJ4PQ1VCSzzJDX8UFA?e=XgW8x7))

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

### Lecture 6 - Diffusion models ([Slides](https://docs.google.com/presentation/d/1A145e7MkpcmgIen9mG1r_nj3nkhuQAdMiP6NyQazkKQ/edit?usp=sharing))

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

## Assignment: Implementing a Conditional Variational Autoencoder (cVAE)

**Objective**:  
Develop a Conditional Variational Autoencoder (cVAE) to understand its structure, functionality, and application in generating conditioned outputs.  

**Deliverables**:  
A Jupyter notebook containing:  
- Data preprocessing steps.  
- Full implementation of the cVAE model.  
- Training and evaluation results.  
- Visualizations of generated samples.  
- A brief written explanation of your results and key observations.  

**Deadline**: TBD

**Grading Criteria**:  
- Completeness of implementation
- Model performance and results
- Clarity and organization of the notebook
- Quality of explanations and visualizations

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