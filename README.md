# REPO IS A WORK IN PROGRESS

Currently, Only the Colab works as I build out the full Repo it is in the ./colab folder

### DLEC (Distributive Layer Expansion Curve)

The DLEC (Distributive Layer Expansion Curve) methodology offers a novel approach to improving neural network models by focusing on the strategic duplication of certain effective layers. Developed with the aim of enhancing model performance, DLEC carefully identifies and amplifies the impact of key layers within the model's architecture.

## Below is the overview:
# Overview Setting Up: 
First, the script ensures all necessary components are in place, from libraries to the model and dataset.
# Database for Activations: 
A SQLite database is established to track layer activations, providing a clear view into how individual neurons react and which layers are most influential — these are our 'beneficial layers.'
# Analyzing and Identifying: 
By analyzing activation data, the script pinpoints which layers are most valuable to the model's performance.
# Configuring DLEC: 
A configuration is then created, guiding how the model should incorporate duplicates of these beneficial layers to boost effectiveness without unnecessarily increasing complexity.
#Reconfiguring and Running the Model: 
Finally, the model is adjusted according to DLEC's insights, focusing enhancement on the identified layers.

## Key Features: 
# Selective Layer Duplication: 
DLEC doesn't just add more layers; it doubles down on the ones that really matter. This methodical selection ensures we're making the most of the model's capabilities without wasteful expansion.
# Smart Resource Management: 
By honing in on specific areas for improvement, DLEC aims to make better use of computational and memory resources, promoting more efficient learning without adding undue complexity to the model.

## Algorithm

![image](https://github.com/Steel-skull/DLEC/assets/79706171/53c19a1a-13d4-4601-b593-cae263a7f9fa)

This approach is about making informed, strategic enhancements to model architecture, prioritizing efficiency and effectiveness in utilizing neural network capabilities.

# First Successful Model:

2.78-> 3.25 a ~16% increase in size

https://huggingface.co/TheSkullery/phi-2-DLEC

As you know there is a usual loss of intelligence with model mergers, especially with Passthrough merging on the par of 3ish points per billion duped, IF you get the right merge, if not your looking at a much larger loss (anywhere from 3-8 points per billion duped), With DLEC, I was able to increase Phi-2 from 2.78b -> 3.25b with around a single point of loss.

This method is still in active development, and I am currently tweaking the algorithm to improve the layer selection process, I am also working on a single layer duping script as merge kit does not currently support this and I am being forced to merge layers that are unneeded and its degrading performance.
