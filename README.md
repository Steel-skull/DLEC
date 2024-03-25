# REPO IS A WORK IN PROGRESS

the Colab fully works, it is in the ./colab folder

I am rewriting the code to make it a usable app it "should" be funtional

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

## How-To
```
python main.py --model <model_name> --dataset <dataset_name> [--dbpath <database_path>]
```

--model (required): 
Specify the name or identifier of the model you want to use for activation analysis. This should be a valid model identifier from the Hugging Face Models Hub. For example:
```
--model google/gemma-2b
```

--dataset (required): 
Specify the name or identifier of the dataset you want to use for activation analysis. This should be a valid dataset identifier from the Hugging Face Datasets Hub. For example:
```
--dataset wikimedia/wikipedia
```

--dbpath (optional): 
Specify the path where you want to store the SQLite database file for the activation analysis results. If not provided, it defaults to ./activations.db in the current directory. For example:
```
--dbpath /path/to/activations.db
```

# Information Loss:
It is common to observe a loss of intelligence when merging models, especially with Passthrough merging, which typically results in a loss of around 3 points per billion parameters duplicated, assuming the merge is done correctly. If the merge is suboptimal, the loss can be much larger, ranging from 3-8 points or more per billion parameters duplicated. However, with DLEC, I was able to increase Phi-2 from 2.78b to 3.25b with a minimal loss of around 0.44 points on average.

DLEC Expanded Model:
[TheSkullery/phi-2-DLEC](https://huggingface.co/TheSkullery/phi-2-DLEC)
2.78 -> 3.25, a ~17% increase in size
```
Metric -> Value
Avg. 46.72
AGIEval 29.64
GPT4All 69.48
TruthfulQA 50.29
```

Original Model:
[abacaj/phi-2-super](https://huggingface.co/abacaj/phi-2-super))
```
Metric -> Value
Avg. 47.16
AGIEval 31.95
GPT4All 70.81
TruthfulQA 48.39
```

Loss or Increase:
Avg. -0.44
AGIEval -2.31
GPT4All -1.33
TruthfulQA +1.90

Example of loss:
[Steelskull/Etheria-55b-v0.1](https://huggingface.co/Steelskull/Etheria-55b-v0.1)
```
Metric -> Value
Avg. 64.69
AI2 Reasoning Challenge 65.10
HellaSwag 81.93
MMLU 73.66
TruthfulQA 56.16
Winogrande 76.09
GSM8k 35.18
```

[Yi-34B-200K-DARE-megamerge-v8](https://huggingface.co/brucethemoose/Yi-34B-200K-DARE-megamerge-v8)
```
Metric -> Value
Avg. 72.56
AI2 Reasoning Challenge 67.75
HellaSwag 86.06
MMLU 77.03
TruthfulQA 56.31
Winogrande 82.79
GSM8k 65.43
```

Merge Loss (Yi-34B-200K-DARE-megamerge-v8 compared to Etheria-55b-v0.1):
Avg. -7.87
AI2 Reasoning Challenge -2.65
HellaSwag -4.13
MMLU -3.37
TruthfulQA +0.15
Winogrande -6.70
GSM8k -30.25

In the example comparing Etheria-55b-v0.1 and Yi-34B-200K-DARE-megamerge-v8, there is a significant decrease in performance across all metrics, with the average score decreasing by 7.87 points. The most notable is in the GSM8k benchmark, where Yi-34B-200K-DARE-megamerge-v8 outperforms Etheria-55b-v0.1 by 30.25 points.

---
This method is still in active development, and I am currently tweaking the algorithm to improve the layer selection process, I am also working on a single layer duping script as merge kit does not currently support this and I am being forced to merge layers that are unneeded and its degrading performance.
