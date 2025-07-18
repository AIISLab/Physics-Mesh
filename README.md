# Generating Structurally Aware Physics CAD Sequences:

# Datasets:
Deepcad/ArchCAD-400k datasets are most ideal here, can use a collection of parametric CAD models to generate a sequence that generates into CAD structure
Annotation Pipeline: LLMs/VLMs to generate multi-view prompts from abstract to detailed parametric instructions following Text2CAD pipeline here: https://arxiv.org/html/2409.17106v1
Adding in additional annotations such as FEA (finite element analysis), linking model to structural performance (still need to look into this)

# Model Architecture:
Text Encoder: likely using BERT to map user prompt to contextual embeddings
Training an adaptive layer to help map text embeddings to latent space for CAD model generation
CAD Decoder: using a transformber-based autoregressive system to generate parametric CAD sequences (sketches, extrusions, etc. following Text2CAD)
Cross Attention: cross-attention mechanism to help align text instructions to cad sequences, so step-by-step generation is plausible

# Training:
Supervised-Pretraining: training model on annotated DeepCad dataset as seen above to learn mapping from text embeddings to CAD sequences
Reinforcement Learning Fine-Tuning: Using PPO/SAC to refine model outputs, making use of physics aware models such as NVIDIA-Warp/FEA to help compute rewards based off structural performance, user preference, etc.

# Action and State Design:
State: Representing the current CAD model (geometry, parameters, stress maps, etc.)
Action: Modifications (changing wall thickness, repositioning beam, etc.)
Reward: Based on prompt alignment, FEA results, CLIP/DPO scores

# Evaluation (Similar to Text2Cad): 
CAD Sequence Evaluation: We assess the parametric correspondence between the generated CAD sequences with the input texts. This is done using the following metrics:
F1 Scores of Line, Arc, Circle and Extrusion using the method proposed in CAD-SIGNet.
Chamfer Distance (CD) measures geometric alignment between the ground truth and reconstructed CAD models of Text2CAD and DeepCAD.
Invality Ratio (IR) Measures the invalidity of the reconstructed CAD models.
Visual Inspection: We compare the performance of Text2CAD and DeepCAD with GPT-4 and Human evaluation.