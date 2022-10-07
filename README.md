# crp-human-experiment

This is the repository for the human study of the Concept-Relevance-Propagation (CRP) paper:
Achtibat, Reduan, et al. "[From" Where" to" What": Towards Human-Understandable Explanations through Concept Relevance Propagation.](https://arxiv.org/pdf/2206.03208)" (2022).

We have three models trained on ImageNet: 
1. vgg16 (standard torchvision pretrained),
2. vgg16_corrupted (trained on border artifact for 30 classes) and
3. vgg16_uncorrupted (trained on border artifact for all classes).

Model checkpoints can be found [here](https://datacloud.hhi.fraunhofer.de/s/iExrEprxtJ5g38A).

In directory ./scripts/ there are several .sh files for running experiments.
- run_prepare_crp.sh to run analysis run for crp
- run_all.sh to run all methods for the samples given

Code is tested with Python 3.8.10. Required packages are listed in requirements.txt.

Please note the HTML template files for amazon mechanical turk as well as instructory images in directory ./html_templates. 
Study results are in .csv format in ./results/csv/ where _T or _F indicates the answers for the corrupted and uncorrupted model, respectively.

Regarding the CRP explanations,
concept labels are firstly predicted using [MILAN](https://github.com/evandez/neuron-descriptions) (Natural Language Descriptions of Deep Visual Features) (scripts located in ./utils/milan)