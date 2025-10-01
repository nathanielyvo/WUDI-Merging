# [ICML 2025] WUDI-Merging
The official repository of ["Whoever Started the Interference Should End It: Guiding Data-Free Model Merging via Task Vectors"](https://arxiv.org/abs/2503.08099)

## 🔥 News

- [2025/03/11] We release the code and paper of WUDI.
- [2025/05/01] **WUDI-merging has been accepted by ICML2025!**

## 💡 Introduction 

 In this work, we theoretically demonstrate that the task vectors of the linear layer constitute an approximate linear subspace for its corresponding input. Therefore, we can minimize interference under the guidance of task vectors. Based on this insight, we propose **WUDI-Merging** (**W**hoever started the interference sho**U**ld en**D** **I**t), a simple yet effective model merging method that eliminates interference without any additional data or rescaling coefficients. Comprehensive empirical evaluations across vision and language benchmarks demonstrate our method's superiority, achieving state-of-the-art performance in data-free model merging scenarios (average 10.9\% improvement versus baseline methods)  while even outperforming mainstream test-time adaptation approaches by 3.3\%, and only very few computing resources are required.
## 📊 Evaluation Results

## Multi-task performance when merging ViT-B/32 models on 8-task vision benchmark

| Method | SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD | Avg Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Non-merging Methods** ||||||||| |
| Pretrained | 62.3 | 59.7 | 60.7 | 45.5 | 31.4 | 32.6 | 48.5 | 43.8 | 48.0 |
| Individual | 79.2 | 77.7 | 96.1 | 99.7 | 97.5 | 98.7 | 99.7 | 79.4 | 90.8 |
| Traditional MTL | 73.9 | 74.4 | 93.9 | 98.2 | 95.8 | 98.9 | 99.5 | 77.9 | 88.9 |
| **Test-time Adaptation Methods** ||||||||| |
| AdaMerging | 64.5 | 68.1 | 79.2 | 93.8 | 87.0 | 91.9 | 97.5 | 59.1 | 80.1 |
| AdaMerging++ | 66.6 | 68.3 | 82.2 | 94.2 | 89.6 | 89.0 | 98.3 | 60.6 | 81.1 |
| Representation Surgery | 63.8 | 59.9 | 83.3 | 97.9 | 87.0 | 87.0 | 98.6 | 69.4 | 80.9 |
| **Data-free Methods** ||||||||| |
| Weight Averaging | 65.3 | 63.4 | 71.4 | 71.7 | 64.2 | 52.8 | 87.5 | 50.1 | 65.8 |
| Fisher Merging | 68.6 | 69.2 | 70.7 | 66.4 | 72.9 | 51.1 | 87.9 | 59.9 | 68.3 |
| RegMean | 65.3 | 63.5 | 75.6 | 78.6 | 78.1 | 67.4 | 93.7 | 52.0 | 71.8 |
| Task Arithmetic | 55.2 | 54.9 | 66.7 | 78.9 | 80.2 | 69.7 | 97.3 | 50.4 | 69.1 |
| Ties-Merging | 59.8 | 58.6 | 70.7 | 79.7 | 86.2 | 72.1 | 98.3 | 54.2 | 72.4 |
| Consensus Merging | 65.7 | 63.6 | 76.5 | 77.2 | 81.7 | 70.3 | 97.0 | 57.1 | 73.6 |
| PCB Merging | 66.7 | 65.5 | 78.5 | 79.3 | 86.4 | 77.1 | 98.2 | 59.1 | 76.3 |
| **WUDI-Merging (Ours)** | **71.1** | **71.0** | **85.7** | **95.6** | **94.2** | **94.7** | **99.5** | **69.7** | **85.2** |


---

## Multi-task performance when merging ViT-L/14 models on 8-task vision benchmark

| Method | SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD | Avg Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Non-merging Methods** ||||||||| |
| Pretrained | 66.8 | 77.7 | 71.0 | 59.9 | 58.4 | 50.5 | 76.3 | 55.3 | 64.5 |
| Individual | 82.3 | 92.4 | 97.4 | 100.0 | 98.1 | 99.2 | 99.7 | 84.1 | 94.2 |
| Traditional MTL | 80.8 | 90.6 | 96.3 | 96.3 | 97.6 | 99.1 | 99.6 | 84.4 | 93.5 |
| **Test-time Adaptation Methods** ||||||||| |
| AdaMerging | 79.0 | 90.3 | 90.8 | 96.2 | 93.4 | 98.0 | 99.0 | 79.9 | 90.8 |
| AdaMerging++ | 79.4 | 90.3 | 91.6 | 97.4 | 93.4 | 97.5 | 99.0 | 79.2 | 91.0 |
| Representation Surgery | 75.7 | 84.4 | 93.1 | 98.8 | 91.3 | 93.4 | 99.1 | 76.1 | 89.0 |
| **Data-free Methods** ||||||||| |
| Weight Averaging | 72.1 | 81.6 | 82.6 | 91.9 | 78.2 | 70.7 | 97.1 | 62.8 | 79.6 |
| Fisher Merging | 69.2 | 88.6 | 87.5 | 93.5 | 80.6 | 74.8 | 93.3 | 70.0 | 82.2 |
| RegMean | 73.3 | 81.8 | 86.1 | 97.0 | 88.0 | 84.2 | 98.5 | 60.8 | 83.7 |
| Task Arithmetic | 73.9 | 82.1 | 86.6 | 94.1 | 87.9 | 86.7 | 98.9 | 65.6 | 84.5 |
| Ties-Merging | 76.5 | 85.0 | 89.3 | 95.7 | 90.3 | 83.3 | 99.0 | 68.8 | 86.0 |
| Consensus Merging | 75.0 | 84.3 | 89.4 | 95.6 | 88.3 | 82.4 | 98.9 | 68.0 | 85.2 |
| PCB Merging | 76.8 | 86.2 | 89.4 | 96.5 | 88.3 | 91.0 | 98.6 | 73.6 | 87.5 |
| **WUDI-Merging (Ours)** | **81.0** | **91.0** | **94.2** | **99.2** | **96.3** | **98.1** | **99.6** | **81.2** | **92.6** |

---

## Performance of merging decoder-based models (WizardLM-13B, WizardMath-13B, llama-2-13b-codealpaca)

| Method | AlpacaEval | GSM8K | MATH | HumanEval | MBPP | Avg. |
|---|---:|---:|---:|---:|---:|---:|
| FT | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Task Arithmetic | 102.7 | 91.0 | 70.5 | 50.0 | 87.7 | 80.4 |
| TIES-Merging | 98.1 | 97.4 | 68.1 | 60.0 | 89.4 | 82.6 |
| Task Arithmetic (w/ DARE) | 103.1 | 88.0 | 72.5 | 63.3 | 92.9 | 84.0 |
| TIES-Merging (w/ DARE) | 107.9 | 90.3 | 65.6 | 80.0 | 92.4 | 87.2 |
| **WUDI-Merging (Ours)** | **105.5** | **105.9** | **103.3** | **58.3** | **84.7** | **91.5** |


---

## Multi-task performance when merging RoBERTa models on 8-task GLUE benchmark (average normalized score)

| Method | RoBERTa-Base | RoBERTa-Large |
|---|---:|---:|
| Pretrained | 41.7 | 38.2 |
| Individual | 100.0 | 100.0 |
| Weight Averaging | 52.6 | 53.3 |
| Task Arithmetic | 67.8 | 70.9 |
| Ties-Merging | 64.7 | 72.4 |
| Task Arithmetic (w/ DARE) | 63.7 | 70.9 |
| Ties-Merging (w/ DARE) | 65.6 | 72.8 |
| **WUDI-Merging (Ours)** | **85.3** | **88.8** |



---

Experimental results of merging Qwen-14B (LoRA fine-tuned) models on all four tasks

| Method | MMLU | TruthfulQA | BBQ | CNN | Avg. |
|---|---:|---:|---:|---:|---:|
| Individual | 68.35 | 53.34 | 93.53 | 19.46 | 58.67 |
| Task Arithmetic | 67.56 | 52.33 | 78.38 | 20.54 | 54.70 |
| Ties-Merging (w/ DARE) | 69.38 | 52.03 | 81.06 | 15.91 | 54.62 |
| **WUDI-Merging (Ours)** | **69.17** | **55.71** | **80.56** | **17.33** | **55.69** |


## 📚 Reproduce
For the experiment of Vit, you can use the code in [WUDI-Merging/vit](https://github.com/nathanielyvo/WUDI-Merging/tree/main/vit)

For the experiment of Roberta, you can use the code in [WUDI-Merging/nlp_roberta](https://github.com/nathanielyvo/WUDI-Merging/tree/main/nlp_roberta)

For the experiment of merging WizardLM-13B, WizardMath-13B, Llama-2-13B-codealpaca, you can refer the code in [WUDI-Merging/nlp_roberta](https://github.com/yule-BUAA/MergeLM)

For the experiment of merging Qwen-14B LoRA fine-tuned models, you can refer the code in [Twin-Merging](https://github.com/LZY-the-boys/Twin-Merging?tab=readme-ov-file)


## 📖 Citation
If you find WUDI-Merging useful for your research and applications, please cite using this BibTeX:
```bash
@article{cheng2025whoever,
  title={Whoever started the interference should end it: Guiding data-free model merging via task vectors},
  author={Cheng, Runxi and Xiong, Feng and Wei, Yongxian and Zhu, Wanyun and Yuan, Chun},
  journal={arXiv preprint arXiv:2503.08099},
  year={2025}
}
```
## 🎫 License

This project is released under the MIT License.

## 💕 Acknowledgments

Thanks to the following excellent open-source projects:\
[Editing Models with Task Arithmetic](https://github.com/mlfoundations/task_vectors)\
[Twin-Merging: Dynamic Integration of Modular Expertise in Model Merging](https://github.com/LZY-the-boys/Twin-Merging?tab=readme-ov-file)\
[Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://github.com/yule-BUAA/MergeLM)
