# [ICML 2025] WUDI-Merging
The official repository of ["Whoever Started the Interference Should End It: Guiding Data-Free Model Merging via Task Vectors"](https://arxiv.org/abs/2503.08099)

## 🔥 News

- [2025/03/11] We release the code and paper of WUDI.
- [2025/05/01] **WUDI-merging has been accepted by ICML2025!**

## 💡 Introduction 

 In this work, we theoretically demonstrate that the task vectors of the linear layer constitute an approximate linear subspace for its corresponding input. Therefore, we can minimize interference under the guidance of task vectors. Based on this insight, we propose **WUDI-Merging** (**W**hoever started the interference sho**U**ld en**D** **I**t), a simple yet effective model merging method that eliminates interference without any additional data or rescaling coefficients. Comprehensive empirical evaluations across vision and language benchmarks demonstrate our method's superiority, achieving state-of-the-art performance in data-free model merging scenarios (average 10.9\% improvement versus baseline methods)  while even outperforming mainstream test-time adaptation approaches by 3.3\%, and only very few computing resources are required.

## 📚 Reproduce
For the experiment of Vit, you can use the code in [WUDI-Merging/vit](https://github.com/nathanielyvo/WUDI-Merging/tree/main/vit)

For the experiment of Roberta, you can use the code in [WUDI-Merging/nlp_roberta](https://github.com/nathanielyvo/WUDI-Merging/tree/main/nlp_roberta)


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
