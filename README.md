# [ICML 2025] WUDI-Merging
The official repository of ["Whoever Started the Interference Should End It: Guiding Data-Free Model Merging via Task Vectors"](https://arxiv.org/abs/2503.08099)

## 🔥 News

- [2025/03/11] We release the code and paper of WUDI.
- [2025/05/01] **WUDI-merging has been accepted by ICML2025!**

## 💡 Introduction 

 In this work, we theoretically demonstrate that the task vectors of the linear layer constitute an approximate linear subspace for its corresponding input. Therefore, we can minimize interference under the guidance of task vectors. Based on this insight, we propose **WUDI-Merging** (**W**hoever started the interference sho**U**ld en**D** **I**t), a simple yet effective model merging method that eliminates interference without any additional data or rescaling coefficients. Comprehensive empirical evaluations across vision and language benchmarks demonstrate our method's superiority, achieving state-of-the-art performance in data-free model merging scenarios (average 10.9\% improvement versus baseline methods)  while even outperforming mainstream test-time adaptation approaches by 3.3\%, and only very few computing resources are required.