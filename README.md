# Reproducing E2GAN: Efficient Training of Efficient GANs for Image-to-Image Translation

This repository contains the implementation to reproduce the results of the paper:

**[E2GAN: Efficient Training of Efficient GANs for Image-to-Image Translation](https://arxiv.org/abs/2401.06127)**  
*Yifan Gong, Zheng Zhan, Qing Jin, Yanyu Li, Yerlan Idelbayev, Xian Liu, Andrey Zharkov, Kfir Aberman, Sergey Tulyakov, Yanzhi Wang, Jian Ren*

This implementation was developed as part of the **[Reproducibility Challenge](https://reproml.org/)**.  

## Abstract

The paper presents **E2GAN**, a novel and efficient framework for training GANs from diffusion models, enabling real-time, on-device image editing with reduced computational costs. Key innovations include:
1. A **generalized base GAN** for diverse image editing tasks.
2. Efficient **Low-Rank Adaptation (LoRA)** fine-tuning.
3. Optimization for minimal data requirements in training.

This repository reproduces the experimental setup and results presented in the paper, verifying the claims and analyzing the proposed methods.

---

## Goals of Reproducibility

1. **Verify key claims** of the paper, including:
   - Reduction in training time compared to traditional methods.
   - Efficiency of LoRA-based fine-tuning.
   - High-quality image editing with minimal data.
2. **Explore limitations and edge cases** of the proposed methods.
3. Provide a clear and well-documented implementation for future research.

## Citation

If you use this codebase or refer to the reproduced results, please cite the original paper and this repository:

### Original Paper
```bibtex
@article{gong2024e2gan,
  title={E2GAN: Efficient Training of Efficient GANs for Image-to-Image Translation},
  author={Gong, Yifan and Zhan, Zheng and Jin, Qing and Li, Yanyu and Idelbayev, Yerlan and Liu, Xian and Zharkov, Andrey and Aberman, Kfir and Tulyakov, Sergey and Wang, Yanzhi and Ren, Jian},
  journal={arXiv preprint: https://arxiv.org/abs/2401.06127},
  year={2024}
}
```

### Reproducibility Challenge
```bibtex
@misc{e2gan2024reproducibility,
  title={Reproducing E2GAN: Efficient Training of Efficient GANs for Image-to-Image Translation},
  author={Stefano Andreotti, Damiano Ficara, Stegano Quaggio, Claudio Ricci},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
