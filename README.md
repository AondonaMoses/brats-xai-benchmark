# brats-xai-benchmark: Benchmarking Explainable AI for Brain Tumour Segmentation and Classification in African MRI

A systematic benchmark of **explainability under domain shift**: three 2D classifiers and four 3D segmentation models, evaluated zero-shot and domain-adapted from **BraTS-GLI** (source) to **BraTS-Africa** (target), with **Grad-CAM, RISE, and Grad-CAM++** scored for overlap, faithfulness, and pointing-game accuracy, not just Dice/AUC.

This repository accompanies the paper *"Benchmarking Explainable AI for Brain Tumour Segmentation and Classification in African MRI"*.

> ⚠️ **Release status:** This repository currently contains the project scaffold, configs, and module interfaces. The full evaluation implementation will be added shortly. Please check back, or watch this repository for updates.

## Motivation

Deep learning for brain tumour analysis holds significant promise for Sub-Saharan Africa, where specialist neuroradiologists are scarce and MRI infrastructure is limited. But models trained on high-income country datasets are rarely evaluated for explainability on African MRI data, and reporting Dice or AUC alone says nothing about whether a model's attention is actually on the tumour. Curated resources such as BraTS-Africa exist precisely to expose this imbalance, yet prior work targets segmentation accuracy alone, whether XAI methods retain their reliability under the same distribution shift is unexamined. This benchmark closes that gap.

The setting poses three coupled questions: does predictive performance transfer across the domain shift, does explanation quality transfer with it, and can high "faithfulness" scores be trusted at face value, or do they mask attributions that are confidently pointing at the wrong anatomy.

## Method

A matched, three-part evaluation:

1. **Matched cross-domain benchmark.** 146 subjects from BraTS-GLI (source, subsampled from the 1,251-subject 2021 training set) are paired with 146 subjects from BraTS-Africa (target: 95 Glioma, 51 Other Neoplasms), balancing cohort size for direct comparison.
2. **Classification and segmentation under domain shift.** Three 2D classifiers, ResNet18, EfficientNet-B0, ViT-Small, are trained on GLI slices and evaluated zero-shot on SSA. Four 3D segmentation models, UNet, Attention UNet, TransUNet, Swin-UNETR, are domain-adapted to SSA via full fine-tuning across five cross-validation folds.
3. **XAI evaluation.** Grad-CAM and RISE are applied to the classifiers; Grad-CAM and Grad-CAM++ to the segmentation models. Explanations are scored by overlap (IoU with the tumour mask), faithfulness (perturbation sensitivity), and pointing-game accuracy (whether peak saliency falls inside the tumour).

## Results

**Zero-shot classification transfer (438 SSA slices):**

| Model | GLI Val AUC | SSA AUC | SSA Bal. Acc. | NCR Recall |
|---|---|---|---|---|
| ResNet18 | 1.000 | 0.653 | 0.500 | 0/47 |
| EfficientNet-B0 | 0.982 | 0.641 | 0.499 | 0/47 |
| **ViT-Small** | 1.000 | 0.642 | **0.597** | **41/47** |

**Segmentation domain adaptation (5-fold CV, mean DSC over WT+TC):**

| Model | Mean ± Std DSC |
|---|---|
| UNet | 0.7330 ± 0.028 |
| Att-UNet | 0.7314 ± 0.011 |
| TransUNet | 0.8142 ± 0.045 |
| **Swin-UNETR** | **0.8411 ± 0.027** |

**Classification XAI (Grad-CAM, 438 SSA slices):**

| Model | Overlap | Faithfulness | Pointing Game |
|---|---|---|---|
| ResNet18 | 0.101 | −0.000 | 0.030 |
| EfficientNet-B0 | 0.196 | 0.002 | 0.244 |
| **ViT-Small** | **0.536** | **0.242** | **0.655** |

Transformer-based architectures consistently outperform CNNs in both cross-domain performance and explanation quality. Critically, high faithfulness does not imply trustworthy explanations: UNet and Attention UNet produce confounded attributions, near-zero segmentation overlap paired with Grad-CAM faithfulness as high as 0.844, meaning the model is genuinely sensitive to the regions it highlights, but those regions are not the lesion.

Full per-class tables, segmentation XAI breakdown (Table 5), and fold-level results are in the paper.

## Repository structure

> The structure below reflects the **full pipeline to be released shortly**.

```
.
├── configs/                     # classification.yaml, segmentation.yaml, xai.yaml
├── src/
│   ├── data/                    # BraTS-GLI / BraTS-Africa loading, matched 146-subject sampling,
│   │                             #   2D slice extraction, preprocessing (resample, skull-strip, z-score)
│   ├── models/                  # ResNet18, EfficientNet-B0, ViT-Small classifiers;
│   │                             #   UNet, Att-UNet, TransUNet, Swin-UNETR segmentation models
│   ├── xai/                     # Grad-CAM, Grad-CAM++, RISE, and the overlap / faithfulness /
│   │                             #   pointing-game metrics
│   └── eval/                    # zero-shot + domain-adaptation evaluation, cross-validation folds
├── scripts/                     # train_classifier.py, train_segmentation.py, adapt_segmentation.py,
│                                 #   run_classification_xai.py, run_segmentation_xai.py
├── figures/                     # XAI metric charts, DSC comparison plots
├── results/                     # metrics tables, logs
├── paper/                       # paper.pdf
├── poster/                      # poster.pdf
└── README.md
```

## Reproduction order

Commands for each step live in `scripts/`.

1. **Data preparation:** obtain BraTS 2021 and BraTS-Africa (see [Data](#data)), point `configs/*.yaml` at local paths, and build the matched 146-subject sample.
2. **Classification, zero-shot:** `python scripts/train_classifier.py --config configs/classification.yaml --model resnet18|efficientnet_b0|vit_small`, then evaluate directly on SSA slices, no fine-tuning.
3. **Segmentation, GLI:** `python scripts/train_segmentation.py --config configs/segmentation.yaml --model unet|att_unet|transunet|swin_unetr`.
4. **Segmentation, SSA adaptation:** `python scripts/adapt_segmentation.py --config configs/segmentation.yaml --gli-ckpt <path> --fold 0..4`.
5. **Classification XAI:** `python scripts/run_classification_xai.py --ckpt <path> --methods gradcam,rise`.
6. **Segmentation XAI:** `python scripts/run_segmentation_xai.py --ckpt <path> --methods gradcam,gradcam++ --fold 0`.

## Data

Experiments use **BraTS 2021** (GLI, source) and **BraTS-Africa** (SSA, target). BraTS-GLI: 146 of 1,251 training subjects, subsampled to match the target domain size, four co-registered modalities (T1, T1-Gd, T2, T2-FLAIR) with whole tumour (WT), tumour core (TC), and enhancing tumour (ET) annotations. BraTS-Africa: 146 Sub-Saharan African subjects (95 Glioma, 51 Other Neoplasms) from lower-field-strength scanners, five stratified cross-validation folds; BraTS-Africa lacks ET annotations, so ET is excluded from all SSA metrics. For 2D classification, the single most tumour-informative axial, coronal, and sagittal slice per subject is extracted (438 slices per domain), with SSA labels binarised (ED vs. NCR). Neither dataset is redistributed here; obtain them from their official sources: [BraTS 2021](https://doi.org/10.7937/jc8x-9874), [BraTS-Africa](https://doi.org/10.7937/v8h6-8x67).

## Setup (planned)

All experiments run on the Narval cluster (NVIDIA A100 SXM4 40 GB, SLURM), Python 3.10, PyTorch 2.1, MONAI 1.3. Classification backbones are ImageNet-pretrained via `timm`; segmentation models are trained from scratch on GLI, with domain-adaptation runs initialised from the GLI-trained checkpoints. All seeds fixed at 42. Full environment and dependency details accompany `requirements.txt`.

## Citation

```bibtex
@inproceedings{olabode2026benchmarking,
  title     = {Benchmarking Explainable AI for Brain Tumour Segmentation and
               Classification in African MRI},
  author    = {Olabode, Ayotola Hannah and Iorumbur, Aondona Moses and
               Raymond, Confidence and Onifade, Olufade Williams},
  year      = {2026}
}
```

## Acknowledgements

Computing resources were provided by the Digital Research Alliance of Canada ([alliancecan.ca](https://alliancecan.ca)), through the Multimodal Imaging of Neurodegenerative Diseases (MiND) Lab, Montreal Neurological Institute, McGill University, Montreal, Canada.

## License

Released under the MIT License. See [`LICENSE`](LICENSE). BraTS 2021 and BraTS-Africa are subject to their own terms; see the linked dataset pages for details.
