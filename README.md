# DCINN-A-General-Paradigm-with-Detail-Preservation-Conditional-Invertible-Network-for-Image-Fusion
# Abstract
Existing deep learning techniques for image fusion either learn image mapping (LIM) directly, which renders them ineffective at preserving details due to the equal consideration to each pixel, or learn detail mapping (LDM), which only attains a limited level of performance because only details are used for reasoning. The recent lossless invertible network (INN) has demonstrated its detail-preserving ability. However, the direct applicability of INN to the image fusion task is limited by the volume-preserving constraint. Additionally, there is the lack of a consistent detail-preserving image fusion framework to produce satisfactory outcomes. To this aim, we propose a general paradigm for image fusion based on a novel conditional INN (named DCINN). The DCINN paradigm has three core components: a decomposing module that converts image mapping to detail mapping; an auxiliary network (ANet) that extracts auxiliary features directly from source images; and a conditional INN (CINN) that learns the detail mapping based on auxiliary features. The novel design benefits from the advantages of INN, LIM, and LDM approaches while avoiding their disadvantages. Particularly, using INN to LDM can easily meet the volume-preserving constraint while still preserving details. Moreover, since auxiliary features serve as conditional features, the ANet allows for the use of more than just details for reasoning without compromising detail mapping. Extensive experiments on three benchmark fusion problems, i.e., pansharpening, hyperspectral and multispectral image fusion, and infrared and visible image fusion, demonstrate the superiority of our approach compared with recent state-of-the-art methods. The code will be available after possible acceptance.
# Training
## 1.Data Preparation

## 2.DCINN Training
For Pansharpening task, run train_dcinn_ps.py

For HMF task, run train_dcinn_ps.py

For IVF (RS dataset) task, run train_dcinn_ivf_rs.py

For IVF (TNO dataset) task, run train_dcinn_ivf_tno.py
# Testing
## 1.Pretrained Model
the pretrained models for all the image fusion task is available at ./pretrained/
## 2.Test dataset
For HMF task, run test_dcinn_ps.py

For IVF (RS dataset) task, run test_dcinn_ivf_rs.py

For IVF (TNO dataset) task, run test_dcinn_ivf_tno.py

This paper has been accepted by the international journal of computer vision (IJCV)
