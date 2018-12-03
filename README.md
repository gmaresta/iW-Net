# iW-Net
iW-Net: an automatic and minimalistic interactive lung nodule segmentation deep network. By Guilherme Aresta, Colin Jacobs, Teresa Araújo, António Cunha, Isabel Ramos, Bram van Ginneken and Aurélio Campilho (December 2018)

**Please cite** http://arxiv.org/abs/1811.12789

We propose iW-Net, a deep learning model that allows for both automatic and interactive segmentation of lung nodules in computed tomography images. iW-Net is composed of two blocks: the first one provides an automatic segmentation and the second one allows to correct it by analyzing 2 points introduced by the user in the nodule's boundary. For this purpose, a physics inspired weight map that takes the user input into account is proposed, which is used both as a feature map and in the system's loss function. Our approach is extensively evaluated on the public LIDC-IDRI dataset, where we achieve a state-of-the-art performance of 0.55 intersection over union vs the 0.59 inter-observer agreement. Also, we show that iW-Net allows to correct the segmentation of small nodules, essential for proper patient referral decision, as well as improve the segmentation of the challenging non-solid nodules and thus may be an important tool for increasing the early diagnosis of lung cancer.

## Instructions
- Download the repository
- run GUI_guided_segmentation.py
- load one of the .npy files via "Browse"
- perform the inital segmentation via "Segment"
- press "Select points" to add 2 points to the nodule
- the segmentation will change accordingly





