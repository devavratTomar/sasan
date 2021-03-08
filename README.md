# Self-Attentive Spatial Adaptive Normalization for Cross-Modality Domain Adaptation

## Requirements
- python >=3.6
- pytorch >=1.6
- tensorflow == 1.15
- medpy
- kornia

## Training

- To launch the training please run `train.py`. The hyperparameters can be updated in `def main` function as a dictionary.
- For faster convergence, please pretrain the attention module for the domain whose segmenation labels are available, by running `python train_segmentation.py attention_mr`
- For training the upper bount U-Net on MRI modality, use the following command - `python train_segmentation.py mr`

- To evaluate the trained model, please run `python run_evaluation.py sasan ct` for evaluating the performance of <b>MRI to CT</b> domain adaptation. For the other direction <b>CT to MRI</b>, run `python run_evaluation.py sasan mr`.

## Pre-trained models, datasets, code:
* [Link to our pre-trained models on Whole Heart Multimodal dataset and code.](https://drive.google.com/drive/folders/1J4oxsVQME3ee95DBwZ_ZZLFlR3DHYXcA?usp=sharing)
* Link to [Whole Heart Multimodal dataset](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/) pre-processed training tf_record files can be found [here](https://drive.google.com/file/d/1m9NSHirHx30S8jvN0kB-vkd7LL0oWCq3/view). The test mr data is available [here](https://drive.google.com/file/d/1RNb-4iYWUaFBY61rFAnT2XT0mtwlnH1V/view) and test ct data is available [here](https://drive.google.com/file/d/1SJM3RluT0wbR9ud_kZtZvCY0dR9tGq5V/view)

## Data preprocessing
* To convert the tf_records training data to `.npy` format please use the script `convert_tfrecords.py <modality>`, where `<modality>` is either `mr` or `ct`.

### Licence

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
