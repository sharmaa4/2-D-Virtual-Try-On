# Virtual Try On Project
GUI implementation for "CP-VTON+: Clothing Shape and Texture Preserving Image-Based Virtual Try-On" from CVPRW 2020, with some changes in the code to create a real life application.

	
## Usage
This pipeline is a combination of consecutive training and testing of GMM + TOM. GMM generates the warped clothes according to the target human. Then, TOM blends the warped clothes outputs from GMM into the target human properties, to generate the final try-on output.

## Running the Colab Notebook

1.) Download code base from [cp-vton-plus-gui](https://drive.google.com/drive/folders/1FbS8tMAJaq8mZeRLpe-x12rfgG6XDKz7?usp=sharing). This directory has thre pre-trained models 
as well.

2.) Create a Colab_Notebooks directory in your google drive and upload this directory there.

3.) Run the Notebook.

4.) Mount your google drive and execute all cells.

5.) After executing the last cells you will get some widgets. You can upload your image and cloth image by clicking on those widgets and then click "Generate Try-On" widget
    to get the results.
    
## Data preparation
For training/testing VITON dataset, our full and processed dataset is available here: https://1drv.ms/u/s!Ai8t8GAHdzVUiQQYX0azYhqIDPP6?e=4cpFTI. After downloading, unzip to your data directory.


### Acknowledgements
This implementation is largely based on the PyTorch implementation of [CP-VTON+](https://github.com/minar09/cp-vton-plus). I am extremely grateful for their public implementation.
