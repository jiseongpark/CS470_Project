"# **CS470_Project**"

# **Getting Started**

Type the following command on your prompt screen to demo:
```python
python3 cifar.py --nce-k 0 --nce-t 0.1 --lr 0.03
```
This python project is based on the paper
["Unsupervised feature learning via non-parametric instance-level discrimination."](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0801.pdf)

# **Pre-trained Model**
you can download the pre-trained model in ["here"]
(https://drive.google.com/drive/folders/1FoEO1IuxjIEiTet2k7DbaLq8VNh8Jdw3?usp=sharing)

# **Abstract**
Neural net classifiers trained on data with annotated class labels can also capture apparent 
visual similarity among categories without being directed to do so. We study whether this 
observation can be extended beyond the conventional domain of supervised learning: Can
we learn a good feature representation that captures apparent similarity among instances, 
instead of classes, by merely asking the feature to be discriminative of individual instances? 
We formalize this intuition as a nonparametric classification problem at the instance level 
and use noise contrast estimates to solve the computational challenges posed by multiple 
instance classes based on ResNet18.

# **Method**
Our method has two main ideas, Rationale and Weighted Formation Feature.
Rationale samples on the embedding space which have similar features will gather 
and take apart if they differ. Thus, if the embedding space has a reasonable 
dimension and size limit, the samples should cluster for some classes. However, 
if each sample are set to individual classes, they should take apart from each 
other due to the influence of cross-entropy. The embedding space has its own 
size limit so that each will undergo clustering since samples attract or repulse each 
other by their own features. That is the key point of the suggested classification 
method.
Weighted Formation Feature is when we traing dataset to our model, network can 
focus on the color tone of each sample. However, too much attention on it can 
lead to undesirable training by learning similar color tone be the same classes 
while each of them are completely mismatched.  For this, we conducted a novel 
technique, the Sobel filter. Concatenate image edge information of each sample 
carried out with Sobel filter and use them as input. 

# **Citing**
```python
@misc{CS470_Project,
  author       = {Park Sungwon, Park Jiseong, 
                  Lee Sungwook and Jeon Hojin},
  title        = {{CS470 Final Project Report}},
  month        = nov,
  year         = 2019,
  url          = {https://github.com/jiseongpark/CS470_Project}
}
```
