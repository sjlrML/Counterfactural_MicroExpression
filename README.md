### Counterfactual Discriminative Micro-Expression Recognition
Abstract Micro-expressions are spontaneous, rapid and subtle facial movements that can hardly be suppressed or fabricated. Micro-expression recognition (MER) is one of the most challenging topics in affective computing. It aims to recognize the subtle facial movements which are quite difficult for humans to perceive in a fleeting period. We propose a **Co**unterfactual **D**iscriminative micro-**E**xpression **R**ecognition (CoDER) method to effectively learn the slight temporal variations for video-based MER. Extensive experiments on four widely-used ME databases demonstrate the effectiveness of CoDER, which obtains comparable or superior MER performance compared with the state-of-the-art methods. Visualization results show CoDER successfully perceives the meaningful temporal variations of the sequential faces.

![](figure2.jpg)
Main idea of **CoDER**. 

Given an input ME sequence, we extract the sequential ME features via a transformer-based backbone network. The sequential features are encoded by LSTM and fed into the counterfactual learning part, where we compare the prediction of the normal and the counterfactually-revised ME sequences. The counterfactually-revised ME sequences are generated via random frame replacement or dropping to manipulate the temporal variation T. Here T is actually an abstract representation and means the temporal characteristics in the ME sequences. Better viewed in color.

### If you use this code in your paper, please cite the following:
```
@ARTICLE{Li_Counterfactual_2024,
  author={Li, Yong and Menglin Liu, Lingjie Lao, Yuanzhi Wang, Zhen Cui},
  journal={Visual Intelligence}, 
  title={Counterfactual Discriminative Micro-Expression Recognition}, 
  year={2024},
  }
```
