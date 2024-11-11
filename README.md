# GPT2_train_on_5B_tokens

124 Million parameter GPT2 model <br/>
This GPT2 model is build from scratch <br/>
Dataset used: Fineweb edu 10B tokens <br/>

I haven't done commenting neatly I hope you will able to understand the code<br/>
In future I will make nice commenting so that you willl able to understand the code full

In this model I have used decoder transformer model slightly different from original paper.<br/>
Also there is some change in code compare to karpathy. <br/>
In future I will try to optimized this model more.<br/>
I use hellaswag benchmark to evaluate model commomsense.<br/>
I was using google colab pro for accesing gpu at first but it's computing unit complete during experimenting with the code.<br/>
Therefore at last I was not able to completely use 10B sample tokens for training.<br/>
Last time few iteration you see here is done by T4 gpu in google colab.
```
Future I will update on the model readibility
```
