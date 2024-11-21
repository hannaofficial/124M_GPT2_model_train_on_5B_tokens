# GPT2_train_on_5B_tokens

124 Million parameter GPT3 model <br/>
This GPT3 model is build from scratch <br/>
Dataset used: Fineweb edu 10B tokens <br/>

In GPT3 they used sparse transformer because of computation efficiency reduced computation from (N^2 to N.k)
Since my model is relatively small with 124 million parameters compared to GPT-3's largest variant with 175 billion parameters,<br/>
I have opted for using a full attention mechanism. This approach helps simplify the initial stages <br/>
of training and avoids the complexity that comes with more advanced mechanisms.

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
In Future I will update on the model code readibility
```
