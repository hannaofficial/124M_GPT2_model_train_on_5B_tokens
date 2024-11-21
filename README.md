# GPT3_trained_on_5B_tokens_code

* 124 Million parameter GPT3 model <br/>
* This GPT3 model is build from scratch <br/>
* Dataset used: Fineweb edu 10B tokens <br/>

* In GPT3 they used sparse transformer because of computation efficiency reduced computation from (N^2 to N.k)
Since my model is relatively small with 124 million parameters compared to GPT-3's largest variant with 175 billion parameters,<br/>
* I have opted for using a full attention mechanism. This approach helps simplify the initial stages <br/>
of training and avoids the complexity that comes with more advanced mechanisms.
* For some initial training step they(in GPT3 paper) use batchsize of 32k then increase to 0.5M batch size after some training step in 124M (I didn't use this)


> *  I haven't commented the code neatly, but I hope you'll be able to understand it.
     In the future, I will improve the comments to make the code clearer. <br/>
     
> * In future I will try to optimized this model more.



```
1. In this model I have used decoder transformer model slightly different from original paper on transformer 'Attention is all you need'.<br/>
2. Also there is some change in code compare to karpathy. <br/>
4. I use hellaswag benchmark to evaluate model commomsense.<br/>
4. I was using google colab pro for accesing gpu at first but it's computing unit complete during experimenting with the code.<br/>
 Therefore at last I was not able to completely use 10B sample tokens for training.<br/>
5. Last time few iteration you see here is done by T4 gpu in google colab.

In Future I will update on the model code readibility
```
