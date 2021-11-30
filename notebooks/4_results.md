# Results

In the notebooks 3_1 and 3_2 we have seen 2 models that satisfy the problem.

The first one treats the problem as a 3 class classification task,   
The second one as a regression task where we considered the 3 final classes as a discretization of the regression score.

Both of the models performed well with about 85% accuracy each. 
The classification achieved slightly better accuracy

Based on the confsion matrix, I would choose **the first model** because its accuracy on neutral texts is better and the one on both positive and negative is the same.

We also need to take into consideration the data we had, as we have seen in the previous notebooks, we don't have a lot a labalized data for a multilingual model and some of the labels are from my point of view wrong. I think we could have obtained better results with a much larger train dataset.

Finally the performance in term of throughput of the model is very important especially in an environment where we are dealing with a large amount of data.