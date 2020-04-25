Summary of all the Analysis of the 6 iterations

ITERATION - 1
=============
Target:
This notebook takes the assignment submitted for session 4. This assignment achieved 99.41% and 99.43% accuracy in the 18th and 19th epoch respectively (less than 20 epochs). This notebook however doesn't use
a) dynamic learning rate, this uses a static learning rate of 0.002.
b) 1X1 Kernel
c) global average pooling
d) Image augmentation / rotation
e) It doesn't print the training accuracy as well, and hence inability to identify overfitting.
As a part of the iteration 1. I would like to reduce the learning rate at different intervals to see if it can reduce the number of epochs to reach consistent 99.4% accuracy from 18/19 to 13/14. I would also like to include the training accuracy to see overfitting. I wouldn't focus on reducing the parameters for now. I will try and include them in subsequent iterations

Result:
Best Training Accuracy achieved in Epoch 14: 99.19% Best Test Accuracy is achieved in Epoch 10: 99.45% I tried with various combinations of gamma and stepsize, the intentions was to have high gamma values with low step sizes so that the learning rate decreases in a smooth manner rather than the step changes. The results were very similar for multiple combinations. The step size of 2 and gamma of 0.45 is selected.

Analysis:
The model is underfitting in the beginning by a big margin, which decreases as the number of epoch increases when the training accuracy catches up and there is increases in testing accuracy as well. The target of 99.4% consistently by 15 epochs is achieved in the testing dataset as well


ITERATION - 2
=============
Target
In the last notebook 99.4%+ was achieved 4 times between 10th and 15th epoch. Now the aim of this iteration is to reduce the parameters to levels below 10000. We will do the same by reducing the kernel size across the layers with minimal impact on accuracy. For the purpose of the same, I'm planning to include additional convolutional layers in the various blocks

Result
Best Training Accuracy achieved in Epoch 14: 97.97%
Best Test Accuracy is achieved in Epoch 14: 99.12%

Analysis
The desired goal of reducing the parameters is achieved by reducing the kernel sizes and a slight increase in layers. This has reduced the accuracy as expected. Now the focus can be in increasing the accuracy by introducing the image augmentation.

ITERATION - 3
=============
Target
In the last notebook the parameters were reduced but the accuracy was at 99.12%. One of the ways of increasing the accuracy would be image augmentation so that images with slight rotation might be scored properly in testing as well.

Result
Best Training Accuracy achieved in Epoch 15: 97.86%
Best Test Accuracy is achieved in Epoch : 99.19%

Analysis
The accuracy had increased marginally, but it has still not reached the required 99.40% even once

ITERATION - 4
=============
Target
In the last notebook the parameters were reduced but even after adding image agumentation the accuracy was only at 99.19%. From the images it doesn't look like there is much to be filtered, so it looks like Global Average Pooling as a way of increasing reseptive field might be better than Max pooling.

Result
Best Training Accuracy achieved in Epoch 14 :98.00%
Best Test Accuracy is achieved in Epoch :99.28%

Analysis
The accuracy again had increased marginally, but it has still not reached the required 99.40% even once. Thinking if I should rearrange the architecture to have only one global average pooling

ITERATION - 5
=============
Target
Planning to decrease the number of layers there by giving adequate channels in each of the available layers. Also introducing the GAP layer towards the end and introducing a 1X1 Kernel after that will make the model better

Result
Best Training Accuracy achieved in Epoch 14 :98.88%
Best Test Accuracy is achieved in Epoch :99.38%

Analysis
The accuracy had increased however after epoch 12 had been alternating, there by indicating the need to reduce the learning rate more agressively than the current gamma of 0.45 every step 2

ITERATION - 6
=============
Target
Hoping to have a more agressive reduction in learning rate so that towards the end the model stablizes faster. Hence reduced gamma to 0.2 and retained step size as 2 as in previous models. Also the training is consistently lower than the test, so probably the dropout of 15% is rather high and not allowing the model to achieve the desired accuracy so reducing the dropout to 10%

Result
Best Training Accuracy achieved in Epoch 15 :99.01%          
Best Test Accuracy is achieved in Epoch 15:99.43%    

Analysis
The accuracy had increased and had been 99.39%, 99.41% and 99.43% in the last three epochs, thereby suggesting that the dropout was intially very high, the addition of GAP and 1X1 kernel at the end and adjusting learning rates were important decisions