# word_recognition_algorithms

**packages** :
<br>numpy
<br>pandas
<br>sklearn
<br>tensorflow
<br>keras
<br>math
<br>sys
<br>time
<br>operator
<br>copy
<br>functools
  
#### Module Analysis in the Algorithm

<br>The whole word recognition algorithm is consisting of two parts: data process part and model training and prediction part. The data process part is very important not only because it is the preparation part to get the needed and format training data and test data but also because it included different processing methods for different tasks. There are two tasks in the word recognition algorithm; one is simple bi-classification task which is meant to predict whether there is one or more clean words in a sequence, another is word position recognition whose task is to predict the accurate position of the clean word in a sequence. The data formats are not the same in the two tasks, so the data process part is well worth description. When it comes to model training and prediction part, there are two training and prediction models, one is Hidden Markov Model, another is Neural Network Model. The HMM model is a basic statistical method while the NN model is the proposed deep learning method in this paper. The NN model includes the construction of the model, the parameters setting, and visualization etc.

<br>The data process part has two modules, one is basic text preparation which is to get the clean words from raw English text on the Internet, and another is data preparation which is to get the needed format and according labels respectively in bi-classification task and word position recognition task.


#### Definitions of the Data Process Algorithm
<br>As for the ‘Basic Text Preparation’ part, the main task is to get the clean words from novel contents. The raw input and the clean words after processing are illustrated as following:
<br>**raw input**: ‘let us for the present say that his name was Greene’
<br>**clean words**: ‘let’‘present’‘say’‘name’‘greene’
<br>As for the ‘Data Preparation’ part for bi-classification task, the main goal is to take the clean words as the input and get the random sequence of length 30 with the classification label 0 or 1. The input and the output formats are generated as following:
<br>**Input clean data**: ‘policeman’&nbsp;‘beat’
<br>**Output format**: sequence  &nbsp;&nbsp;&nbsp;label
<br>policemanxarcwwhrwflzvukenkxoh &nbsp;&nbsp;&nbsp;  1 
<br>juypbcsnfbziaudmufkmyirkpmrzqm &nbsp;&nbsp;&nbsp;   0 
<br>cbmmktfcufzmrtwsmxklgyzfbeattp &nbsp;&nbsp;&nbsp;  1 
<br>fjfecinqawsjqrcsmlxygazedfunql &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0 
<br>hmmnvaifkfiplggyyfdyubvzgdexzz &nbsp;&nbsp;&nbsp;  0 
<br>hmmnvaifkfiplggyyfdyubvzgdexzz &nbsp;&nbsp;&nbsp;  1

