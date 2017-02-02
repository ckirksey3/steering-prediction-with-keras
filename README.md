# CarND-BehavioralCloning
Behavioral Cloning with Self-Driving Cars (Udacity Nanodegree)

provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

Is the creation of the training dataset and training process documented?
how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset should be included.

## Model Architecture Design
My design process began by reading the [End to End Deep Learning with Self-Driving Cars paper by NVIDIA](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

In order to improve the performance of my system, I built image normalization into the network itself using a Lambda layer, allowing normalization to be done with a GPU. Then I use five convolutional layers to extract the relevant features from the image. For these layers, I use ReLU activation since it's wide range (relative to Sigmoid) makes it suited for the steering problem where we need to calculate an angle rather than just a probability. 

Next I use three fully connected layers and an output layer to generate the final steering angle from the output of the feature extraction layers. Initially this model had difficulty generalizing the input so I added two dropout layers to combat overfitting.

The model uses an Adam optimizer to minimize the means squared error between it's output and the recorded steering angle from the simulator that corresponds to the image of the road at that time. 

## Training Data
The bulk of my time for this project was spent on data processing. Initially I recorded some of my own data, but vastly underestimated how much data I needed. Any training that I did resulted in my model outputting a constant steering angle since I didn't have enough situations for the model to recognize patterns. The model was basically just averaging the input angles.

Eventually Udacity released their own larger dataset so I trained on that, but my model still gave a constant output.
