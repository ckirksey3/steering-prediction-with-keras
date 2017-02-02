# CarND-BehavioralCloning
Behavioral Cloning with Self-Driving Cars (Udacity Nanodegree)
fdsfaasdf
provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

Is the creation of the training dataset and training process documented?
how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset should be includedf

## Model Architecture Design
My design process began by reading the [End to End Deep Learning with Self-Driving Cars paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) by NVIDIA. 

In order to improve the performance of my system, I built image normalization into the network itself using a Lambda layer, allowing normalization to be done with a GPU. Then I use five convolutional layers to extract the relevant features from the image. For these layers, I use ReLU activation since it's wide range (relative to Sigmoid) makes it suited for the steering problem where we need to calculate an angle rather than just a probability. 
ff
Next I use three fully connected layers and an output layer to generate the final steering angle from the output of the feature extraction layers. Initially this model had difficulty generalizing the input so I added two dropout layers to combat overfitting.

The model uses an Adam optimizer to minimize the means squared error between it's output and the recorded steering angle from the simulator that corresponds to the image of the road at that time. 

![Model Graph](data_analysis/model_graph.png?raw=true "NVIDIA Model Architecture")

## Training Data
The bulk of my time for this project was spent on data processing. Initially I recorded some of my own data, but vastly underestimated how much data I needed. Any training that I did resulted in my model outputting a constant steering angle since I didn't have enough situations for the model to recognize patterns. The model was basically just averaging the input angles.

Eventually Udacity released their own larger dataset so I trained on that, but my model still gave a constant output. After talking this over with several classmates, I realized that this was largely caused by the overwhelming presence of near 0 degree steering angles in the data set. I decided to do some analysis on my dataset to see how different techniques could impact this issue.

### Unprocessed Angle Distribution
![Original](data_analysis/original_angle_dist.png?raw=true "Data Analysis")
After graphing a histogram of the angle distribution in my dataset, it was immediately obvious that there were far too many near-zero values in my dataset. My model was just giving a near-zero constant output for every image since this strategy would actually help it mimize the means squared error of the output as nearly every recorded angle was near zero.

### Axis Flipping
![Axis Flipping](data_analysis/original_with_axis_flip.png?raw=true "Data Analysis")
My first image processing technique was to flip the image along the y-axis and then reverse the steering angle. This doubled my training set and ensured an even distribution of positive and negative steering angles. 

### Camera Switching
![Camera Switching](data_analysis/camera_switching_added.png?raw=true "Data Analysis")
Prior to this, I had been using the center camera images for all of my training. In this step, I began randomly selecting the left, right, or center image and adding a small adjustment to the steering angle for any non-center image. This helped reduce the number of absolute zero angles and provide more variability between epochs of training.

### Zero Penalizing
![Zero Penalizing](data_analysis/everything_with_zero_penalizing.png?raw=true "Data Analysis")
The primary processing technique that helped me get a working model was to progressively reduce the number of training points with near zero steering angles. Mohan Karthik explained this process in his [Medium post](https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff#.304ci98i2) about the project. 

### Image Preprocessing
In addition to techniques for improving the angle distribution, I also experimented with different preprocessing techniques to aid in feature extraction. One that seemed to be particularly helpful was cropping some of the top and bottom of the image to reduce the visual noise and focus the model on the road curvature
![Cropped Screenshot](data_analysis/cropped.jpg?raw=true "Preprocessing")
