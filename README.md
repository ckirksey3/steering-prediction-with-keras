# CarND-BehavioralCloning
For this project, I created a convolutional neural network using the Keras Sequential model to predict the appropriate steering angle for staying on the track in a driving simulator. The model was trained on images and steering angles taken from driving in the simulator with some simple image preprocessing and the desampling of data points with near-zero steering angles to emphasize turning.

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

Although I originally converted all images to grayscale, I later removed this step as my model was having difficulty distinguishing the paved road from the dirt shortcut when the lane markers were removed. Finally, I resized the image to be smaller and square for quicker processing by the CNN.

### Angle Buckets
Despite all of this work to evenly distribute the angles, my model was still getting confused by sharp angles. There wasn't a sufficient frequency of sharp turns for the model to be evenly trained. While its bias to near-zero had been sufficiently reduced, it still had a bias toward small turns. I solved this by building a histogram of all of the steering angles in the training set, creating buckets for each of those ranges that include all the data points with steering angles in that range, and sampling evenly from each of those buckets. This was the step that finally pushed me over the edge and enabled my model to drive the entire course.

The inspiration for this improvement came through some long discussions with (@manavkataria)[https://github.com/manavkataria] about the impact of poor angle distribution. He had the original idea to store them in a set of buckets generated by a histogram.

![Cropped Screenshot](data_analysis/compressed.png?raw=true "Preprocessing")

## Result Video
It turns!

[![DRIVING VIDEO](https://img.youtube.com/vi/vm2fFU0PTaU/0.jpg)](https://www.youtube.com/watch?v=vm2fFU0PTaU&feature=youtu.be)
