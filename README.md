# Overview

These scripts handle the data pre-processing, training, and execution of a Convolutional Neural Network based classifier
for thermal vision.

The output is a TensorFlow model that can identify thermal video clips of animals

# Scripts

### load.py
Processes CPTV clips and metadata, extracts the tagged frames (supplied by the metadata) and adds them to the hdf5 database.

The tracking algorithm tries to distinguish between animal tracks and false positives, but is not 100% reliable.  For this reason the output of the tracking algorithm should be checked by hand.


### build.py
Creates training, validation and testing datasets from database of clips & tracks.
Datasets contain frames and segments ( e.g. 45 frames).

Frames (important frames) are calculated by choosing frames with a mass ( the count of pixels that have been deemed an object by track extraction) between the lower and upper quartiles of a tracks mass distribution.

Frames are also checked to see if they are noisy frames.
This attempts to remove the following types of noisy frames:
- Frames where it has bad tracking e.g. in the middle of an animal
- Frames where only part of the animal is shown (maybe leaving the cameras field of vision)
- Frames where nothing is there but noise

Segments are calculated either by (depending on config):

Choosing random permutations of the important frames.
- Number of Segments (# of important frames - segment duration) // 9 segments are selected

or by choosing segment duration consecutive frames whose mass is above a certain amount        
- Number of Segments up to (# of frames - segment duration) // segment-frame-spacing

Datasets are split by camera and location ( to try and remove any bias that may occur from using a camera in multiple sets).

Some labels have low amounts of data so a single camera is split into 2 cameras e.g. Wallabies and Leoparidaes

### train.py
Trains a neural net using a provided test / train / validation dataset.

### classify.py
Uses a pre-trained model to identifying and classifying any animals in a CPTV file.

### evaluate.py
Evaluates the performance of a classify.py run and generates reports.

# Setup 

Install the following prerequisites (tested with Ubuntu 18.0 and Python Python 3.6.9)
`apt-get install -y tzdata git python3 python3-dev python3-venv libcairo2-dev build-essential libgirepository1.0-dev libjpeg-dev python-cairo libhdf5-dev`
1. Create a virtual environment in python3 and install the necessary prerequisites </br>
`pip install -r requirements.txt`

2. Copy the classifier_Template.yaml to classifier.yaml and then edit this file with your own settings.   You will need to set up the paths for it work on your system. (Note: Currently these settings only apply to classify.py and extract.py)

3. Optionally install GPU support for tensorflow (note this requires additional [setup](
https://www.tensorflow.org/install/gpu))</br>
`pip install tensorflow-gpu`

4. MPEG4 output requires FFMPEG to be installed which can be found [here](https://www.ffmpeg.org/) On linux `apt-get install ffmpeg`.  On windows the installation path will need to be added to the system path.


5. Create a classifier configuration

Copy `classifier_TEMPLATE.yaml` to `classifier.yaml`. Edit.

6. Please view [INSTRUCTIONS.md](INSTRUCTIONS.md) for further instructions.
