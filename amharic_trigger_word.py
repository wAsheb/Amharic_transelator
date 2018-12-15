
# coding: utf-8

# Trigger Word Detection

# Lets get started! Run the following cell to load the package you are going to use.    


# In[1]:

import numpy as np
#from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *
get_ipython().magic('matplotlib inline')


# # 1 - Data synthesis: Creating a speech dataset 


# In[4]:
playSound("./raw_data/activates/1.wav")

x = graph_spectrogram("audio_examples/example_train.wav")


# The graph above represents how active each frequency is (y axis) over a number of time-steps (x axis). 

# The dimension of the output spectrogram depends upon the hyperparameters of the spectrogram software and the length of the input. 
# In this notebook, we will be working with 10 second audio clips as the "standard length" for our training examples. 
# The number of timesteps of the spectrogram will be 5511. You'll see later that the spectrogram will be the input $x$ into the network, and so $T_x = 5511$.
# 

# In[9]:

_, data = wavfile.read("audio_examples/example_train.wav")
print("Time steps in audio recording before spectrogram", data[:,0].shape)
print("Time steps in input after spectrogram", x.shape)


# In[10]:

Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram

# In[11]:

Ty = 1375 # The number of time steps in the output of our model

# In[12]:

# Load audio segments using pydub 
activates, negatives, backgrounds = load_raw_audio()

print("background len: " + str(len(backgrounds[0])))    # Should be 10,000, since it is a 10 sec clip
print("activate[0] len: " + str(len(activates[0])))     # Maybe around 1000, since an "activate" audio clip is usually around 1 sec (but varies a lot)
print("activate[1] len: " + str(len(activates[1])))     # Different "activate" clips can have different lengths 


# **Overlaying positive/negative words on the background**:

# This is another reason for synthesizing the training data: 
# In contrast, if you have 10sec of audio recorded on a microphone, it's quite time consuming for a person to listen to it and mark manually exactly when "activate" finished. 


# In[13]:

def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
    
    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")
    
    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    
    segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)

# In[14]:

# is_overlapping

def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    
    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments
    
    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """
    
    segment_start, segment_end = segment_time
    print(segment_start)
  

    overlap = False
    

    for previous_start, previous_end in previous_segments:
        if (segment_start <= previous_end) and (segment_end >= previous_end):
            
            overlap = True #(segment_time[1] -segment_time[0]) -(previous_end-previous_start)


    return overlap

# In[16]:

def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the 
    audio segment does not overlap with existing segments.
    
    Arguments:
    background -- a 10 second background audio recording.  
    audio_clip -- the audio clip to be inserted/overlaid. 
    previous_segments -- times where audio segments have already been placed
    
    Returns:
    new_background -- the updated background audio
    """
    
    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)
    

    segment_time = get_random_time_segment(segment_ms)
    
    
    while (is_overlapping(segment_time, previous_segments)):
        segment_time = get_random_time_segment(segment_ms)

    
    previous_segments.append(segment_time)
    
    new_background = background.overlay(audio_clip, position = segment_time[0])
    
    return new_background, segment_time


# In[17]:

np.random.seed(5)
audio_clip, segment_time = insert_audio_clip(backgrounds[0], activates[0], [(3790, 4400)])
audio_clip.export("test_amharic.wav", format="wav")
print("Segment Time: ", segment_time)
playSound("test_amharic.wav")



# In[18]:

playSound("audio_examples/insert_reference.wav")

# In[19]:


def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.
    
    
    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms
    
    Returns:
    y -- updated labels
    """
    Ty = y.shape[1]

    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    

    y[0,segment_end_y+1:segment_end_y+51] = 1

    
    return y


# In[20]:

arr1 = insert_ones(np.zeros((1, Ty)), 9700)
plt.plot(insert_ones(arr1, 4251)[0,:])
print("sanity checks:", arr1[0][1333], arr1[0][634], arr1[0][635])


# In[21]:

# create_training_example

def create_training_example(background, activates, negatives):
    """
    Creates a training example with a given background, activates, and negatives.
    
    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"
    
    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """
    
    # Set the random seed
    np.random.seed(18)
    
    # Make background quieter
    background = background - 20

    
    y = np.zeros((1,Ty))

    
    previous_segments = []
    
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    
    
    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end)
 

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    
    for random_negative in random_negatives:
        # Insert the audio clip on the background 
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
    
    
    # Standardize the volume of the audio clip 
    background = match_target_amplitude(background, -20.0)

    # Export new training example 
    file_handle = background.export("train" + ".wav", format="wav")
    print("File (train.wav) was saved in your directory.")
    
    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = graph_spectrogram("train.wav")
    
    return x, y


# In[22]:

x, y = create_training_example(backgrounds[0], activates, negatives)


# Now you can listen to the training example you created and compare it to the spectrogram generated above.

# In[23]:

playSound("train.wav")


# In[24]:

playSound("audio_examples/train_reference.wav")

# Finally, you can plot the associated labels for the generated training example.

# In[25]:

plt.plot(y[0])


# ## 1.4 - Full training set
# 
# You've now implemented the code needed to generate a single training example. We used this process to generate a large training set. To save time, we've already generated a set of training examples. 

# In[26]:

# Load preprocessed training examples
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")


# ## 1.5 - Development set
# 
# To test our model, we recorded a development set of 25 examples. While our training data is synthesized, we want to create a development set using the same distribution as the real inputs. Thus, we recorded 25 10-second audio clips of people saying "activate" and other random words, and labeled them by hand. This follows the principle described in Course 3 that we should create the dev set to be as similar as possible to the test set distribution; that's why our dev set uses real rather than synthesized audio. 
# 

# In[27]:

# Load preprocessed dev set examples
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")


# # 2 - Model
# 
# Now that you've built a dataset, lets write and train a trigger word detection model! 
# 
# The model will use 1-D convolutional layers, GRU layers, and dense layers. Let's load the packages that will allow you to use these layers in Keras. This might take a minute to load. 

# In[28]:

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam


# ## 2.1 - Build the model
# 
# Here is the architecture we will use. Take some time to look over the model and see if it makes sense. 

# Note that we use a uni-directional RNN rather than a bi-directional RNN. 
# This is really important for trigger word detection, since we want to be able to detect the trigger word almost immediately after it is said. If we used a bi-directional RNN, we would have to wait for the whole 10sec of audio to be recorded before we could tell if "activate" was said in the first second of the audio clip.  

# Implementing the model can be done in four steps:

# In[29]:

def model(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    
    X_input = Input(shape = input_shape)
    

    X = Conv1D(filters=196,kernel_size=15,strides=4)(X_input)                                 # CONV1D
    X = BatchNormalization()(X)                # Batch normalization
    X = Activation('relu')(X)                                 # ReLu activation
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)

   
    X = GRU(units = 128, return_sequences = True)(X)             # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization
    
    
    X = GRU(units = 128, return_sequences = True)(X)          # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                      # dropout (use 0.8)
    X = BatchNormalization()(X)                # Batch normalization
    X = Dropout(0.8)(X)                      # dropout (use 0.8)
    
    
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)

    

    model = Model(inputs = X_input, outputs = X)
    
    return model  


# In[30]:

model = model(input_shape = (Tx, n_freq))

# print model summary

model.summary()

# Trigger word detection takes a long time to train. To save time, we've already trained a model for about 3 hours on a GPU using the architecture you built above, and a large training set of about 4000 examples. Let's load the model. 

# In[32]:

model = load_model('./models/tr_model.h5')


# You can train the model further, using the Adam optimizer and binary cross entropy loss, as follows. 
# In[33]:

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])


# In[34]:

model.fit(X, Y, batch_size = 5, epochs=1)


# Finally, let's see how your model performs on the dev set.

loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)

# 1. Compute the spectrogram for the audio file
# 2. Use `np.swap` and `np.expand_dims` to reshape your input to size (1, Tx, n_freqs)
# 5. Use forward propagation on your model to compute the prediction at each output step


# In[36]:

def detect_triggerword(filename):
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename)
    # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    return predictions


# 1. Loop over the predicted probabilities at each output step
# 2. When the prediction is larger than the threshold and more than 75 consecutive time steps have passed, insert a "chime" sound onto the original audio clip
# 
# Use this code to convert from the 1,375 step discretization to the 10,000 step discretization and insert a "chime" using pydub:

# In[37]:

chime_file = "./audio_examples/random_amhric2.wav"
def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # Step 1: Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Step 2: Loop over the output steps in the y
    for i in range(Ty):
        # Step 3: Increment consecutive output steps
        consecutive_timesteps += 1
        # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            # Step 5: Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
            # Step 6: Reset consecutive output steps to 0
            consecutive_timesteps = 0
        
    audio_clip.export("chime_output.wav", format='wav')

# Let's explore how our model performs on two unseen audio clips from the development set. Lets first listen to the two dev set clips. 

# In[38]:

playSound("./raw_data/dev/1.wav")

# In[39]:

playSound("./raw_data/dev/2.wav")

# Now lets run the model on these audio clips and see if it adds a chime after "activate"!

# In[40]:

filename = "./raw_data/dev/1.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, prediction, 0.5)
playSound("./test_amharic2.wav")


# In[41]:

filename  = "./raw_data/dev/2.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, prediction, 0.5)
playSound("./test_amharic.wav")



# Record a 10 second audio clip of you saying the word "activate" and other random words, 
# Be sure to upload the audio as a wav file.If your audio recording is not 10 seconds, the code below will either trim or pad it as needed to make it 10 seconds. 
# 

# In[42]:

# Preprocess the audio to the correct format
def preprocess_audio(filename):
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    # Set frame rate to 44100
    segment = segment.set_frame_rate(44100)
    # Export as wav
    segment.export(filename, format='wav')


# Once you've uploaded your audio file, put the path to your file in the variable below.

# In[43]:

your_filename = "audio_examples/random_amharic.wav"


# In[44]:

preprocess_audio(your_filename)
playSound(your_filename) # listen to the audio you uploaded 


# Finally, use the model to predict when you say activate in the 10 second audio clip, and trigger a chime. 
# If beeps are not being added appropriately, try to adjust the chime_threshold.

# In[45]:

chime_threshold = 0.5
prediction = detect_triggerword(your_filename)
chime_on_activate(your_filename, prediction, chime_threshold)
playSound("./amarigna_mukera.wav")

