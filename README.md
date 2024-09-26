
# 🎭 Project: Build a Facial Keypoint Detection System 🎯

## Welcome to My Facial Keypoint Detection Adventure! 🕵️‍♂️🎉

In this project, I embarked on a thrilling journey to teach a computer how to detect the unique features on a face. We’re talking eyes, nose, mouth, and everything in between (no, not that 🥸). The goal: build a magical model that can look at any image and predict the location of **68 key facial points**! Why 68, you ask? Because the face has more landmarks than a road trip map! 😁

Ready to dive in and witness the magic of computer vision and deep learning in action? Let’s go! 🚀✨

---

## Project Introduction 📚

### Why Do We Care About Keypoints? 🤔

Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many cool applications. Think **facial tracking** (no more hiding from the camera! 😜), **pose recognition** (strike a pose! 🕺), **facial filters** (because everyone needs dog ears 🐶), and even **emotion recognition** (yes, the computer knows you’re sad about all those bugs). By the end of this project, our model should be able to look at any image, detect faces, and predict the locations of these keypoints like a pro! 🥇

![screen-shot-2018-04-10-at-8 24 14-pm](https://github.com/user-attachments/assets/1f74e0fc-25ad-4ddb-9b57-698e73af95a7)

---

## Solver Functionality 🛠️

| Criteria                                            | Submission Requirements                                                                                                                                         |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🕵️‍♂️ Define a CNN in models.py                    | **(AUTOGRADED)** Define a convolutional neural network with at least one convolutional layer. The network should take in a grayscale, square image, just like a 90s TV. 📺 |
| 🧠 Train the CNN with defined loss & optimizer       | **(AUTOGRADED)** Train your CNN with a proper loss function (for those pesky errors) and an optimizer (like a gym trainer for your model 💪).                          |
| 🔎 Detect faces using Haar Cascades                  | **(AUTOGRADED)** Use a Haar cascade detector to find faces. Think of it as the GPS for finding faces in images. 🧭                                                |
| 🎯 Predict and display keypoints for each face       | **(AUTOGRADED)** After finding a face, use your trained model to predict keypoints and show them off. Like a dot-to-dot puzzle, but for faces. 🔴🔵                    |

---

## How I Survived the Project 🧗‍♂️

### Step 1: The Beginning of the End 🏁

First things first, I opened up the project instructions and wondered if I had taken a wrong turn into a deep learning labyrinth. But fear not! I powered through by setting up my environment and downloading all the required files. Little did I know, this was just the calm before the storm. 🌪️

1. **Clone the Repository**: Because who doesn't like starting with a clean slate? 📜
    ```bash
    git clone https://github.com/udacity/computer-vision.git
    ```

2. **Navigate to the Project Folder**: Like a detective, I entered the heart of the project:
    ```bash
    cd Projects/2_Facial_Keypoints
    ```

3. **Create & Activate the Conda Environment**: Time to suit up! 🧙‍♂️
    ```bash
    conda create --name keypoints_env python=3.6
    conda activate keypoints_env
    ```

4. **Install the Requirements**: All set to tackle the project (hopefully). 🤞
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Loading and Visualizing the Data 📊

Like any good scientist, I started by looking at the data. We’re talking pictures of faces, lots of them. The goal was to get familiar with what we’re working with. So, I loaded up the images and keypoints and plotted them to see if my model would eventually have a fighting chance. 🖼️

- **Expectation**: Perfectly plotted points over every face.
- **Reality**: Points in places that made it look like a Picasso painting. 🎨

### Step 3: Building the Neural Network 🏗️🧠

This is where the rubber meets the road. I had to create a Convolutional Neural Network (CNN) in `models.py` that could handle all those facial keypoints.

```python
class KeypointsCNN(nn.Module):
    def __init__(self):
        super(KeypointsCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*53*53, 68*2)  # 68 keypoints with (x, y) coordinates
```

- **What I Thought**: This model will be a masterpiece. 🎨
- **Reality**: Why is it predicting keypoints on the wall behind the face? 😳

### Step 4: Training the Beast 🔥

This part involved a lot of waiting, hoping, and crying (mostly crying). I trained the model using a loss function and an optimizer, both of which were supposed to reduce my errors. Sometimes they did, and sometimes they just mocked me. 🤖

```python
criterion = nn.MSELoss()  # Because nothing says 'loss' like Mean Squared Error
optimizer = optim.Adam(net.parameters(), lr=0.001)  # Adam, not to be confused with the first man on Earth
```

### Step 5: Detecting Faces with Haar Cascades 🎩

Haar Cascades are like metal detectors, but for faces. They find where the faces are in an image so we know where to zoom in. I implemented this detector, and voilà, faces were found!

```python
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(image, 1.2, 2)
```

- **Expectation**: Finding faces everywhere!
- **Reality**: False positives on furniture. Is that a sofa or a face? 🤔

### Step 6: Predicting Keypoints & Creating Masterpieces 🎨

With faces detected and preprocessed, it was time to unleash my trained model. After a few adjustments (and a lot of retries), the model started predicting keypoints on the detected faces!

- **Expectation**: Perfect keypoints on every face.
- **Reality**: Some looked more like alien invasion plans. 👽

### Step 7: Adding Fun Filters 🤡

Why stop at keypoints when you can add some fun filters? I added glasses, hats, and even a mustache to spice things up. Now my model wasn’t just predicting keypoints; it was predicting coolness. 😎

```python
def add_mustache(image, keypoints):
    # Place the mustache at the right keypoints
    return image_with_mustache
```

---

## How to Run This Masterpiece 👩‍💻

Wanna see this bad boy in action? Here’s how to do it:

1. Clone the repo:
   ```bash
   git clone <this-repo-url>
   ```
   
2. Activate the environment:
   ```bash
   conda activate keypoints_env
   ```

3. Navigate to the project folder:
   ```bash
   cd Projects/2_Facial_Keypoints
   ```

4. Start the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

5. Open each notebook and execute all cells in the proper order. Watch the magic unfold! 🔮

---

## Reviewer’s Note ✨

### You've done a fantastic job completing the Facial Keypoints Detection Project! 😊

A special note from my awesome Udacity reviewer:

> **Facial Keypoints Detection is a well-known [machine learning challenge](https://www.kaggle.com/c/facial-keypoints-detection/overview). If you want to improve this very model to allow it to work well in extreme conditions like bad lighting, bad head orientation (add appropriate PyTorch transformations like ColorJitter, HorizontalFlip, Rotation and more - [see this post](https://www.analyticsvidhya.com/blog/2019/12/image-augmentation-deep-learning-pytorch/) for more details), etc., the best thing you can do is to simply follow [NaimishNet](https://arxiv.org/abs/1710.00977) implementation details with some tweaks (optimizer, learning rate, batch size, etc) as per the latest improvements and your machine requirements. But for production-level performance, you can always use pre-trained models for better performance, say [Dlib library](https://github.com/davisking/dlib) provides real-time facial landmarks seamlessly. You can find a [tutorial here](https://pyimagesearch.com/2018/04/02/faster-facial-landmark-detector-with-dlib/).

---

## Conclusion 🏁

Building this facial keypoint detection system was a rollercoaster of emotions. From face-detecting Haar Cascades to keypoint-predicting CNNs, I’ve gone through it all. But now, the model is up and running, predicting keypoints like a pro (most of the time). 🎉

##Just remember — even the mightiest programmers start with baby steps (and occasional tantrums). And hey, sometimes those baby steps mean seeking help from fellow coders, dissecting their code, and piecing together solutions. So yes, I’ve had my fair share of peeking into others' projects, learning from their work, and figuring out how things tick. It’s all part of the journey. So, maff kar do muja if I borrowed an idea or two along the way—because, in the end, it’s about growing and improving. 😅

If you want to add your own touch, feel free to clone, fork, or just use it to impress your friends. Just remember, with great code comes great responsibility! 🕷️

---

## License ⚖️

This project is released under the “Do What You Want With This, But Don’t Blame Me for Bugs” License. Feel free to share, modify, and experiment, but don’t forget to add your own flair to it. After all, who doesn’t love a model that can put a mustache on anyone? 😉

---
