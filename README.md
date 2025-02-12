
# Jim's Machine Learning Notebook

### My Python Environment

Python is king in the ML world.

* Apple MacOS comes with a python installed in `/usr/local/bin`. As of 2/8/2025 that is version `3.9.6`. I override
  this by putting `/opt/homebrew/bin` earlier in the path than `/usr/local/bin`.
* Python installed via Homebrew 'brew' command: Latest available as of 2/7/2025: `3.13.1`
* Installed `/opt/homebrew/Cellar/python@3.13/3.13.1/Frameworks/Python.framework/Versions/3.13/bin/python3.13`
* Homebrew creates many sym links in `/opt/homebrew/bin/python*`

### Special requirements for building and running `tensorflow` as of `tensorflow` version `2.18.0`

#### Building `tensorflow`

* **Python**: tensorflow has a Python version upper limit of `3.12`
* The tensorflow build calls into a package that looks for the system environment variable
  `HERMETIC_PYTHON_VERSION`. If it's not set, it tries to determine a best python to use but that fail.
  So, I set it to '`HERMETIC_PYTHON_VERSION=3.12`' in .zshrc.
* **Bazel** (build tool): version upper limit: `6.5.0` instead of latest `7.x`.
    * Installed 6.5 in`/usr/local/bin/bazel`.
    * Point your PyCharm or CLion IDE settings to that bazel binary when building tensorflow.

#### Running `tensorflow`

* **numpy**: A Python core package. Some code in `tensorflow` cannot use the latest `numpy-2.0.2` because `tensorflow`
  references a symbol, `numpy.bool`, that was removed in numpy-2.x in favor of the generic type `bool`. Therefore, i
  have pinned `numpy` to `1.23.5` in `requirements.txt`.


### Machine-learning Engines and Libraries
Engines with APIs (usually Python). Allows you to train a model for practical use.

Provider: Google tensorflow, DeepMind, Google Brain, JAX
Type: Engine and Products
Notes: TensorFlow is king of real-world applications, not counting OpenAI's
for-pay model.

Provider: PyTorch  
Engine: PyTorch  
Notes:

Provider: scikit  
Engine: SciKit  
Notes:

Provider: OpenAI
Engine: OpenAI  
Notes: Many 3rd-party models. They've gone totally for-pay. No longer 
available for tinkering. 

Provider: Keras
Type: Model training 
Notes: Provides layers (Densee, Conv2D, LSTM) and pre-built models (ResNet, VGG).
Associated with TensorFlow.

Provider: Pandas
Type: Dataset building

### Machine-learning User-facing Products

#### Provider: Mistral
User-Facing Product: Data Row 3 Col 2  
Engine: Data Row 3 Col 3  
Notes:

#### Provider: Cohere  
User-Facing Product:  
Engine:  
Website: [cohere.com]()  
Notes:

#### Provider: Hugging Face
User-Facing Product:  
Engine:  PyTorch, TensorFlow, JAX deep learning
Website: [huggingface.co]()  
Notes:

#### Provider: Inflection AI
User-Facing Product: Pi (a chatbot, personal assistant)  
Engine:  
Website: [inflection.ai]()  
Notes:

#### Provider: Perplexity AI
User-Facing Product: Pi (a chatbot, personal assistant)  
Engine:  
Website: [www.perplexity.ai]()  
Notes:

### General Machine-Learning Resources

#### Resource: Kaggle
[kaggle.com]()  
An AI & ML community started by Google.  
Dataset library.

#### Microsoft: COCO
Dataset collection.

Labelbox - dataset annotation
Roboflow - dataset annotation

### Test Cases

#### Test Case : A predictive n-dimensional data point model

Description: Load a 2D x, y datapoint set that is roughly quadratic and have the model generate y 
values given x.

variants:
* 2 points should produce a simple linear essentially perfect model
* a scatter of many points with linear model should fit the best line
* 3 points to define a quadratic with 2nd order model


#### Test Case : Generate full 2K bitmaps from textual descriptions

Train: Associate text descriptions with 2D images 
Output: 2D bitmaps that fit a query. The model assembles images 
wherein keywords in the query are matched to 

#### Test Case : Local Document knowledge store
A model takes in a bunch of general information in the form of source and word documents, emails.
The model can then 'search' for concepts, present them by data, author,
source document.

#### Test Case : Voice and Music generator


