# Jim's Machine Learning Notebook

Test of git push from clean env
And a second test change from jims-ml-sandbox-new

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

### Tools for ML

For **general-purpose dataset exploration** (without extra coding), start with **PandasGUI** or **D-Tale** for
CSVs/tables.

If **TensorFlow-specific datasets** need exploration, use **Facet** by Google for visual analysis.

Consider **Dash/Plotly** if you want to create a custom GUI with interactive views.

### Core Python Libs for ML

`numpy` is Python's core math library. Includes operations against Python `ndarray`s.

`matplotlib` A fully OO API for graphical visualization of data. Produces line-graphs, bar charts, scatter plots,
2D images.

`pyplot` The procedural counterpart to `matplotlib`. Although `matplotlib` is recommended for any reusable code
(as opposed to one-off scripts), some high-level procedures simplify the more complex OO code.

`pyarrow` is a powerful Python library developed as part of the Apache Arrow project. Apache Arrow is an open-source
framework designed to provide a high-performance standard for working with in-memory columnar data, which is useful for
big data processing. `pyarrow` is the Python binding for Apache Arrow.

### ML Engines and High-Level ML Libraries

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
Notes: Provides layers (Dense, Conv2D, LSTM) and pre-built models (ResNet, VGG).
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

### General Code Approach

1. Build datasets for training and testing the final model
2. Define a training pipeline in terms of a sequence of TF Layers
3. Compile the pipeline
4. Feed training data into the pipeline (cpu intensive)
5. (optionally save the trained model)
6. Query the trained model

#### NumPy Datasets (ndarray)

- Use numpy to create training dataset and inputs in the form of python `ndarray`.
- Pipe numpy ndarrays -> tf Dataset, for training, validation, and testing.
- Use Pandas for text file read and write (CSV, etc)

#### TensorFlow Datasets (tf.data.Dataset)

Datasets from tensorflow_datasets are of type tf.data.Dataset (and optional tfds.core.DatasetInfo).

#### Use Keras as Facade into TensorFlow Layers

1. Use keras to define processing layers and create a model.
2. Use keras to define an optimizer and loss function.
3. Compile keras model (layers and optimizer).
4. Train the model by calling model.fit().      <=== THIS IS WHERE THE CPU CHURNS and SPIKES
5. Test the model using known expected outputs
5. For all queries, call model.predict( ) with inputs

### Problems

#### Problem 1: A predictive n-dimensional data point model

Description: Load a 2D x, y datapoint set that is roughly quadratic and have the model generate y
values given x.

Variants:

* 2 points should produce a simple linear essentially perfect model
* a scatter of many points with linear model should fit the best line
* 3 points to define a quadratic with 2nd order model
* Use this simple case as a way to understand standard Keras layers
* Write a custom Keras layer, possibly a finite element solver

#### Problem 2: App Suggester

Instead of a simple statistical model in which the applications are just the most commonly used, this will factor in day
of the week time of day animal favor, recent history over older history

- Time of day
- Location GPS
- Day of week

#### Problem 3: Generate full 2K bitmaps from textual descriptions

Train: Associate text descriptions with 2D images
Output: 2D bitmaps that fit a query. The model assembles images
wherein keywords in the query are matched to

#### Problem 4: Local Document knowledge store

A model takes in a bunch of general information in the form of source and word documents, emails.
The model can then 'search' for concepts, present them by data, author,
source document.

For example, a set of state laws and regulations.

#### Problem 5: Voice and Music generator


