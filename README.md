# Deploy a Text Generation Model with MLflow and Streamlit

A practical project demonstrating how to package, manage, and deploy a pre-trained text generation model using MLflow and Streamlit.

The project uses a Hugging Face GPT-2 model for text generation, MLflow to wrap and save the model as a reusable Python model, and Streamlit to create an interactive web interface where users can enter prompts and generate text.

## Overview

Deploying a machine learning model involves more than simply training or loading a model. A complete deployment workflow needs to handle model management, input processing, inference, and user interaction.

This project demonstrates a simple end-to-end deployment workflow:

```text
Hugging Face GPT-2
        |
        v
MLflow Model Wrapper
        |
        v
Saved MLflow Model
        |
        v
Streamlit Application
        |
        v
User Prompt
        |
        v
Generated Text
```

The notebook is organized into three main tasks:

1. Wrap the text generation model with MLflow
2. Build an interactive Streamlit interface
3. Launch and test the deployed application

## Technologies Used

| Technology                | Purpose                          |
| ------------------------- | -------------------------------- |
| Python                    | Core programming language        |
| Hugging Face Transformers | Loading and running GPT-2        |
| GPT-2                     | Text generation model            |
| MLflow                    | Model packaging and management   |
| MLflow PyFunc             | Custom model wrapper             |
| Streamlit                 | Interactive web interface        |
| Pandas                    | Handling structured model inputs |
| Jupyter Notebook          | Development environment          |

## Model

The project uses the Hugging Face model:

```text
gpt2
```

The model is loaded using the Transformers text-generation pipeline:

```python
generator = pipeline(
    "text-generation",
    model="gpt2"
)
```

The model generates continuation text based on a user-provided prompt.

## Task 1: MLflow Model Wrapper

The first part of the project demonstrates how to wrap a Hugging Face text generation pipeline using the MLflow `PythonModel` interface.

A custom class is created:

```python
class TextGenerationModel(mlflow.pyfunc.PythonModel):
```

The wrapper implements the required `predict()` method and handles multiple input formats.

Supported inputs include:

* Pandas DataFrame
* Pandas Series
* Python list
* Single string

This allows the model to work with different input formats while maintaining a consistent prediction interface.

### Text Generation Configuration

The implemented model uses:

```text
max_new_tokens = 50
num_return_sequences = 1
do_sample = True
```

The generated text is extracted from the Hugging Face pipeline output and returned as a list of generated strings.

### Saving the Model

The wrapped model is saved using:

```python
mlflow.pyfunc.save_model(
    path="text_generation_model",
    python_model=TextGenerationModel(generator)
)
```

This creates a reusable MLflow model that can later be loaded independently of the original notebook cell.

## Task 2: Streamlit Interface

The second part of the project creates a simple web application using Streamlit.

The application provides:

* A title and description
* A text area for entering prompts
* A Generate Text button
* Input validation
* Generated text display

The interface follows this workflow:

```text
User enters prompt
        |
        v
Generate Text button
        |
        v
Load MLflow model
        |
        v
Run prediction
        |
        v
Display generated text
```

### User Input

The application uses a Streamlit text area:

```python
prompt = st.text_area(
    "Enter your prompt:",
    placeholder="Write something to start the generation..."
)
```

The application checks whether the user has entered a valid prompt before running inference.

### Model Loading

The saved MLflow model is loaded using:

```python
model = mlflow.pyfunc.load_model(
    "text_generation_model"
)
```

The prompt is then passed to the model:

```python
result = model.predict([prompt])
```

The generated text is displayed directly in the Streamlit interface.

## Task 3: Deployment and Testing

The final part of the project demonstrates how to launch the Streamlit application.

The application can be started using:

```bash
streamlit run app.py
```

When Streamlit starts successfully, it provides available URLs that can be used to access the application.

The deployment workflow is tested by:

1. Launching the Streamlit server
2. Opening the available application URL
3. Entering a text prompt
4. Clicking Generate Text
5. Checking the generated output

## Installation

Install the required dependencies:

```bash
pip install numpy==1.26.4
pip install scipy==1.13.1
pip install transformers==4.44.2
pip install mlflow
pip install streamlit
pandas
```

For the final command, use:

```bash
pip install streamlit pandas
```

Or install the main dependencies together:

```bash
pip install numpy==1.26.4 scipy==1.13.1 transformers==4.44.2 mlflow streamlit pandas
```

## Running the Project

### 1. Clone the Repository

```bash
git clone <your-repository-link>
cd <repository-name>
```

### 2. Install Dependencies

```bash
pip install numpy==1.26.4 scipy==1.13.1 transformers==4.44.2 mlflow streamlit pandas
```

### 3. Run the Streamlit Application

If the Streamlit code is saved in `app.py`:

```bash
streamlit run app.py
```

### 4. Open the Application

Streamlit will display the available URL in the terminal.

Open the provided URL in your browser and enter a prompt to generate text.

## Project Structure

```text
Text-Generation-Deployment/
|
├── deploy.ipynb
├── app.py
├── text_generation_model/
└── README.md
```

The exact files in the repository may vary depending on how the MLflow model and Streamlit application are saved.

## Example Usage

Enter a prompt such as:

```text
Artificial intelligence is transforming the world by
```

The GPT-2 model will generate a continuation based on the provided text.

## MLflow Benefits

MLflow provides a standardized way to package and manage the model.

In this project, MLflow is used to:

* Create a custom model wrapper
* Standardize model prediction
* Save the model
* Load the model later
* Separate model management from the user interface

This makes it easier to integrate the model into an application without placing the original model-loading logic directly inside the interface.

## Streamlit Benefits

Streamlit provides a simple way to convert Python-based machine learning workflows into interactive web applications.

In this project, Streamlit handles:

* User input
* Button interactions
* Input validation
* Model inference
* Output presentation

This allows users to interact with the text generation model without directly working with Python code.

## Key Learning Outcomes

This project provides practical experience with:

* Loading pre-trained Hugging Face models
* Text generation using GPT-2
* Creating custom MLflow PyFunc models
* Implementing the MLflow `predict()` interface
* Handling multiple model input formats
* Saving and loading MLflow models
* Building Streamlit interfaces
* Connecting a machine learning model to a web application
* Testing an AI application locally
* Understanding the basic machine learning deployment lifecycle

## Important Considerations

The current implementation is designed for learning and demonstration purposes.

The model is loaded when the application processes a generation request. For a production application, model loading could be optimized using Streamlit caching so that the model does not need to be loaded repeatedly.

Additional production considerations could include:

* Model caching
* Input length limits
* Error handling
* Authentication
* Logging
* Resource management
* Model versioning
* Containerization
* Cloud deployment
* GPU-based inference

## Future Improvements

The project can be extended by adding:

* Temperature control
* Maximum token controls
* Top-k and top-p sampling
* Multiple generated responses
* Model selection
* Streamlit model caching
* MLflow experiment tracking
* MLflow model registry
* Docker deployment
* Cloud deployment using AWS or GCP
* GPU inference
* Better error handling
* Application logging

## Conclusion

This project demonstrates a complete introductory workflow for deploying a text generation model.

A pre-trained GPT-2 model is wrapped using MLflow, saved as a reusable model, and integrated into a Streamlit application. Users can then provide prompts through a web interface and receive generated text from the deployed model.

The project provides a practical foundation for understanding how machine learning models can move from a development environment into an interactive application.

## Author

Aayush Kumar
