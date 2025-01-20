🍺 Welcome to the Beer LSTM Project!
====================================

Project Overview
----------------

Ever wondered what insights you can brew from beer-related data? This project dives into beer analytics using a Long Short-Term Memory (LSTM) model. Upload your datasets, visualize them, and let the LSTM model do its magic!

Repository Structure
--------------------

-   **📁 data/**

    -   `dataset1.csv`: First brew of data goodness.

    -   `dataset2.csv`: Another round of beer data.

    -   `dataset3.csv`: The third pint of data for your analysis.

-   **📁 templates/**

    -   `index.html`: The gateway to our Flask app. Upload, visualize, and enjoy!

-   **🍺 LSTM_model.py**

    -   Craft your LSTM model right here. Train it, tune it, and let it flow!

-   **📦 lstm_model.keras**

    -   Pre-trained LSTM model, ready to serve predictions.

-   **🚀 main.py**

    -   Launch the Flask app and start pouring in your datasets.

Getting Started
---------------

### Prerequisites

Before you get started, make sure you have the essentials:

-   Python 3.x

-   Flask

-   TensorFlow/Keras

-   Pandas

-   NumPy

### Installation

1.  Clone this repository:

    ```
    git clone https://github.com/Aarush1137/ABInBev_LSTMmodel.git
    cd ABInBev_LSTMmodel
    ```

2.  Install the necessary Python packages:

    ```
    pip install -r requirements.txt
    ```

### Running the Application

1.  Fire up the Flask app:

    ```
    python main.py
    ```

2.  Head over to your favorite browser and navigate to `http://localhost:5000` to explore the app.

### Using the Application

1.  Upload your CSV dataset via the web interface.

2.  Sit back and watch as your data gets visualized and the LSTM model predicts like a pro!

File Descriptions
-----------------

### `LSTM_model.py`

This is where the magic happens! Craft and train your LSTM model. Customize it to suit your taste.

### `main.py`

The brain behind the operation. Runs the Flask app, handles uploads, and serves up visualizations and predictions.

### `index.html`

Your front-row seat to the action. Upload datasets and see the results instantly!

Reminder
--------

🚨 **Important:** Don't forget to update the file paths in the Python scripts (`LSTM_model.py` and `main.py`) to match your local setup!

Contributing
------------

Got a recipe for improvement? Fork the repo, stir in your changes, and submit a pull request!

License
-------

This project is licensed under the MIT License. Cheers to open source! 🍻
