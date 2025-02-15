# Cocktail Advisor Chat

This is a FastAPI application for a cocktail advisor chat service.

## Prerequisites

- Python 3.8+
- `venv` for virtual environment management

## Setup

1. **Clone the repository:**

    ```sh
    git clone <repository_url>
    cd cocktail-advisor-chat
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Create a `.env` file in the root directory and add your GEMINI API key:**

    ```env
    GEMINI_API_KEY=your_gemini_api_key_here
    ```

## Running the Application

1. **Start the FastAPI server:**

    ```sh
    uvicorn main:app --reload
    ```
    or
    ```sh
    python main.py
    ```

2. **Access the application:**

    Open your browser and navigate to `http://127.0.0.1:8000`.

