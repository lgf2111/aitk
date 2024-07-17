# aitk

AI Toolkit

## Getting Started

To get started with aitk, follow these steps:

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/aitk.git
   cd aitk
   ```

2. Create a virtual environment with Python version lower than 3.12 (Whisper requires Python 3.11 or lower):

   ```
   python3.11 -m venv .venv
   ```

3. Activate the virtual environment:

   - On Windows:
     ```
     .venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source .venv/bin/activate
     ```

4. Install the project dependencies using Poetry:
   ```
   poetry install
   ```

Now you're ready to use aitk!

To try out aitk, follow these steps:

### Testing

1. API Testing:
   To manually test functions as an API, run:

   ```
   fastapi dev aitk/api.py
   ```

2. CLI Testing:
   To manually test functions as a CLI, run:

   ```
   python -m aitk
   ```

3. Automated Testing:
   To run automated tests, use:
   ```
   pytest
   ```
