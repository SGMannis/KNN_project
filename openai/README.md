# OpenAI API Configuration

To use the OCR features of this project, you must set your OpenAI API key as an environment variable. **Do not hardcode the key into the scripts.**

## macOS / Linux (zsh)

1. Open your terminal.
2. Open your shell configuration file:

   ```bash
   nano ~/.zshrc
   ```

3. Add the following line at the end (replace `your_key_here` with your actual key):

   ```bash
   export OPENAI_API_KEY='your_key_here'
   ```

4. Save and exit (Ctrl+O, Enter, Ctrl+X).
5. Apply the changes:

   ```bash
   source ~/.zshrc
   ```

## Windows

1. Open **Command Prompt** (CMD) as Administrator.
2. Run the following command (replace `your_key_here` with your actual key):

   ```cmd
   setx OPENAI_API_KEY "your_key_here"
   ```

3. **Restart your IDE** (VS Code, PyCharm, etc.) or Terminal for the changes to take effect.

## Python Usage

The scripts will automatically load the key using the following logic:

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```
