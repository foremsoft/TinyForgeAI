# Module 1: Setup Your Computer

**Time needed:** 20 minutes
**Prerequisites:** Module 0 (concepts)
**Goal:** Install everything needed to start building AI

---

## What We'll Install

```
┌─────────────────────────────────────────────────────────────┐
│                    Your AI Toolkit                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. Python          - The programming language             │
│   2. pip             - Tool to install Python packages      │
│   3. VS Code         - Where you'll write code              │
│   4. TinyForgeAI     - The AI training platform             │
│   5. Git (optional)  - For downloading code                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Install Python

Python is the most popular language for AI. Let's install it.

### Windows

1. **Go to python.org**
   - Open your web browser
   - Type: `python.org`
   - Press Enter

2. **Download Python**
   - Click the yellow "Download Python 3.12.x" button
   - The download starts automatically

3. **Run the Installer**
   - Find the downloaded file (usually in Downloads folder)
   - Double-click `python-3.12.x-amd64.exe`

4. **IMPORTANT: Check "Add Python to PATH"**
   ```
   ┌─────────────────────────────────────────────────────────┐
   │  Install Python 3.12.x                                  │
   │                                                         │
   │  ☑ Add python.exe to PATH    ← CHECK THIS BOX!         │
   │                                                         │
   │  [Install Now]                                          │
   └─────────────────────────────────────────────────────────┘
   ```

5. **Click "Install Now"**
   - Wait for installation to complete
   - Click "Close" when done

### Mac

1. **Open Terminal**
   - Press `Cmd + Space`
   - Type "Terminal"
   - Press Enter

2. **Install Homebrew (if not installed)**
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Install Python**
   ```bash
   brew install python@3.12
   ```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
```

---

## Step 2: Verify Python Installation

Let's make sure Python is installed correctly.

### Windows

1. **Open Command Prompt**
   - Press `Windows key`
   - Type "cmd"
   - Press Enter

2. **Check Python version**
   ```
   python --version
   ```

   You should see:
   ```
   Python 3.12.x
   ```

3. **Check pip version**
   ```
   pip --version
   ```

   You should see:
   ```
   pip 24.x from ...
   ```

### Mac/Linux

1. **Open Terminal**

2. **Check Python**
   ```bash
   python3 --version
   pip3 --version
   ```

### Troubleshooting

**Problem: "python is not recognized"**
```
Solution: Python wasn't added to PATH
Fix: Reinstall Python and CHECK the "Add to PATH" box
```

**Problem: "pip is not recognized"**
```
Solution: Run this command:
python -m ensurepip --upgrade
```

**Problem: Wrong Python version**
```
Solution: You might have multiple Python versions
Try: python3 --version
Or:  py -3 --version (Windows)
```

---

## Step 3: Install VS Code (Code Editor)

VS Code is a free, beginner-friendly code editor.

### All Operating Systems

1. **Go to code.visualstudio.com**
   - Open browser
   - Type: `code.visualstudio.com`

2. **Download VS Code**
   - Click "Download for [Your OS]"
   - Run the installer
   - Follow the prompts (default options are fine)

3. **Open VS Code**
   - Launch VS Code
   - You'll see a welcome screen

4. **Install Python Extension**
   - Click the Extensions icon (4 squares on left sidebar)
   - Search "Python"
   - Click "Install" on "Python" by Microsoft

```
┌──────────────────────────────────────────────────────────────┐
│  VS Code                                           - □ X    │
├──────────────────────────────────────────────────────────────┤
│ ┌─────┐                                                      │
│ │ 📁  │  EXPLORER                                           │
│ │ 🔍  │                                                      │
│ │ 🔀  │                                                      │
│ │ 🐛  │                                                      │
│ │ ⬛  │  ← Click this (Extensions)                          │
│ └─────┘                                                      │
│         Search: Python                                       │
│         ┌────────────────────────────────────────┐          │
│         │ 🐍 Python                    [Install] │          │
│         │    Microsoft                           │          │
│         └────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────┘
```

---

## Step 4: Create Your Project Folder

Let's create a dedicated folder for your AI projects.

### Windows

1. **Open File Explorer**
2. **Go to Documents**
3. **Right-click → New → Folder**
4. **Name it: `AI-Projects`**

Or use Command Prompt:
```
mkdir %USERPROFILE%\Documents\AI-Projects
cd %USERPROFILE%\Documents\AI-Projects
```

### Mac/Linux

```bash
mkdir ~/Documents/AI-Projects
cd ~/Documents/AI-Projects
```

---

## Step 5: Install TinyForgeAI

Now the exciting part - installing TinyForgeAI!

### Option A: Install from PyPI (Easiest)

```bash
pip install tinyforgeai
```

### Option B: Install from GitHub (Latest Version)

1. **Clone the repository**
   ```bash
   cd ~/Documents/AI-Projects
   git clone https://github.com/foremsoft/TinyForgeAI.git
   cd TinyForgeAI
   ```

2. **Install in development mode**
   ```bash
   pip install -e .
   ```

### What Gets Installed

```
TinyForgeAI installs these tools for you:
├── transformers     - AI model library
├── torch            - Deep learning framework
├── fastapi          - Web API framework
├── sentence-transformers - For semantic search
└── ... and more
```

---

## Step 6: Verify TinyForgeAI Installation

Let's make sure everything works!

### Create a Test Script

1. **Open VS Code**

2. **Open your AI-Projects folder**
   - File → Open Folder
   - Select `AI-Projects`

3. **Create a new file**
   - File → New File
   - Save as: `test_install.py`

4. **Type this code:**

```python
# test_install.py - Verify TinyForgeAI installation

print("=" * 50)
print("TinyForgeAI Installation Test")
print("=" * 50)

# Test 1: Python version
import sys
print(f"\n✓ Python version: {sys.version}")

# Test 2: Basic imports
try:
    import json
    import csv
    print("✓ Standard library: OK")
except ImportError as e:
    print(f"✗ Standard library: {e}")

# Test 3: TinyForgeAI components
try:
    # These are components from TinyForgeAI
    from difflib import SequenceMatcher  # Used for text matching
    print("✓ Text matching: OK")
except ImportError as e:
    print(f"✗ Text matching: {e}")

# Test 4: Web framework
try:
    import fastapi
    print(f"✓ FastAPI version: {fastapi.__version__}")
except ImportError:
    print("✗ FastAPI not installed (optional)")

# Test 5: AI libraries
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  GPU available: {torch.cuda.is_available()}")
except ImportError:
    print("✗ PyTorch not installed (needed for training)")

try:
    import transformers
    print(f"✓ Transformers version: {transformers.__version__}")
except ImportError:
    print("✗ Transformers not installed (needed for training)")

print("\n" + "=" * 50)
print("Installation test complete!")
print("=" * 50)
```

5. **Run the script**
   - Open Terminal in VS Code: View → Terminal
   - Type: `python test_install.py`
   - Press Enter

### Expected Output

```
==================================================
TinyForgeAI Installation Test
==================================================

✓ Python version: 3.12.x
✓ Standard library: OK
✓ Text matching: OK
✓ FastAPI version: 0.100.x
✓ PyTorch version: 2.x.x
  GPU available: False (or True if you have GPU)
✓ Transformers version: 4.x.x

==================================================
Installation test complete!
==================================================
```

### Troubleshooting

**Missing packages?** Install them:
```bash
pip install fastapi uvicorn torch transformers
```

**GPU not detected?** That's OK! You can train on CPU (slower but works).

**Permission errors?** Try:
```bash
pip install --user [package-name]
```

---

## Step 7: Install Additional Tools (Optional but Recommended)

### For Document Processing
```bash
pip install python-docx PyMuPDF  # Word docs and PDFs
```

### For Database Connections
```bash
pip install sqlalchemy  # Database support
```

### For Web Interfaces
```bash
pip install uvicorn jinja2  # Web server and templates
```

### Install All at Once
```bash
pip install python-docx PyMuPDF sqlalchemy uvicorn jinja2 sentence-transformers
```

---

## Your Development Environment

After setup, your environment looks like this:

```
Your Computer
├── Python 3.12
│   ├── pip (package installer)
│   └── Packages:
│       ├── tinyforgeai
│       ├── torch
│       ├── transformers
│       ├── fastapi
│       └── ...
│
├── VS Code
│   └── Python Extension
│
└── AI-Projects/
    └── TinyForgeAI/  (if cloned from GitHub)
```

---

## Quick Reference: Common Commands

| Task | Command |
|------|---------|
| Check Python version | `python --version` |
| Install a package | `pip install package-name` |
| List installed packages | `pip list` |
| Run a Python script | `python script.py` |
| Open VS Code | `code .` (from terminal) |

---

## Checkpoint Quiz

**1. Why did we check "Add Python to PATH"?**
<details>
<summary>Click for answer</summary>
So we can run Python from any folder in the command line. Without PATH, the computer doesn't know where to find Python.
</details>

**2. What is pip?**
<details>
<summary>Click for answer</summary>
pip is Python's package installer. It downloads and installs Python packages (like TinyForgeAI) from the internet.
</details>

**3. What is VS Code?**
<details>
<summary>Click for answer</summary>
VS Code (Visual Studio Code) is a free code editor where you write and run your Python code. It has helpful features like syntax highlighting and a built-in terminal.
</details>

---

## Summary

| Step | What You Did |
|------|--------------|
| 1 | Installed Python 3.12 |
| 2 | Verified Python and pip work |
| 3 | Installed VS Code editor |
| 4 | Created AI-Projects folder |
| 5 | Installed TinyForgeAI |
| 6 | Verified everything works |
| 7 | Installed extra tools |

---

## What's Next?

In **Module 2: Your First AI Script**, you'll:
- Write your first AI-powered code
- See something "magical" happen with just 10 lines
- Understand how the code works, line by line

**You've done the hard part - setup is complete!**

---

[← Back to Module 0](00-what-is-ai.md) | [Continue to Module 2 →](02-your-first-ai-script.md)
