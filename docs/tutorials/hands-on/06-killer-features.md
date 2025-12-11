# Tutorial 06: Killer Features - URL Training, Data Augmentation & Playground

**Time:** 25 minutes
**Difficulty:** Beginner
**Prerequisites:** TinyForgeAI installed

## What You'll Learn

- Train models directly from URLs (no data preparation!)
- Generate 500+ training samples from just 5 examples
- Create shareable playgrounds for your models

These three features eliminate the biggest pain points in AI development.

---

## Part 1: One-Click URL Training (10 min)

### The Problem
Traditionally, training a model requires:
1. Export data from your source
2. Clean and format the data
3. Convert to JSONL format
4. Then finally train

### The Solution
With URL training, you just paste a link:

```bash
foremforge train-url https://notion.so/my-faq-page --out ./my_model
```

That's it. TinyForge extracts Q&A pairs automatically.

### Supported Sources

| Source | Example URL |
|--------|-------------|
| Notion | `https://notion.so/workspace/FAQ-abc123` |
| Google Docs | `https://docs.google.com/document/d/xxx` |
| Google Sheets | `https://docs.google.com/spreadsheets/d/xxx` |
| GitHub | `https://github.com/user/repo/blob/main/data.jsonl` |
| Any Website | `https://example.com/faq` |
| Raw Files | `https://example.com/data.jsonl` |

### Try It: Preview Mode

Before training, preview what will be extracted:

```bash
foremforge train-url https://example.com/faq --preview
```

Output:
```
Previewing: https://example.com/faq

Source Type: website
Title: Frequently Asked Questions
Samples Extracted: 15

Sample Preview (first 3):
--------------------------------------------------

[1] Input: What are your business hours?...
    Output: We're open Monday through Friday, 9am to 5pm...

[2] Input: How do I return an item?...
    Output: You can return any item within 30 days...

[3] Input: Do you ship internationally?...
    Output: Yes, we ship to over 50 countries...
```

### Try It: Train from URL

```bash
# Dry run (no actual training, fast)
foremforge train-url https://example.com/faq --out ./faq_model --dry-run

# Real training
foremforge train-url https://example.com/faq --out ./faq_model --epochs 3
```

### Notion Integration

For private Notion pages, you'll need an API token:

1. Go to https://www.notion.so/my-integrations
2. Create a new integration
3. Copy the token
4. Share your page with the integration

```bash
export NOTION_TOKEN=secret_xxx
foremforge train-url https://notion.so/my-workspace/FAQ --out ./model
```

---

## Part 2: Training Data Generator (10 min)

### The Problem
You have 5 FAQ examples but need 500 for good training.
Manually writing 495 more is painful.

### The Solution
The data augmenter generates variations automatically:

```bash
foremforge augment -i my_5_examples.jsonl -o augmented_500.jsonl -n 500
```

### How It Works

The augmenter uses multiple strategies:

| Strategy | What It Does | Example |
|----------|--------------|---------|
| **Synonym** | Replaces words with synonyms | "help" → "assist" |
| **Template** | Applies question templates | "How do I X?" → "What's the best way to X?" |
| **Paraphrase** | Restructures sentences | "Can't" → "Cannot" |
| **Back-translation** | Simulates translation artifacts | Natural variations |
| **LLM** | Uses AI for high-quality variations | Best quality, requires API key |

### Try It: Basic Augmentation

Create a file `examples.jsonl`:
```json
{"input": "What are your hours?", "output": "We're open 9am-5pm, Monday to Friday."}
{"input": "How do I return an item?", "output": "Visit our returns page or call customer service."}
{"input": "Do you ship internationally?", "output": "Yes, we ship to over 50 countries."}
```

Augment it:
```bash
foremforge augment -i examples.jsonl -o augmented.jsonl -n 100
```

Check the result:
```bash
wc -l augmented.jsonl  # Should show ~100 lines
head -5 augmented.jsonl  # Preview first 5
```

### Try It: LLM-Powered Augmentation

For highest quality, use an LLM:

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-xxx

# Or for OpenAI
export OPENAI_API_KEY=sk-xxx

# Run with LLM
foremforge augment -i examples.jsonl -o high_quality.jsonl -n 500 --use-llm
```

### Python API

```python
from backend.augment import DataGenerator, AugmentConfig

config = AugmentConfig(
    target_count=500,
    strategies=["synonym", "template", "paraphrase"],
)

generator = DataGenerator(config)

examples = [
    {"input": "What are your hours?", "output": "9am-5pm weekdays"},
    {"input": "How do I return items?", "output": "Use our returns page"},
]

augmented = generator.generate(examples, target_count=500)
generator.save(augmented, "training_data.jsonl")
```

---

## Part 3: Shareable Playground (5 min)

### The Problem
You trained a model. Now how do you:
- Demo it to your boss?
- Get feedback from colleagues?
- Share it on social media?

### The Solution
Create an instant playground:

```bash
foremforge playground --model ./my_model
```

Opens a beautiful web interface at http://localhost:8080

### Three Options

#### Option A: Local Server
```bash
foremforge playground --model ./my_model --port 8080
```
- Runs on your machine
- Good for personal testing

#### Option B: Public Sharing
```bash
foremforge playground --model ./my_model --share
```
- Creates a public URL (via ngrok)
- Anyone with the link can test
- Perfect for demos and reviews

#### Option C: Standalone HTML
```bash
foremforge playground --model ./my_model --export demo.html
```
- Creates a single HTML file
- Works completely offline
- Host anywhere (GitHub Pages, S3)

### Try It: Create a Playground

```bash
# Using a model from earlier tutorials
foremforge playground --model ./my_model --title "My FAQ Bot"
```

Visit http://localhost:8080 and test your model!

### Add Example Inputs

```bash
foremforge playground --model ./my_model \
    --examples "What are your hours?" \
    --examples "How do I return an item?" \
    --examples "Do you ship internationally?"
```

---

## Putting It All Together

Here's a complete workflow using all three features:

```bash
# 1. Extract data from your FAQ page
foremforge train-url https://example.com/faq --preview

# 2. If you need more data, augment it
foremforge train-url https://example.com/faq --out ./temp --dry-run
foremforge augment -i ./temp/training_data.jsonl -o ./augmented.jsonl -n 500

# 3. Train with augmented data
foremforge train --data ./augmented.jsonl --out ./my_model --dry-run

# 4. Create a shareable playground
foremforge playground --model ./my_model --share

# 5. Share the link with your team!
```

---

## Troubleshooting

### URL Training Issues

**"No samples extracted"**
- Check if the page has public access
- Try `--preview` to see what's being extracted
- For Notion/Google, ensure proper authentication

**"Connection failed"**
```bash
pip install httpx beautifulsoup4
```

### Augmentation Issues

**"Not enough variation"**
- Try different strategies: `--strategies synonym template paraphrase`
- Use `--use-llm` for better quality

**"LLM not working"**
- Verify API key is set: `echo $ANTHROPIC_API_KEY`
- Check quota/billing on API provider

### Playground Issues

**"Model not loaded"**
- Ensure model path is correct
- Check for `model_stub.json` or trained model files

**"Share link not working"**
- Install ngrok: `pip install ngrok`
- Or use cloudflare: `cloudflared tunnel --url http://localhost:8080`

---

## What's Next?

- [MCP Integration](05-mcp-integration.md) - Control TinyForgeAI with AI assistants
- [Deploy Your Project](04-deploy-your-project.md) - Put your model in production
- [Full Documentation](../../README.md) - Complete reference

---

## Summary

You learned three killer features:

1. **URL Training** - Train from any URL, no data prep needed
2. **Data Augmentation** - Generate 500 samples from 5 examples
3. **Shareable Playground** - Instant demos anyone can use

These features make TinyForgeAI the fastest way to go from idea to working AI.
