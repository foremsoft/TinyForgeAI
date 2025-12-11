# Wiki Content

This folder contains content for the [GitHub Wiki](https://github.com/foremsoft/TinyForgeAI/wiki).

## Files

| File | Wiki Page | Description |
|------|-----------|-------------|
| `Home.md` | Home | Wiki homepage |
| `Training-UIs.md` | Training-UIs | Gradio, Streamlit, React interfaces |
| `Beginners-Course.md` | Beginners-Course | 4-hour AI course overview |
| `Hands-On-Tutorials.md` | Hands-On-Tutorials | 90-minute project tutorials |

## Syncing to GitHub Wiki

To update the GitHub Wiki with these files:

### Option 1: Manual Copy
1. Go to https://github.com/foremsoft/TinyForgeAI/wiki
2. Edit each page and paste the content from these files

### Option 2: Clone Wiki Repository
```bash
# Clone the wiki repo
git clone https://github.com/foremsoft/TinyForgeAI.wiki.git

# Copy files
cp docs/wiki/*.md TinyForgeAI.wiki/

# Push changes
cd TinyForgeAI.wiki
git add .
git commit -m "Update wiki content"
git push
```

## Adding New Pages

1. Create a new `.md` file in this folder
2. Use GitHub Wiki naming conventions (e.g., `My-Page-Name.md`)
3. Update this README with the new file
4. Sync to GitHub Wiki using one of the methods above
