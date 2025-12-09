# TinyForgeAI Dashboard

A lightweight web dashboard for managing TinyForgeAI training jobs, models, and RAG search.

## Features

- **Overview**: Dashboard statistics and quick actions
- **Training Jobs**: Create and monitor training jobs
- **Models**: View and deploy trained models
- **RAG Search**: Search indexed documents

## Quick Start

### Option 1: Serve with Python (No dependencies)

```bash
# From the dashboard directory
cd dashboard
python -m http.server 3000

# Open http://localhost:3000
```

### Option 2: Serve with the Dashboard API

The dashboard is designed to work with the Dashboard API backend:

```bash
# Start the dashboard API (from project root)
uvicorn services.dashboard_api.main:app --reload --port 8001

# Serve the dashboard
cd dashboard
python -m http.server 3000
```

### Option 3: Use with FastAPI Static Files

The Dashboard API can serve the dashboard directly:

```python
from fastapi.staticfiles import StaticFiles

app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")
```

## Project Structure

```
dashboard/
├── index.html      # Main HTML file
├── styles.css      # CSS styles
├── app.js          # JavaScript application
└── README.md       # This file
```

## Configuration

### API URL

By default, the dashboard connects to `http://localhost:8001` (Dashboard API).

To change this, edit `app.js`:

```javascript
const API_BASE_URL = 'http://your-api-server:8001';
```

## Pages

### Overview
- Statistics cards (jobs, models, documents, requests)
- Quick action buttons
- Recent activity feed

### Training Jobs
- List of training jobs with status
- Create new jobs with configuration
- Progress tracking

### Models
- Grid of trained models
- Download and deploy options
- Model metadata

### RAG Search
- Full-text search across indexed documents
- Search results with relevance scores
- Index statistics

## API Endpoints Used

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/stats` | GET | Dashboard statistics |
| `/api/jobs` | GET | List training jobs |
| `/api/jobs` | POST | Create training job |
| `/api/models` | GET | List trained models |
| `/api/search` | GET | Search documents |

## Development

### Running Locally

1. Start the Dashboard API:
   ```bash
   make dashboard
   # or
   uvicorn services.dashboard_api.main:app --reload --port 8001
   ```

2. Serve the dashboard:
   ```bash
   cd dashboard
   python -m http.server 3000
   ```

3. Open http://localhost:3000

### Customization

The dashboard uses vanilla HTML/CSS/JavaScript for simplicity:

- **Styling**: Edit `styles.css` - uses CSS custom properties for theming
- **Behavior**: Edit `app.js` - all JavaScript in one file
- **Structure**: Edit `index.html` - standard HTML

### Adding a New Page

1. Add navigation link in `index.html`:
   ```html
   <a href="#" class="nav-link" data-page="newpage">New Page</a>
   ```

2. Add page section:
   ```html
   <section id="page-newpage" class="page">
       <h2>New Page</h2>
       <!-- Content -->
   </section>
   ```

3. The navigation is handled automatically by `app.js`

## Tech Stack

- HTML5
- CSS3 (with CSS Custom Properties)
- Vanilla JavaScript (ES6+)
- No build tools required
- No npm/node dependencies

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Contributing

See the main [CONTRIBUTING.md](../docs/CONTRIBUTING.md) for guidelines.
