# TinyForgeAI Dashboard

A React-based web dashboard for managing TinyForgeAI training and inference services.

## Features

- **Train Page**: Configure and run model training (dry-run or real)
- **Services Page**: Manage deployed inference services
- **Playground Page**: Test inference endpoints interactively
- **Logs Page**: View service logs with filtering

## Quick Start

### Prerequisites

- Node.js 18+ or 20+
- npm or yarn

### Installation

```bash
cd dashboard
npm install
```

### Development

```bash
npm run dev
```

Open http://localhost:3000 in your browser.

### Build for Production

```bash
npm run build
npm run preview
```

## Project Structure

```
dashboard/
├── src/
│   ├── main.jsx          # Entry point
│   ├── App.jsx           # Main app with routing
│   └── pages/
│       ├── TrainPage.jsx     # Training configuration
│       ├── ServicesPage.jsx  # Service management
│       ├── PlaygroundPage.jsx # API testing
│       └── LogsPage.jsx      # Log viewer
├── index.html
├── package.json
├── vite.config.js
└── README.md
```

## API Integration

The dashboard is configured to proxy API requests to `http://localhost:8000` (see `vite.config.js`).

To connect to a running TinyForgeAI service:

1. Start an inference service:
   ```bash
   foremforge serve --dir ./my_service --port 8000
   ```

2. The Playground page will automatically connect to `http://localhost:8000/predict`

## Customization

### Changing the API URL

Edit `vite.config.js`:

```javascript
server: {
  proxy: {
    '/api': {
      target: 'http://your-api-server:8000',
      changeOrigin: true,
    }
  }
}
```

### Adding New Pages

1. Create a new component in `src/pages/`
2. Add a route in `src/App.jsx`
3. Add navigation link in the nav bar

## Tech Stack

- React 18
- React Router 6
- Vite 5

## Contributing

See the main [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
