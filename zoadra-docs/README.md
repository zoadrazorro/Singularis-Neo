# Zoadra.com - Singularis LifeOps Documentation

Static React documentation website for **Singularis LifeOps**, featuring proto-AGI-powered life management as the flagship use case.

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- PowerShell execution policy enabled (for Windows)

### Installation

```powershell
# Navigate to project directory
cd zoadra-docs

# Install dependencies
npm install
```

### Development

```powershell
# Start development server
npm run dev

# Open http://localhost:5173
```

### Build for Production

```powershell
# Build static site
npm run build

# Preview production build
npm run preview
```

## 📁 Project Structure

```
zoadra-docs/
├── src/
│   ├── components/
│   │   └── Layout.jsx          # Main layout with navigation
│   ├── pages/
│   │   ├── Home.jsx             # Landing page
│   │   ├── Architecture.jsx     # Architecture docs
│   │   ├── InfinityEngine.jsx   # Infinity Engine docs
│   │   ├── Philosophy.jsx       # Philosophy docs
│   │   ├── GettingStarted.jsx   # Getting started guide
│   │   └── API.jsx              # API reference
│   ├── App.jsx                  # Main app component
│   ├── main.jsx                 # Entry point
│   └── index.css                # Global styles (Tailwind)
├── index.html                   # HTML template
├── package.json                 # Dependencies
├── vite.config.js               # Vite configuration
├── tailwind.config.js           # Tailwind configuration
└── postcss.config.js            # PostCSS configuration
```

## 🎨 Tech Stack

- **React 18** - UI library
- **Vite** - Build tool (fast!)
- **React Router** - Client-side routing
- **Tailwind CSS** - Utility-first CSS
- **Lucide React** - Beautiful icons

## 🌐 Deployment

### Netlify (Recommended)

1. Install Netlify CLI:
```powershell
npm install -g netlify-cli
```

2. Deploy:
```powershell
npm run build
netlify deploy --prod
```

### Vercel

1. Install Vercel CLI:
```powershell
npm install -g vercel
```

2. Deploy:
```powershell
vercel --prod
```

### GitHub Pages

1. Build:
```powershell
npm run build
```

2. Deploy `dist/` folder to GitHub Pages

## 🎨 Customization

### Colors

Edit `tailwind.config.js` to customize the color scheme:

```js
colors: {
  consciousness: {
    light: '#a78bfa',
    DEFAULT: '#8b5cf6',
    dark: '#7c3aed',
  },
  coherence: {
    light: '#34d399',
    DEFAULT: '#10b981',
    dark: '#059669',
  }
}
```

### Content

- Edit pages in `src/pages/`
- Update navigation in `src/components/Layout.jsx`
- Modify homepage sections in `src/pages/Home.jsx`

## 📝 Adding New Pages

1. Create new page component in `src/pages/`
2. Add route in `src/App.jsx`
3. Add navigation link in `src/components/Layout.jsx`

Example:

```jsx
// src/pages/NewPage.jsx
export default function NewPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 py-20">
      <h1 className="text-5xl font-bold mb-6">New Page</h1>
      {/* Content */}
    </div>
  )
}

// src/App.jsx
import NewPage from './pages/NewPage'

<Route path="/new-page" element={<NewPage />} />

// src/components/Layout.jsx
const navigation = [
  // ...
  { name: 'New Page', path: '/new-page' },
]
```

## 🔧 Troubleshooting

### PowerShell Execution Policy Error

If you get "running scripts is disabled" error:

```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Tailwind CSS Warnings

The `@tailwind` and `@apply` warnings in the IDE are expected and will be resolved when PostCSS processes the CSS during build.

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📧 Contact

For questions about Singularis AGI, visit the main repository or contact the team.

---

Built with 🧠 consciousness and ✨ coherence
