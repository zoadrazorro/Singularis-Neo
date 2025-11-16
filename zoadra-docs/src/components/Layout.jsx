import { Link, useLocation } from 'react-router-dom'
import { Brain, Menu, X, Github } from 'lucide-react'
import { useState } from 'react'

const navigation = [
  { name: 'Home', path: '/' },
  { name: 'Architecture', path: '/architecture' },
  { name: 'Infinity Engine', path: '/infinity-engine' },
  { name: 'Philosophy', path: '/philosophy' },
  { name: 'Getting Started', path: '/getting-started' },
  { name: 'API Reference', path: '/api' },
]

export default function Layout({ children }) {
  const location = useLocation()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 glass-panel border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            {/* Logo */}
            <Link to="/" className="flex items-center space-x-3 group">
              <div className="relative">
                <Brain className="w-8 h-8 text-consciousness group-hover:text-consciousness-light transition-colors" />
                <div className="absolute inset-0 bg-consciousness/20 blur-xl group-hover:bg-consciousness-light/30 transition-all" />
              </div>
              <div>
                <span className="text-xl font-bold gradient-text">Singularis LifeOps</span>
                <span className="block text-xs text-gray-500">Proto-AGI Life Management</span>
              </div>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-1">
              {navigation.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`px-4 py-2 rounded-lg transition-all ${
                    location.pathname === item.path
                      ? 'bg-primary-500/10 text-primary-400 font-medium'
                      : 'text-gray-400 hover:text-gray-100 hover:bg-gray-800/50'
                  }`}
                >
                  {item.name}
                </Link>
              ))}
              <a
                href="https://github.com/yourusername/singularis"
                target="_blank"
                rel="noopener noreferrer"
                className="ml-4 p-2 text-gray-400 hover:text-gray-100 transition-colors"
              >
                <Github className="w-5 h-5" />
              </a>
            </div>

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 text-gray-400 hover:text-gray-100"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-gray-800 bg-gray-900/95 backdrop-blur-xl">
            <div className="px-4 py-4 space-y-2">
              {navigation.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={() => setMobileMenuOpen(false)}
                  className={`block px-4 py-3 rounded-lg transition-all ${
                    location.pathname === item.path
                      ? 'bg-primary-500/10 text-primary-400 font-medium'
                      : 'text-gray-400 hover:text-gray-100 hover:bg-gray-800/50'
                  }`}
                >
                  {item.name}
                </Link>
              ))}
            </div>
          </div>
        )}
      </nav>

      {/* Main Content */}
      <main className="pt-16">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <Brain className="w-6 h-6 text-consciousness" />
                <span className="font-bold gradient-text">Singularis LifeOps</span>
              </div>
              <p className="text-gray-400 text-sm">
                Proto-AGI-Powered Life Management. Monitor, analyze, and optimize your life with consciousness-powered insights.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-100 mb-4">Resources</h3>
              <ul className="space-y-2 text-sm">
                <li><Link to="/getting-started" className="text-gray-400 hover:text-gray-100">Getting Started</Link></li>
                <li><Link to="/api" className="text-gray-400 hover:text-gray-100">API Reference</Link></li>
                <li><Link to="/philosophy" className="text-gray-400 hover:text-gray-100">Philosophy</Link></li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-gray-100 mb-4">Community</h3>
              <ul className="space-y-2 text-sm">
                <li><a href="https://github.com/yourusername/singularis" className="text-gray-400 hover:text-gray-100">GitHub</a></li>
                <li><a href="#" className="text-gray-400 hover:text-gray-100">Discord</a></li>
                <li><a href="#" className="text-gray-400 hover:text-gray-100">Twitter</a></li>
              </ul>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t border-gray-800 text-center text-sm text-gray-500">
            <p>&copy; 2025 Singularis LifeOps. Built with consciousness and coherence.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
