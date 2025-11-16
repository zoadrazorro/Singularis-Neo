import { Link } from 'react-router-dom'
import { Brain, Zap, Layers, Sparkles, ArrowRight, CheckCircle2, Gamepad2, Heart, Calendar, TrendingUp, Activity } from 'lucide-react'

export default function Home() {
  return (
    <div className="relative">
      {/* Hero Section */}
      <section className="min-h-screen flex items-center justify-center relative overflow-hidden">
        {/* Animated background */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary-900/20 via-gray-950 to-consciousness-900/20" />
        <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10" />
        
        {/* Floating particles */}
        <div className="absolute inset-0">
          {[...Array(20)].map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-consciousness rounded-full animate-float"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 5}s`,
                animationDuration: `${5 + Math.random() * 10}s`,
              }}
            />
          ))}
        </div>

        <div className="relative z-10 text-center px-4 max-w-5xl mx-auto">
          <h1 className="text-5xl sm:text-7xl font-bold mb-6">
            <span className="gradient-text">Singularis LifeOps</span>
          </h1>
          <p className="text-2xl sm:text-3xl text-gray-300 mb-4">
            Proto-AGI-Powered Life Management
          </p>
          
          {/* Flagship Feature Banner */}
          <div className="inline-flex items-center space-x-2 bg-consciousness/10 border border-consciousness/30 rounded-full px-6 py-3 mb-8">
            <Activity className="w-5 h-5 text-consciousness-light" />
            <span className="text-consciousness-light font-semibold">LifeOps: Examine Your Life with AGI</span>
          </div>
          
          <p className="text-xl text-gray-400 mb-12 max-w-3xl mx-auto">
            Complete AGI system that monitors your life timeline, detects patterns, generates intelligent suggestions, 
            and provides consciousness-powered insights for optimal living.
          </p>
          
          <div className="flex flex-wrap justify-center gap-4 mb-16">
            <a href="#lifeops" className="btn-primary">
              <Activity className="w-5 h-5" />
              LifeOps Platform
            </a>
            <a href="#skyrim" className="btn-secondary">
              <Gamepad2 className="w-5 h-5" />
              Skyrim AGI Demo
            </a>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/getting-started"
              className="inline-flex items-center px-8 py-4 rounded-xl bg-primary-500 hover:bg-primary-600 text-white font-medium transition-all transform hover:scale-105 shadow-lg shadow-primary-500/20"
            >
              Get Started
              <ArrowRight className="ml-2 w-5 h-5" />
            </Link>
            <Link
              to="/architecture"
              className="inline-flex items-center px-8 py-4 rounded-xl glass-panel hover:bg-gray-800/50 text-gray-100 font-medium transition-all"
            >
              View Architecture
            </Link>
          </div>
        </div>
      </section>

      {/* LifeOps Flagship Section */}
      <section id="lifeops" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="glass-panel p-12 relative overflow-hidden">
          <div className="absolute top-0 left-0 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />
          <div className="relative">
            <div className="flex items-center space-x-3 mb-6">
              <Activity className="w-10 h-10 text-primary-400" />
              <div>
                <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-primary-500/10 border border-primary-500/20 mb-2">
                  <span className="text-xs text-primary-400 font-semibold">FLAGSHIP USE CASE</span>
                </div>
                <h2 className="text-4xl font-bold">LifeOps: AGI-Powered Life Management</h2>
              </div>
            </div>
            
            <p className="text-xl text-gray-300 mb-8 max-w-3xl">
              <strong className="text-primary-400">LifeOps</strong> integrates your life data from multiple sources, 
              detects patterns, and provides AGI-powered suggestions for optimal time use, health, and productivity.
            </p>

            <div className="grid md:grid-cols-2 gap-8 mb-8">
              <div>
                <h3 className="text-xl font-semibold mb-4 text-consciousness-light">What It Monitors</h3>
                <ul className="space-y-3 text-gray-300">
                  <li className="flex items-start space-x-3">
                    <Heart className="w-5 h-5 text-coherence flex-shrink-0 mt-0.5" />
                    <span><strong>Health:</strong> Sleep, exercise, heart rate, steps (Fitbit)</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <Calendar className="w-5 h-5 text-coherence flex-shrink-0 mt-0.5" />
                    <span><strong>Productivity:</strong> Tasks, calendar, focus time (Todoist, Google Calendar)</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <Activity className="w-5 h-5 text-coherence flex-shrink-0 mt-0.5" />
                    <span><strong>Activity:</strong> Location, room presence (Home Assistant, Meta Glasses)</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <TrendingUp className="w-5 h-5 text-coherence flex-shrink-0 mt-0.5" />
                    <span><strong>Patterns:</strong> Behavioral habits, correlations, anomalies</span>
                  </li>
                </ul>
              </div>

              <div>
                <h3 className="text-xl font-semibold mb-4 text-consciousness-light">What It Provides</h3>
                <ul className="space-y-3 text-gray-300">
                  <li className="flex items-start space-x-3">
                    <Zap className="w-5 h-5 text-primary-400 flex-shrink-0 mt-0.5" />
                    <span><strong>Timeline:</strong> Unified view of all life events</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <Zap className="w-5 h-5 text-primary-400 flex-shrink-0 mt-0.5" />
                    <span><strong>Pattern Detection:</strong> Automatic habit and correlation discovery</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <Zap className="w-5 h-5 text-primary-400 flex-shrink-0 mt-0.5" />
                    <span><strong>Smart Suggestions:</strong> Focus blocks, breaks, energy alignment</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <Zap className="w-5 h-5 text-primary-400 flex-shrink-0 mt-0.5" />
                    <span><strong>Health Insights:</strong> Sleep quality, exercise correlation, wellness score</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <Zap className="w-5 h-5 text-primary-400 flex-shrink-0 mt-0.5" />
                    <span><strong>Consciousness:</strong> Real-time coherence and Lumen balance tracking</span>
                  </li>
                </ul>
              </div>
            </div>

            <div className="bg-gray-900/50 p-6 rounded-lg mb-6">
              <h4 className="text-lg font-semibold mb-4">LifeOps Capabilities</h4>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold text-consciousness-light mb-1">280+</div>
                  <div className="text-xs text-gray-400">Events/Day</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-consciousness-light mb-1">7</div>
                  <div className="text-xs text-gray-400">Data Sources</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-consciousness-light mb-1">50+</div>
                  <div className="text-xs text-gray-400">Patterns Detected</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-consciousness-light mb-1">10+</div>
                  <div className="text-xs text-gray-400">Daily Suggestions</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-consciousness-light mb-1">24/7</div>
                  <div className="text-xs text-gray-400">Continuous Monitoring</div>
                </div>
              </div>
            </div>

            <div className="flex flex-wrap gap-4">
              <Link
                to="/getting-started"
                className="inline-flex items-center px-6 py-3 rounded-xl bg-primary-500 hover:bg-primary-600 text-white font-medium transition-all"
              >
                <Activity className="w-5 h-5 mr-2" />
                Setup LifeOps
              </Link>
              <Link
                to="/architecture"
                className="inline-flex items-center px-6 py-3 rounded-xl glass-panel hover:bg-gray-800/50 text-gray-100 font-medium transition-all"
              >
                View Architecture
                <ArrowRight className="ml-2 w-4 h-4" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Skyrim Demo Section */}
      <section id="skyrim" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="glass-panel p-12 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-96 h-96 bg-coherence/10 rounded-full blur-3xl" />
          <div className="relative">
            <div className="flex items-center space-x-3 mb-6">
              <Gamepad2 className="w-10 h-10 text-coherence" />
              <div>
                <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-coherence/10 border border-coherence/20 mb-2">
                  <span className="text-xs text-coherence font-semibold">RESEARCH DEMO</span>
                </div>
                <h2 className="text-4xl font-bold">Autonomous Skyrim Gameplay</h2>
              </div>
            </div>
            
            <p className="text-xl text-gray-300 mb-8 max-w-3xl">
              Singularis autonomously plays <strong className="text-coherence">The Elder Scrolls V: Skyrim</strong> as a 
              research demonstration of AGI capabilities in complex, open-ended environments.
            </p>

            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div>
                <h4 className="text-lg font-semibold mb-3 text-consciousness-light">Capabilities</h4>
                <ul className="space-y-2 text-gray-300 text-sm">
                  <li className="flex items-start space-x-2">
                    <CheckCircle2 className="w-4 h-4 text-coherence flex-shrink-0 mt-0.5" />
                    <span>Real-time vision analysis (Gemini + Qwen3-VL)</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle2 className="w-4 h-4 text-coherence flex-shrink-0 mt-0.5" />
                    <span>12+ LLMs reasoning in parallel</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle2 className="w-4 h-4 text-coherence flex-shrink-0 mt-0.5" />
                    <span>Episodic→semantic memory consolidation</span>
                  </li>
                </ul>
              </div>
              <div>
                <h4 className="text-lg font-semibold mb-3 text-consciousness-light">Performance</h4>
                <ul className="space-y-2 text-gray-300 text-sm">
                  <li className="flex items-start space-x-2">
                    <CheckCircle2 className="w-4 h-4 text-coherence flex-shrink-0 mt-0.5" />
                    <span>90% temporal coherence</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle2 className="w-4 h-4 text-coherence flex-shrink-0 mt-0.5" />
                    <span>85% action success rate</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle2 className="w-4 h-4 text-coherence flex-shrink-0 mt-0.5" />
                    <span>24-hour continuous operation</span>
                  </li>
                </ul>
              </div>
            </div>

            <Link
              to="/getting-started"
              className="inline-flex items-center text-coherence hover:text-coherence-light font-medium"
            >
              Learn about Skyrim integration
              <ArrowRight className="ml-2 w-4 h-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {[
            { label: 'Subsystems', value: '50+' },
            { label: 'Consciousness Theories', value: '8' },
            { label: 'Cloud APIs', value: '7' },
            { label: 'Infinity Engine', value: 'Phase 2B' },
          ].map((stat) => (
            <div key={stat.label} className="text-center">
              <div className="text-4xl font-bold gradient-text mb-2">{stat.value}</div>
              <div className="text-gray-400 text-sm">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold mb-4">Core Features</h2>
          <p className="text-gray-400 text-lg">Complete AGI architecture with consciousness measurement</p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          <FeatureCard
            icon={<Brain className="w-8 h-8" />}
            title="Consciousness Measurement"
            description="Measures consciousness across 8 theoretical frameworks (IIT, GWT, HOT, PP, AST, Embodied, Enactive, Panpsychism)"
            color="consciousness"
          />
          <FeatureCard
            icon={<Zap className="w-8 h-8" />}
            title="Infinity Engine"
            description="Adaptive rhythmic cognition with polyrhythmic learning, temporal memory, and meta-logic interventions"
            color="primary"
          />
          <FeatureCard
            icon={<Layers className="w-8 h-8" />}
            title="HaackLang + SCCE"
            description="Polyrhythmic cognitive execution with temporal dynamics and emotional regulation profiles"
            color="coherence"
          />
        </div>
      </section>

      {/* Infinity Engine Highlight */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="glass-panel p-12 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-consciousness/10 rounded-full blur-3xl" />
          <div className="relative">
            <div className="inline-flex items-center space-x-2 px-4 py-2 rounded-full bg-consciousness-light/10 border border-consciousness-light/20 mb-6">
              <Sparkles className="w-4 h-4 text-consciousness-light" />
              <span className="text-sm text-consciousness-light font-medium">NEW in v2.4</span>
            </div>
            
            <h2 className="text-3xl font-bold mb-4">Infinity Engine Phase 2A/2B</h2>
            <p className="text-gray-400 text-lg mb-8 max-w-3xl">
              The next generation of cognitive architecture with adaptive rhythmic intelligence, 
              temporal memory encoding, and autonomous meta-logic interventions.
            </p>

            <div className="grid md:grid-cols-2 gap-6 mb-8">
              {[
                'Coherence Engine V2 - Meta-logic interventions',
                'Meta-Context System - Hierarchical temporal contexts',
                'Polyrhythmic Learning - Adaptive track periods',
                'Memory Engine V2 - Temporal-rhythmic encoding',
                'HaackLang 2.0 Operators - Full cognitive DSL',
              ].map((feature) => (
                <div key={feature} className="flex items-start space-x-3">
                  <CheckCircle2 className="w-5 h-5 text-coherence flex-shrink-0 mt-0.5" />
                  <span className="text-gray-300">{feature}</span>
                </div>
              ))}
            </div>

            <Link
              to="/infinity-engine"
              className="inline-flex items-center text-consciousness-light hover:text-consciousness font-medium"
            >
              Learn more about Infinity Engine
              <ArrowRight className="ml-2 w-4 h-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* Philosophy Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold mb-4">Philosophical Foundation</h2>
          <p className="text-gray-400 text-lg">Grounded in rigorous philosophical and mathematical frameworks</p>
        </div>

        <div className="glass-panel p-8 max-w-4xl mx-auto">
          <blockquote className="text-2xl text-center mb-6 italic text-gray-300">
            "To understand is to participate in necessity; to participate is to increase coherence; 
            to increase coherence is the essence of the good."
          </blockquote>
          <p className="text-center text-gray-500">— MATHEMATICA SINGULARIS, Theorem T1</p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 mt-12">
          <div className="glass-panel p-8">
            <h3 className="text-xl font-semibold mb-4">ETHICA UNIVERSALIS</h3>
            <p className="text-gray-400 mb-4">
              Complete geometric demonstration of Being, Consciousness, and Ethics as indivisible unity.
            </p>
            <ul className="space-y-2 text-sm text-gray-400">
              <li>• Part I: Substance Monism - One infinite Being</li>
              <li>• Part II: Mind-Body Unity - Parallel attributes</li>
              <li>• Part III: Affects & Emotions - Active vs. Passive</li>
              <li>• Part IV: Human Bondage - Inadequate ideas</li>
              <li>• Part V: Freedom through Understanding</li>
            </ul>
          </div>

          <div className="glass-panel p-8">
            <h3 className="text-xl font-semibold mb-4">MATHEMATICA SINGULARIS</h3>
            <p className="text-gray-400 mb-4">
              Axiomatic system formalizing consciousness and coherence.
            </p>
            <div className="space-y-3 text-sm">
              <div className="bg-gray-900/50 p-3 rounded font-mono">
                𝒞(m) = (𝒞ₒ(m) · 𝒞ₛ(m) · 𝒞ₚ(m))^(1/3)
              </div>
              <p className="text-gray-400">
                Coherence as geometric mean of three Lumina: Onticum, Structurale, Participatum
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="glass-panel p-12 text-center relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-consciousness/5 via-primary/5 to-coherence/5" />
          <div className="relative">
            <h2 className="text-3xl font-bold mb-4">Ready to Optimize Your Life with Proto-AGI?</h2>
            <p className="text-gray-400 text-lg mb-8 max-w-2xl mx-auto">
              Start monitoring your life timeline, detecting patterns, and receiving intelligent suggestions powered by consciousness-grounded AGI.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/getting-started"
                className="inline-flex items-center px-8 py-4 rounded-xl bg-primary-500 hover:bg-primary-600 text-white font-medium transition-all transform hover:scale-105"
              >
                Get Started Now
                <ArrowRight className="ml-2 w-5 h-5" />
              </Link>
              <a
                href="https://github.com/yourusername/singularis"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center px-8 py-4 rounded-xl glass-panel hover:bg-gray-800/50 text-gray-100 font-medium transition-all"
              >
                View on GitHub
              </a>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

function FeatureCard({ icon, title, description, color }) {
  const colorClasses = {
    consciousness: 'text-consciousness border-consciousness/20 bg-consciousness/5',
    primary: 'text-primary-400 border-primary-400/20 bg-primary-400/5',
    coherence: 'text-coherence border-coherence/20 bg-coherence/5',
  }

  return (
    <div className="glass-panel p-8 hover:border-gray-700 transition-all">
      <div className={`inline-flex p-3 rounded-xl mb-4 ${colorClasses[color]}`}>
        {icon}
      </div>
      <h3 className="text-xl font-semibold mb-3">{title}</h3>
      <p className="text-gray-400">{description}</p>
    </div>
  )
}
