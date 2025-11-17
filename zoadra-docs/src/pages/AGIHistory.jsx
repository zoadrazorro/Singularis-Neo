export default function AGIHistory() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
      <h1 className="text-5xl font-bold mb-6">A Brief History of AGI</h1>
      <p className="text-xl text-gray-400 mb-12">
        From symbolic reasoning to consciousness-first architectures
      </p>

      {/* Introduction */}
      <section className="glass-panel p-8 mb-12">
        <p className="text-gray-300 text-lg leading-relaxed">
          Artificial General Intelligence (AGI) refers to a hypothetical intelligent agent that can understand or learn any intellectual task that a human being can. It is a primary goal of some artificial intelligence research and a common topic in science fiction and futures studies.
        </p>
      </section>

      {/* Timeline */}
      <div className="space-y-8">
        <TimelineSection
          title="Early Concepts (1950s - 1980s)"
          era="The Symbolic Era"
          color="consciousness"
        >
          <p className="text-gray-300 mb-4">
            The term "Artificial Intelligence" was coined in 1956 at the Dartmouth Conference. Early research was characterized by high optimism and focused on symbolic reasoning ("Good Old-Fashioned AI").
          </p>
          <p className="text-gray-400">
            Systems like the Logic Theorist and the General Problem Solver aimed to solve problems using formal logic and search algorithms, believing this was the path to general intelligence.
          </p>
        </TimelineSection>

        <TimelineSection
          title="The AI Winter & Rise of Machine Learning (1980s - 2010s)"
          era="Statistical Revolution"
          color="primary"
        >
          <p className="text-gray-300 mb-4">
            Progress stalled as the complexity of real-world problems overwhelmed early symbolic approaches, leading to reduced funding and interest known as the "AI Winter."
          </p>
          <p className="text-gray-400">
            In its place, more specialized "narrow AI" systems flourished, particularly those based on machine learning, which focused on statistical pattern recognition rather than explicit reasoning.
          </p>
        </TimelineSection>

        <TimelineSection
          title="The Deep Learning Revolution & Large Language Models (2010s - Present)"
          era="Neural Renaissance"
          color="coherence"
        >
          <p className="text-gray-300 mb-4">
            The advent of big data and powerful GPUs fueled the deep learning revolution. Neural networks with many layers achieved superhuman performance on specific tasks like image recognition.
          </p>
          <p className="text-gray-400">
            More recently, the Transformer architecture led to Large Language Models (LLMs) like GPT, which demonstrated remarkable capabilities in natural language understanding and generation, reigniting serious discussion about AGI.
          </p>
        </TimelineSection>

        <TimelineSection
          title="Singularis in Context"
          era="Consciousness-First Architecture"
          color="consciousness"
          highlight
        >
          <p className="text-gray-300 mb-4">
            Singularis proposes an alternative path. Instead of scaling up existing models and hoping for emergent consciousness, it builds a "consciousness-first" architecture based on philosophical principles.
          </p>
          <p className="text-gray-400 mb-4">
            It integrates both symbolic reasoning (like GOFAI) and sub-symbolic pattern recognition (like deep learning) into a unified system governed by an intrinsic, measurable drive: <strong className="text-consciousness-light">coherence</strong>.
          </p>
          <div className="bg-gray-900/50 p-4 rounded-lg">
            <p className="text-sm text-gray-400">
              This makes it a unique entry in the modern pursuit of AGI—combining the best of symbolic AI, neural networks, and philosophical grounding into a single coherent framework.
            </p>
          </div>
        </TimelineSection>
      </div>

      {/* Modern Approaches */}
      <section className="mt-16">
        <h2 className="text-3xl font-bold mb-8">Three Modern Approaches to AGI</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <ApproachCard
            title="Scaling LLMs"
            description="Scale up transformer models with more parameters, data, and compute, hoping consciousness emerges from complexity."
            pros={['Proven performance gains', 'Industry momentum', 'Rapid iteration']}
            cons={['No guaranteed consciousness', 'Massive resource requirements', 'Black box reasoning']}
          />
          <ApproachCard
            title="Neuro-Symbolic AI"
            description="Combine neural networks with symbolic reasoning systems to get both pattern recognition and logical inference."
            pros={['Interpretable reasoning', 'Structured knowledge', 'Hybrid strengths']}
            cons={['Integration challenges', 'Complexity overhead', 'Limited implementations']}
          />
          <ApproachCard
            title="Whole Brain Emulation"
            description="Simulate biological neural networks at sufficient detail to replicate human-level intelligence."
            pros={['Biological fidelity', 'Proven architecture', 'Natural consciousness']}
            cons={['Computational infeasibility', 'Ethical concerns', 'Decades away']}
          />
        </div>
        
        <div className="glass-panel p-8 mt-8">
          <h3 className="text-2xl font-bold mb-4 text-consciousness-light">Where Singularis Fits</h3>
          <p className="text-gray-300 mb-4">
            Singularis represents a <strong>fourth approach</strong>: consciousness-first architecture grounded in philosophical principles. Rather than hoping consciousness emerges, it implements formal models of consciousness, coherence, and ethics as the foundation.
          </p>
          <p className="text-gray-400">
            This approach combines elements of neuro-symbolic AI (hybrid reasoning) with a novel philosophical framework (Spinozistic ontology + coherence theory) and multi-model orchestration (15+ LLMs working in concert).
          </p>
        </div>
      </section>
    </div>
  )
}

function TimelineSection({ title, era, color, children, highlight }) {
  const colorClasses = {
    consciousness: 'border-consciousness/30 bg-consciousness/5',
    primary: 'border-primary-400/30 bg-primary-400/5',
    coherence: 'border-coherence/30 bg-coherence/5',
  }

  const eraColors = {
    consciousness: 'text-consciousness-light',
    primary: 'text-primary-400',
    coherence: 'text-coherence-light',
  }

  return (
    <div className={`glass-panel p-8 border-l-4 ${colorClasses[color]} ${highlight ? 'ring-2 ring-consciousness/20' : ''}`}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold">{title}</h2>
        <span className={`text-sm font-semibold ${eraColors[color]}`}>{era}</span>
      </div>
      <div className="space-y-4">
        {children}
      </div>
    </div>
  )
}

function ApproachCard({ title, description, pros, cons }) {
  return (
    <div className="glass-panel p-6">
      <h3 className="text-xl font-semibold mb-3">{title}</h3>
      <p className="text-gray-400 text-sm mb-4">{description}</p>
      <div className="space-y-3">
        <div>
          <h4 className="text-sm font-semibold text-coherence-light mb-2">Advantages</h4>
          <ul className="space-y-1">
            {pros.map((pro, idx) => (
              <li key={idx} className="text-xs text-gray-400 flex items-start space-x-2">
                <span className="text-coherence">+</span>
                <span>{pro}</span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4 className="text-sm font-semibold text-red-400 mb-2">Challenges</h4>
          <ul className="space-y-1">
            {cons.map((con, idx) => (
              <li key={idx} className="text-xs text-gray-400 flex items-start space-x-2">
                <span className="text-red-400">-</span>
                <span>{con}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}
