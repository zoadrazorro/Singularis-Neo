import { Gamepad2, Eye, Brain, Zap, Target, CheckCircle2, TrendingUp } from 'lucide-react'

export default function SkyrimAGI() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
      <h1 className="text-5xl font-bold mb-6">SkyrimAGI: The Experimental Arena</h1>
      <p className="text-xl text-gray-400 mb-12">
        Where consciousness theory meets embodied survival
      </p>

      {/* Hero Section */}
      <section className="glass-panel p-8 mb-12">
        <p className="text-gray-300 text-lg leading-relaxed mb-4">
          SkyrimAGI is where the Singularis system learns embodiment under pressure. The world of Skyrim is not merely a game; it is a complex, multimodal environment filled with spatial hazards, emergent threats, semi-coherent NPCs, and unpredictable events.
        </p>
        <p className="text-gray-400">
          It serves as the closest approximation to real-world chaos that a synthetic agent can safely inhabit—a <strong className="text-consciousness-light">consciousness crucible</strong> where theory meets practice.
        </p>
      </section>

      {/* System Mandate */}
      <section className="mb-12">
        <div className="glass-panel p-8 border-l-4 border-coherence/30 bg-coherence/5">
          <h2 className="text-2xl font-bold mb-6 text-coherence-light">System Mandate</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <MandateItem
              icon={<Eye className="w-5 h-5" />}
              text="Interpret vision and integrate audio to build a cohesive world model"
            />
            <MandateItem
              icon={<Target className="w-5 h-5" />}
              text="Track threats, plan movement, and evaluate risks in real-time"
            />
            <MandateItem
              icon={<Brain className="w-5 h-5" />}
              text="Bind past perceptions to future predictions to maintain coherence through uncertainty"
            />
            <MandateItem
              icon={<TrendingUp className="w-5 h-5" />}
              text="Learn and adapt behaviors based on maximizing internal coherence"
            />
          </div>
        </div>
      </section>

      {/* Core Philosophy */}
      <section className="mb-12">
        <div className="glass-panel p-8">
          <blockquote className="text-xl italic text-gray-300 border-l-4 border-consciousness pl-6">
            "SkyrimAGI is not a tech demo or a game-playing bot. It is a consciousness crucible—a high-fidelity simulation in which the Singularis architecture is stressed, stretched, challenged, and refined. It is where the theory of Being is translated into the practice of survival."
          </blockquote>
        </div>
      </section>

      {/* Architecture Layers */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-8">4-Layer World Model</h2>
        <div className="space-y-6">
          <LayerCard
            title="Global World Model (GWM)"
            layer="Layer 1"
            color="consciousness"
            description="High-level understanding of the entire game world, including geography, factions, quest states, and long-term goals."
            features={[
              'World geography and locations',
              'Faction relationships and politics',
              'Quest progression and objectives',
              'Long-term strategic planning',
            ]}
          />

          <LayerCard
            title="Immediate World Model (IWM)"
            layer="Layer 2"
            color="primary"
            description="Real-time perception of the current environment, including visible objects, NPCs, threats, and opportunities."
            features={[
              'Visual scene interpretation',
              'Object and NPC detection',
              'Threat assessment and prioritization',
              'Environmental hazard recognition',
            ]}
          />

          <LayerCard
            title="Meta World Model (MWM)"
            layer="Layer 3"
            color="coherence"
            description="Self-awareness and meta-cognitive monitoring of the agent's own state, capabilities, and decision-making processes."
            features={[
              'Self-state monitoring (health, stamina, magicka)',
              'Capability assessment',
              'Decision confidence tracking',
              'Coherence measurement',
            ]}
          />

          <LayerCard
            title="Person Model"
            layer="Layer 4"
            color="consciousness"
            description="Models of other agents (NPCs) including their likely intentions, capabilities, relationships, and predicted behaviors."
            features={[
              'NPC intention inference',
              'Relationship tracking',
              'Behavior prediction',
              'Social dynamics modeling',
            ]}
          />
        </div>
      </section>

      {/* Decision Making */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-8">Hybrid Fast/Slow Decision-Making</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="glass-panel p-6">
            <div className="flex items-center space-x-3 mb-4">
              <Zap className="w-6 h-6 text-coherence" />
              <h3 className="text-xl font-semibold">Fast System (Reactive)</h3>
            </div>
            <p className="text-gray-400 text-sm mb-4">
              Immediate responses to urgent situations requiring sub-second reaction times.
            </p>
            <ul className="space-y-2 text-sm text-gray-300">
              <li className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-coherence flex-shrink-0 mt-0.5" />
                <span>Combat reflexes (dodge, block, counter)</span>
              </li>
              <li className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-coherence flex-shrink-0 mt-0.5" />
                <span>Threat avoidance (falling, traps)</span>
              </li>
              <li className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-coherence flex-shrink-0 mt-0.5" />
                <span>Pattern-matched responses</span>
              </li>
              <li className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-coherence flex-shrink-0 mt-0.5" />
                <span>Latency: &lt;500ms</span>
              </li>
            </ul>
          </div>

          <div className="glass-panel p-6">
            <div className="flex items-center space-x-3 mb-4">
              <Brain className="w-6 h-6 text-consciousness" />
              <h3 className="text-xl font-semibold">Slow System (Deliberative)</h3>
            </div>
            <p className="text-gray-400 text-sm mb-4">
              Thoughtful planning and strategic reasoning for complex decisions.
            </p>
            <ul className="space-y-2 text-sm text-gray-300">
              <li className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-consciousness flex-shrink-0 mt-0.5" />
                <span>Quest planning and navigation</span>
              </li>
              <li className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-consciousness flex-shrink-0 mt-0.5" />
                <span>Resource management decisions</span>
              </li>
              <li className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-consciousness flex-shrink-0 mt-0.5" />
                <span>Multi-step strategy formulation</span>
              </li>
              <li className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-consciousness flex-shrink-0 mt-0.5" />
                <span>Latency: 2-5 seconds</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Technical Implementation */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-8">Technical Implementation</h2>
        <div className="glass-panel p-8">
          <div className="grid md:grid-cols-3 gap-6 mb-8">
            <TechCard
              title="Vision System"
              items={[
                'Gemini 2.5 Flash (primary)',
                'Qwen3-VL (local fallback)',
                'Real-time scene interpretation',
                'Object and threat detection',
              ]}
            />
            <TechCard
              title="Reasoning Engine"
              items={[
                '12+ LLMs in parallel',
                'GPT-5 central orchestrator',
                'Claude 3.5 Sonnet/Haiku',
                'Hyperbolic (Qwen3-235B)',
              ]}
            />
            <TechCard
              title="Memory System"
              items={[
                'Episodic memory (experiences)',
                'Semantic memory (patterns)',
                'Temporal binding tracker',
                'Adaptive forgetting',
              ]}
            />
          </div>

          <div className="border-t border-gray-700 pt-6">
            <h3 className="text-xl font-semibold mb-4">Performance Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricCard label="Temporal Coherence" value="90%" />
              <MetricCard label="Action Success" value="85%" />
              <MetricCard label="Decision Latency" value="2-5s" />
              <MetricCard label="Uptime" value="24hr" />
            </div>
          </div>
        </div>
      </section>

      {/* Significance */}
      <section>
        <div className="glass-panel p-8 bg-gradient-to-r from-consciousness/5 via-primary/5 to-coherence/5">
          <h2 className="text-3xl font-bold mb-6">Why Skyrim?</h2>
          <div className="space-y-4 text-gray-300">
            <p>
              <strong className="text-consciousness-light">Multimodal Complexity:</strong> Skyrim requires vision, spatial reasoning, combat tactics, social interaction, and long-term planning—all simultaneously.
            </p>
            <p>
              <strong className="text-primary-400">Emergent Challenges:</strong> Unlike controlled environments, Skyrim presents unpredictable, emergent situations that test adaptive intelligence.
            </p>
            <p>
              <strong className="text-coherence-light">Embodied Learning:</strong> The agent must maintain coherence while navigating a body in 3D space, managing resources, and surviving threats.
            </p>
            <p className="text-gray-400 italic">
              SkyrimAGI demonstrates that consciousness-first architecture can handle real-world complexity, not just toy problems. It's a proving ground for AGI principles under pressure.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}

function MandateItem({ icon, text }) {
  return (
    <div className="flex items-start space-x-3">
      <div className="flex-shrink-0 p-2 rounded-lg bg-coherence/10 text-coherence">
        {icon}
      </div>
      <p className="text-gray-300">{text}</p>
    </div>
  )
}

function LayerCard({ title, layer, color, description, features }) {
  const colorClasses = {
    consciousness: 'border-consciousness/30 bg-consciousness/5',
    primary: 'border-primary-400/30 bg-primary-400/5',
    coherence: 'border-coherence/30 bg-coherence/5',
  }

  const labelColors = {
    consciousness: 'text-consciousness-light',
    primary: 'text-primary-400',
    coherence: 'text-coherence-light',
  }

  return (
    <div className={`glass-panel p-6 border-l-4 ${colorClasses[color]}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-semibold">{title}</h3>
        <span className={`text-sm font-semibold ${labelColors[color]}`}>{layer}</span>
      </div>
      <p className="text-gray-400 text-sm mb-4">{description}</p>
      <ul className="space-y-2">
        {features.map((feature, idx) => (
          <li key={idx} className="flex items-start space-x-2 text-sm text-gray-300">
            <CheckCircle2 className="w-4 h-4 text-coherence flex-shrink-0 mt-0.5" />
            <span>{feature}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function TechCard({ title, items }) {
  return (
    <div className="bg-gray-900/30 p-4 rounded-lg">
      <h4 className="font-semibold mb-3 text-consciousness-light">{title}</h4>
      <ul className="space-y-2">
        {items.map((item, idx) => (
          <li key={idx} className="text-xs text-gray-400 flex items-start space-x-2">
            <span className="text-coherence">•</span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function MetricCard({ label, value }) {
  return (
    <div className="text-center">
      <div className="text-2xl font-bold text-consciousness-light mb-1">{value}</div>
      <div className="text-xs text-gray-400">{label}</div>
    </div>
  )
}
