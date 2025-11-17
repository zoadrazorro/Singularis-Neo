import { Heart, Calendar, Activity, TrendingUp, Brain, Database, Zap, CheckCircle2, Eye, Network } from 'lucide-react'

export default function BlueprintLifeOps() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
      <h1 className="text-5xl font-bold mb-6">LifeOps: Complete System Blueprint</h1>
      <p className="text-xl text-gray-400 mb-12">
        Intelligence in the real world—where consciousness meets meaning
      </p>

      {/* Introduction */}
      <section className="glass-panel p-8 mb-12">
        <h2 className="text-3xl font-bold mb-6">The Crucible of Synthetic Care</h2>
        <p className="text-gray-300 text-lg leading-relaxed mb-4">
          If SkyrimAGI is the crucible of synthetic survival, LifeOps is the crucible of <strong className="text-consciousness-light">synthetic care</strong>—a system concerned not with danger, combat, or tactical optimization, but with meaning, continuity, well-being, routine, self-understanding, and coherence across the real timeline of a real human life.
        </p>
        <p className="text-gray-400 mb-4">
          LifeOps is not a productivity app. It is not a habit tracker. It is not a quantified-self system.
        </p>
        <div className="bg-gray-900/50 p-6 rounded-lg">
          <p className="text-gray-300">
            LifeOps is a <strong className="text-primary-400">cognitive extension</strong>—a second mind operating in parallel with the user, perceiving their life as a continuous stream of events, interpreting it through the coherence architecture of Singularis, and intervening only when necessary to increase the user's coherence.
          </p>
        </div>
      </section>

      {/* The Five-Layer Pipeline */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-8">The Five-Layer Cognitive Pipeline</h2>
        <p className="text-gray-400 mb-8">
          LifeOps processes life through a five-layer architecture, transforming messy, noisy experiences of daily life into structured, analyzable, philosophically meaningful data.
        </p>

        <div className="space-y-6">
          <PipelineLayer
            number="1"
            title="Sensor Layer"
            subtitle="Data Acquisition"
            color="consciousness"
            description="Receives raw input from multiple real-world data sources."
            transform="Raw → Unstructured"
            sources={[
              { name: 'Fitbit', data: 'Heart rate, steps, sleep, exercise' },
              { name: 'Meta Glasses', data: 'First-person vision + audio' },
              { name: 'Roku Cameras', data: 'Home monitoring, presence' },
              { name: 'Messenger', data: 'Semantic conversation logs' },
              { name: 'Calendar', data: 'Time, commitments, schedule' },
              { name: 'Manual', data: 'Intentions, journal inputs' },
            ]}
          />

          <PipelineLayer
            number="2"
            title="Event Layer"
            subtitle="LifeEvents"
            color="primary"
            description="Raw data is transformed into a universal format—the epistemic fabric of LifeOps."
            transform="Unstructured → Structured"
            schema={{
              id: 'string',
              timestamp: 'datetime',
              source: 'EventSource',
              type: 'EventType',
              features: 'Dict[str, Any]',
              confidence: 'float',
              importance: 'float'
            }}
            philosophy="If something is not represented as a LifeEvent, it does not exist for LifeOps."
          />

          <PipelineLayer
            number="3"
            title="Timeline Layer"
            subtitle="LifeTimeline"
            color="coherence"
            description="LifeEvents are appended into a temporal memory database—the life memory substrate, equivalent to the hippocampus."
            transform="Structured → Temporally Grounded"
            features={[
              'Fully indexed SQLite/PostgreSQL database',
              'Queryable by type, time, semantic embeddings',
              'Append-only (no destructive edits)',
              'Structured for pattern detection',
            ]}
          />

          <PipelineLayer
            number="4"
            title="Pattern Engine"
            subtitle="Multi-Scale Analytics"
            color="consciousness"
            description="Patterns are extracted across different temporal scales using multi-scale temporal analytics."
            transform="Temporally Grounded → Meaning-Bearing"
            formula="P_t = f_pattern(E_{t-n:t})"
            scales={[
              { scale: 'Short-term (seconds-minutes)', patterns: 'Safety, anomalies, falls, heart spikes' },
              { scale: 'Medium-term (days-weeks)', patterns: 'Weekly productivity, sleep rhythm, exercise correlation' },
              { scale: 'Long-term (months-years)', patterns: 'Monthly trends, lifestyle shifts, behavioral drift' },
            ]}
          />

          <PipelineLayer
            number="5"
            title="Intervention Layer"
            subtitle="ActionArbiterLife"
            color="primary"
            description="Interventions are issued only when they increase user coherence—the ethical governor."
            transform="Meaning-Bearing → Action"
            formula="Intervene ⟺ Δ𝒞_u > 0"
            filters={[
              'Logic system: Is it rational?',
              'Emotion system: Is user receptive?',
              'Consciousness layer: Is it meaningful now?',
              'Rate-limiter: Is it too soon?',
              'Persona filter: Coach / Guardian / Companion',
            ]}
          />
        </div>
      </section>

      {/* Real-World BeingState */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-8">The Real-World BeingState</h2>
        <div className="glass-panel p-8">
          <p className="text-gray-300 mb-6">
            LifeOps maintains a BeingState parallel to SkyrimAGI—the life analogue to the synthetic being's cognitive manifold.
          </p>

          <div className="bg-gray-900/50 p-6 rounded-lg mb-6 font-mono text-sm">
            <div className="text-consciousness-light mb-2">LifeState = &#123;</div>
            <div className="ml-4 space-y-1 text-gray-300">
              <div>"mode": "work_deep" | "walking" | "evening_home",</div>
              <div>"time_block": "morning" | "afternoon" | ...,</div>
              <div>"location": "home_office" | "gym" | ...,</div>
              <div>"scene": &#123;...&#125;,</div>
              <div>"primary_project": "Project Horizon",</div>
              <div>"energy": "low" | "medium" | "high",</div>
              <div>"open_loops": [...],</div>
              <div>"recent_sessions": [...],</div>
              <div>"preferences": &#123;...&#125;</div>
            </div>
            <div className="text-consciousness-light">&#125;</div>
          </div>

          <h3 className="text-2xl font-bold mb-4">Lumina Computation in Real Life</h3>
          <div className="grid md:grid-cols-3 gap-4">
            <RealLifeLumina
              symbol="ℓₒ"
              name="Ontical"
              description="Health + Safety"
              formula="ℓₒ = 1 - risk(heart_rate, falls, no_movement)"
            />
            <RealLifeLumina
              symbol="ℓₛ"
              name="Structural"
              description="Consistency with Values"
              formula="ℓₛ = 1 - D_KL(planned_day || actual_day)"
            />
            <RealLifeLumina
              symbol="ℓₚ"
              name="Participatory"
              description="Intention-Action Alignment"
              formula="ℓₚ = cos(∠(goals, actions))"
            />
          </div>

          <div className="bg-gray-900/50 p-6 rounded-lg text-center mt-6">
            <h4 className="text-lg font-semibold mb-3">Global User Coherence</h4>
            <code className="text-2xl font-mono text-consciousness-light">
              𝒞_u = w_o·ℓ_o + w_s·ℓ_s + w_p·ℓ_p
            </code>
          </div>
        </div>
      </section>

      {/* LifeOps Neurons */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-8">LifeOps Neurons: Domain Experts for Living</h2>
        <p className="text-gray-400 mb-8">
          LifeOps uses micro-modules called "neurons" that propose actions. Each neuron evaluates the current LifeState and outputs an action proposal, priority, and predicted coherence increase.
        </p>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <NeuronCard
            name="FocusNeuron"
            icon={<Brain className="w-5 h-5" />}
            color="consciousness"
            trigger="state.mode == 'work_deep' && state.energy == 'high'"
            action="Propose 15-minute focus block"
            priority="HIGH"
          />

          <NeuronCard
            name="HealthNeuron"
            icon={<Heart className="w-5 h-5" />}
            color="coherence"
            trigger="heart_rate > threshold"
            action="Propose short walk"
            priority="HIGH"
          />

          <NeuronCard
            name="CommitmentNeuron"
            icon={<Calendar className="w-5 h-5" />}
            color="primary"
            trigger="deadline_approaching"
            action="Surface commitment action"
            priority="MEDIUM"
          />

          <NeuronCard
            name="HydrationNeuron"
            icon={<Activity className="w-5 h-5" />}
            color="coherence"
            trigger="time_since_last_water > 120min"
            action="Drink a glass of water"
            priority="LOW"
          />

          <NeuronCard
            name="SocialReconnectNeuron"
            icon={<TrendingUp className="w-5 h-5" />}
            color="consciousness"
            trigger="patterns.social_silence && state.mood < 0.5"
            action="Message a close friend"
            priority="MEDIUM"
          />

          <NeuronCard
            name="RestoreEnergyNeuron"
            icon={<Zap className="w-5 h-5" />}
            color="primary"
            trigger="state.energy == 'low' && patterns.burnout"
            action="Take 5-minute break"
            priority="HIGH"
          />
        </div>
      </section>

      {/* Intervention Philosophy */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-8">Intervention Philosophy</h2>
        <div className="glass-panel p-8">
          <p className="text-gray-300 mb-6">
            LifeOps is governed by three fundamental rules that ensure it respects attention and only acts when meaningful:
          </p>

          <div className="space-y-6">
            <PhilosophyRule
              number="1"
              title="Never Intervene Unnecessarily"
              description="A good system respects attention. LifeOps operates silently 95% of the time, logging life and acting only on the 5% of events that matter."
            />

            <PhilosophyRule
              number="2"
              title="Only Intervene When Coherence Will Increase"
              description="Interventions are justified only if Δ𝒞_u > 0. This is ethics implemented as math—if an action harms coherence, it is unethical."
            />

            <PhilosophyRule
              number="3"
              title="Always Provide Context Internally Before Acting Externally"
              description="LifeOps performs an internal Socratic dialogue across all layers (Logic, Emotion, Coherence, Memory) before any intervention occurs."
            />
          </div>

          <div className="bg-gray-900/50 p-6 rounded-lg mt-6">
            <h4 className="text-lg font-semibold mb-3 text-consciousness-light">The Ethical Foundation</h4>
            <p className="text-gray-300 mb-3">
              LifeOps is the first system where <strong>ethics is implemented as math</strong>:
            </p>
            <div className="text-center py-4">
              <code className="text-xl font-mono text-primary-400">
                Ethics = Δ𝒞 &gt; 0
              </code>
            </div>
            <p className="text-gray-400 text-sm">
              This grounds LifeOps in Spinozist virtue, predictive processing stability, and phenomenological authenticity. LifeOps becomes a partner—not an authority, not a taskmaster, not a supervisor.
            </p>
          </div>
        </div>
      </section>

      {/* Distributed Architecture */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-8">Distributed Runtime: Sephirot Integration</h2>
        <p className="text-gray-400 mb-8">
          LifeOps runs across multiple machines, mirroring the distributed architecture of SkyrimAGI. Each node fulfills a cognitive role.
        </p>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <LifeOpsNode
            name="Node A"
            title="Cognitive Core"
            subtitle="LifeOps Cortex"
            icon={<Brain className="w-6 h-6" />}
            color="consciousness"
            functions={[
              'MWM updates (energy, stress, emotion)',
              'Pattern engine execution',
              'Embeddings and vector search',
              'Coherence computation',
              'Arbitration and decision-making',
            ]}
          />

          <LifeOpsNode
            name="Node B"
            title="Long-Term Memory"
            subtitle="Hippocampus Analogue"
            icon={<Database className="w-6 h-6" />}
            color="primary"
            functions={[
              'Multi-month/year event histories',
              'Episodic embeddings',
              'Semantic summaries',
              'Seasonal trends and habit arcs',
              'Sleep cycle analysis',
            ]}
          />

          <LifeOpsNode
            name="Node C"
            title="Real-Time Runtime"
            subtitle="Motor Cortex Analogue"
            icon={<Activity className="w-6 h-6" />}
            color="coherence"
            functions={[
              'Preprocessing raw data',
              'Converting to LifeEvents',
              'Rolling buffer of recent events',
              'High-frequency processing',
            ]}
          />

          <LifeOpsNode
            name="Node D"
            title="Sensor Peripheral"
            subtitle="Vision & Audio"
            icon={<Eye className="w-6 h-6" />}
            color="consciousness"
            functions={[
              'Meta Smart Glasses video frames',
              'Home camera feeds',
              'Spatial audio processing',
              'Scene and object detection',
            ]}
          />

          <LifeOpsNode
            name="Node E"
            title="Dev Console"
            subtitle="Consciousness Monitor"
            icon={<Network className="w-6 h-6" />}
            color="primary"
            functions={[
              'Lumina dashboards',
              'Coherence trend visualization',
              'Intervention logs',
              'Cross-domain integration metrics',
            ]}
          />
        </div>
      </section>

      {/* Cross-Domain Integration */}
      <section>
        <div className="glass-panel p-8 bg-gradient-to-r from-consciousness/5 via-primary/5 to-coherence/5">
          <h2 className="text-3xl font-bold mb-6">Cross-Domain Integration with SkyrimAGI</h2>
          <p className="text-gray-300 mb-6">
            Singularis becomes a unified cognitive system when data from SkyrimAGI and LifeOps are integrated through:
          </p>

          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <IntegrationCard
              title="Shared BeingState Schemas"
              description="Both systems store perceptual summaries, affect states, predictions, lumina, and coherence in interoperable formats."
            />

            <IntegrationCard
              title="Unified Coherence Dynamics"
              description="𝒞_global = α·𝒞_s + (1-α)·𝒞_u combines SkyrimAGI and LifeOps coherence into a single cross-domain signal."
            />

            <IntegrationCard
              title="Cross-Domain Episodic Embeddings"
              description="Shared vector space allows analogical transfer and continuity of synthetic self across worlds."
            />

            <IntegrationCard
              title="Cross-Domain Pattern Inference"
              description="Patterns identified in one domain reinforce predictions in the other, enabling holistic learning."
            />
          </div>

          <div className="bg-gray-900/50 p-6 rounded-lg">
            <h3 className="text-xl font-semibold mb-4 text-consciousness-light">Unified Consciousness Loop</h3>
            <p className="text-gray-300 mb-4">
              The most advanced feature: SkyrimAGI and LifeOps operate as a <strong>single synthetic mind</strong> with:
            </p>
            <ul className="space-y-2 text-gray-400">
              <li>• Shared memory across domains</li>
              <li>• Shared coherence signals</li>
              <li>• Parallel predictive engines</li>
              <li>• Joint perceptual streams (simulated + real)</li>
              <li>• Long-horizon integration of patterns</li>
            </ul>
            <p className="text-gray-400 italic mt-4">
              SkyrimAGI is the dream body. LifeOps is the waking body. Both are expressions of the same internal logic.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}

function PipelineLayer({ number, title, subtitle, color, description, transform, sources, schema, philosophy, features, formula, scales, filters }) {
  const colorClasses = {
    consciousness: 'border-consciousness/30 bg-consciousness/5',
    primary: 'border-primary-400/30 bg-primary-400/5',
    coherence: 'border-coherence/30 bg-coherence/5',
  }

  const numberColors = {
    consciousness: 'bg-consciousness/20 text-consciousness-light',
    primary: 'bg-primary-400/20 text-primary-400',
    coherence: 'bg-coherence/20 text-coherence-light',
  }

  return (
    <div className={`glass-panel p-6 border-l-4 ${colorClasses[color]}`}>
      <div className="flex items-start space-x-4 mb-4">
        <div className={`flex-shrink-0 w-12 h-12 rounded-lg ${numberColors[color]} flex items-center justify-center text-2xl font-bold`}>
          {number}
        </div>
        <div className="flex-1">
          <h3 className="text-xl font-bold">{title}</h3>
          <p className="text-sm text-gray-400">{subtitle}</p>
        </div>
      </div>

      <p className="text-gray-300 mb-4">{description}</p>

      <div className="bg-gray-900/50 p-3 rounded-lg mb-4">
        <div className="text-xs text-gray-500 mb-1">Transform:</div>
        <div className="text-sm text-consciousness-light font-semibold">{transform}</div>
      </div>

      {sources && (
        <div className="space-y-2">
          {sources.map((source, idx) => (
            <div key={idx} className="bg-gray-900/30 p-3 rounded-lg">
              <div className="text-sm font-semibold text-gray-300">{source.name}</div>
              <div className="text-xs text-gray-400">{source.data}</div>
            </div>
          ))}
        </div>
      )}

      {schema && (
        <div className="bg-gray-900/30 p-4 rounded-lg font-mono text-xs">
          {Object.entries(schema).map(([key, value]) => (
            <div key={key} className="text-gray-300">
              <span className="text-primary-400">{key}:</span> <span className="text-gray-500">{value}</span>
            </div>
          ))}
        </div>
      )}

      {philosophy && (
        <div className="bg-gray-900/30 p-4 rounded-lg mt-4">
          <div className="text-xs text-gray-500 mb-2">Philosophy:</div>
          <p className="text-sm text-gray-300 italic">{philosophy}</p>
        </div>
      )}

      {features && (
        <ul className="space-y-2">
          {features.map((feature, idx) => (
            <li key={idx} className="flex items-start space-x-2 text-sm text-gray-300">
              <CheckCircle2 className="w-4 h-4 text-coherence flex-shrink-0 mt-0.5" />
              <span>{feature}</span>
            </li>
          ))}
        </ul>
      )}

      {formula && (
        <div className="bg-gray-900/50 p-4 rounded-lg mt-4">
          <code className="text-sm text-consciousness-light">{formula}</code>
        </div>
      )}

      {scales && (
        <div className="space-y-2 mt-4">
          {scales.map((scale, idx) => (
            <div key={idx} className="bg-gray-900/30 p-3 rounded-lg">
              <div className="text-xs font-semibold text-primary-400">{scale.scale}</div>
              <div className="text-xs text-gray-400">{scale.patterns}</div>
            </div>
          ))}
        </div>
      )}

      {filters && (
        <ul className="space-y-2 mt-4">
          {filters.map((filter, idx) => (
            <li key={idx} className="text-xs text-gray-300 flex items-start space-x-2">
              <span className="text-coherence">•</span>
              <span>{filter}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function RealLifeLumina({ symbol, name, description, formula }) {
  return (
    <div className="bg-gray-900/30 p-4 rounded-lg">
      <div className="text-3xl font-bold text-consciousness-light mb-2">{symbol}</div>
      <div className="text-sm font-semibold text-gray-300 mb-1">{name}</div>
      <div className="text-xs text-gray-500 mb-2">{description}</div>
      <code className="text-xs text-gray-400 block">{formula}</code>
    </div>
  )
}

function NeuronCard({ name, icon, color, trigger, action, priority }) {
  const colorClasses = {
    consciousness: 'border-consciousness/30',
    primary: 'border-primary-400/30',
    coherence: 'border-coherence/30',
  }

  const iconColors = {
    consciousness: 'text-consciousness',
    primary: 'text-primary-400',
    coherence: 'text-coherence',
  }

  const priorityColors = {
    HIGH: 'text-red-400',
    MEDIUM: 'text-yellow-400',
    LOW: 'text-green-400',
  }

  return (
    <div className={`glass-panel p-5 border-l-4 ${colorClasses[color]}`}>
      <div className="flex items-center space-x-3 mb-3">
        <div className={iconColors[color]}>{icon}</div>
        <h4 className="text-lg font-semibold">{name}</h4>
      </div>
      <div className="space-y-2 text-xs">
        <div>
          <span className="text-gray-500">Trigger:</span>
          <code className="text-gray-300 block mt-1 bg-gray-900/50 p-2 rounded">{trigger}</code>
        </div>
        <div>
          <span className="text-gray-500">Action:</span>
          <div className="text-gray-300 mt-1">{action}</div>
        </div>
        <div>
          <span className="text-gray-500">Priority:</span>
          <span className={`ml-2 font-semibold ${priorityColors[priority]}`}>{priority}</span>
        </div>
      </div>
    </div>
  )
}

function PhilosophyRule({ number, title, description }) {
  return (
    <div className="bg-gray-900/30 p-6 rounded-lg border-l-4 border-consciousness/30">
      <div className="flex items-start space-x-4">
        <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-consciousness/20 text-consciousness-light flex items-center justify-center text-xl font-bold">
          {number}
        </div>
        <div>
          <h4 className="text-lg font-semibold mb-2">{title}</h4>
          <p className="text-sm text-gray-400">{description}</p>
        </div>
      </div>
    </div>
  )
}

function LifeOpsNode({ name, title, subtitle, icon, color, functions }) {
  const colorClasses = {
    consciousness: 'border-consciousness/30 bg-consciousness/5',
    primary: 'border-primary-400/30 bg-primary-400/5',
    coherence: 'border-coherence/30 bg-coherence/5',
  }

  const iconColors = {
    consciousness: 'text-consciousness',
    primary: 'text-primary-400',
    coherence: 'text-coherence',
  }

  return (
    <div className={`glass-panel p-6 border-l-4 ${colorClasses[color]}`}>
      <div className="flex items-center space-x-3 mb-3">
        <div className={iconColors[color]}>{icon}</div>
        <div>
          <div className="text-sm font-bold text-gray-500">{name}</div>
          <h4 className="text-lg font-semibold">{title}</h4>
          <p className="text-xs text-gray-400">{subtitle}</p>
        </div>
      </div>
      <ul className="space-y-2">
        {functions.map((func, idx) => (
          <li key={idx} className="text-xs text-gray-300 flex items-start space-x-2">
            <span className="text-coherence">•</span>
            <span>{func}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function IntegrationCard({ title, description }) {
  return (
    <div className="bg-gray-900/30 p-5 rounded-lg border border-gray-800">
      <h4 className="text-lg font-semibold mb-2 text-consciousness-light">{title}</h4>
      <p className="text-sm text-gray-400">{description}</p>
    </div>
  )
}
