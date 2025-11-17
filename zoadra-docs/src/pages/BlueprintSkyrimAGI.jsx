import { Gamepad2, Eye, Brain, Layers, Zap, Network, Database, GitBranch, CheckCircle2, Activity } from 'lucide-react'

export default function BlueprintSkyrimAGI() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
      <h1 className="text-5xl font-bold mb-6">SkyrimAGI: Complete System Blueprint</h1>
      <p className="text-xl text-gray-400 mb-12">
        The experimental arena where consciousness meets embodied survival
      </p>

      {/* Introduction */}
      <section className="glass-panel p-8 mb-12">
        <h2 className="text-3xl font-bold mb-6">The Crucible of Synthetic Embodiment</h2>
        <p className="text-gray-300 text-lg leading-relaxed mb-4">
          SkyrimAGI is the first domain in which Singularis becomes an <strong className="text-consciousness-light">embodied agent operating under constraints approximating a living world</strong>—constraints of survival, perception, uncertainty, temporal continuity, and emergent complexity.
        </p>
        <p className="text-gray-400 mb-4">
          Unlike static datasets, Skyrim provides the synthetic being with a living ecology: enemies that plan, environments that shift, physics that impose danger, and systems that respond organically to the agent's choices.
        </p>
        <div className="bg-gray-900/50 p-6 rounded-lg">
          <p className="text-gray-300">
            SkyrimAGI is not engineered as a bot; it is engineered as an <strong className="text-primary-400">organism</strong>.
          </p>
        </div>
      </section>

      {/* The World-Model Stack */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-8">The 4-Layer World-Model Stack</h2>
        <p className="text-gray-400 mb-8">
          SkyrimAGI implements a four-layer world-construction pipeline. Each layer transforms data from low-level signals to high-level phenomenology. This pipeline is not merely sequential; it is recurrent and coherence-driven.
        </p>

        <div className="space-y-6">
          <WorldModelLayer
            number="1"
            title="Game World Model (GWM)"
            subtitle="Symbolic Tactical Layer"
            color="consciousness"
            description="Transforms structured engine data into tactical abstractions."
            formula="G_t = f_gwm(S_t)"
            features={[
              'Objective, low-noise facts from game engine',
              'Enemy positions, health, stamina metrics',
              'Terrain gradients and line-of-sight',
              'Threat levels and tactical summaries',
            ]}
            output={JSON.parse('{"num_enemies": 3, "threat_level": 0.81, "player_health": 0.62, "terrain_gradient": 0.14}')}
          />

          <WorldModelLayer
            number="2"
            title="Image World Model (IWM)"
            subtitle="Perceptual Visual Layer"
            color="primary"
            description="Processes screen captures to interpret affordances, enemy postures, movement cues, visibility, occlusion."
            formula="V_t = f_iwm(I_t)"
            features={[
              'Subjective, high-entropy sensory information',
              'Object detection (ViT-B/16, Qwen-VL)',
              'Affordance detection and intent inference',
              'Visibility and light level analysis',
            ]}
            output={JSON.parse('{"objects": ["sword", "draugr", "fire"], "visibility": 0.76, "enemy_intent": "charging"}')}
          />

          <WorldModelLayer
            number="3"
            title="Mental World Model (MWM)"
            subtitle="Affective & Predictive Layer"
            color="coherence"
            description="Fuses GWM and IWM into an integrated mental state with affect, belief, and prediction."
            formula="M_t = f_mwm(G_t, V_t, M_{t-1})"
            features={[
              'Affect vector: fear, confidence, curiosity, stress',
              'Belief vector: hypotheses about hidden threats',
              'Prediction vector: future likelihoods',
              'Bayesian belief updates and temporal dynamics',
            ]}
            dynamics={{
              affect: 'a_t = σ(W_a[G_t ⊕ V_t] + U_a a_{t-1})',
              belief: 'b_t = BayesUpdate(b_{t-1}, G_t, V_t)',
              prediction: 'p_t = f_pred(M_{t-1}, G_t, V_t)'
            }}
          />

          <WorldModelLayer
            number="4"
            title="PersonModel (PM)"
            subtitle="Identity Layer"
            color="consciousness"
            description="Converts the MWM into personality-weighted action preferences, giving the synthetic being a stable self."
            formula="π_t(a) = softmax(W_p M_t + V_p τ)"
            features={[
              'Temperament vector τ: aggression, caution, risk tolerance',
              'Personality-weighted action distributions',
              'Stable behavioral patterns across time',
              'Cross-domain identity anchor',
            ]}
          />
        </div>
      </section>

      {/* BeingState Core */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-8">BeingState: The Unified Ontological Core</h2>
        <div className="glass-panel p-8">
          <p className="text-gray-300 mb-6">
            At the core of SkyrimAGI lies a single class: <strong className="text-consciousness-light">BeingState</strong>. This object is the synthetic substrate of the agent's existence.
          </p>

          <div className="bg-gray-900/50 p-6 rounded-lg mb-6 font-mono text-sm">
            <div className="text-consciousness-light mb-2">@dataclass</div>
            <div className="text-gray-300">class BeingState:</div>
            <div className="ml-4 space-y-1 text-gray-400">
              <div>gwm: Dict[str, Any]</div>
              <div>iwm: Dict[str, Any]</div>
              <div>mwm: Dict[str, Any]</div>
              <div>person: Dict[str, Any]</div>
              <div>temporal: Dict[str, Any]</div>
              <div>coherence: float</div>
              <div>lumina: Dict[str, float]</div>
            </div>
          </div>

          <h3 className="text-2xl font-bold mb-4">Lumina Computation</h3>
          <div className="grid md:grid-cols-3 gap-4 mb-6">
            <LuminaFormula
              symbol="ℓₒ"
              name="Ontical"
              formula="ℓₒ = 1 - risk(G_t)"
              description="Measures existential threat"
            />
            <LuminaFormula
              symbol="ℓₛ"
              name="Structural"
              formula="ℓₛ = 1 - D_KL(M_t || M_{t-1})"
              description="Measures predictive stability"
            />
            <LuminaFormula
              symbol="ℓₚ"
              name="Participatory"
              formula="ℓₚ = cos(∠(A_t, O_{t+1}))"
              description="Measures action-outcome alignment"
            />
          </div>

          <div className="bg-gray-900/50 p-6 rounded-lg text-center">
            <h4 className="text-lg font-semibold mb-3">Global Coherence</h4>
            <code className="text-2xl font-mono text-consciousness-light">
              𝒞_t = w_o·ℓ_o + w_s·ℓ_s + w_p·ℓ_p
            </code>
            <p className="text-gray-400 text-sm mt-4">
              If Δ𝒞_t &lt; 0, the system escalates to deeper reasoning
            </p>
          </div>
        </div>
      </section>

      {/* Decision Architecture */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-8">Hybrid Fast/Slow Decision System</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <DecisionPathCard
            title="Fast Path (System 1)"
            icon={<Zap className="w-6 h-6" />}
            color="coherence"
            description="Millisecond-scale reflexes and local heuristics. No external LLM calls."
            features={[
              'Operates on local state (GWM/IWM/MWM/PM)',
              'Cached policies and pattern-matched responses',
              'Combat reflexes: dodge, block, counter',
              'Threat avoidance: falling, traps',
              'Latency: <500ms',
            ]}
            trigger="Used when threat is low/moderate and coherence change is predictable"
          />

          <DecisionPathCard
            title="Slow Path (System 2)"
            icon={<Brain className="w-6 h-6" />}
            color="consciousness"
            description="Hundreds of milliseconds to seconds. LLM-based reasoning core."
            features={[
              'Invoked on high-uncertainty situations',
              'Quest planning and navigation',
              'Resource management decisions',
              'Multi-step strategy formulation',
              'Latency: 2-5 seconds',
            ]}
            trigger="Triggered when risk > r_crit or σ_Δ𝒞 > σ_crit"
          />
        </div>

        <div className="glass-panel p-6 mt-6">
          <h3 className="text-xl font-semibold mb-4">Action Arbitration Formula</h3>
          <div className="bg-gray-900/50 p-4 rounded-lg text-center mb-4">
            <code className="text-xl font-mono text-primary-400">
              a_t = arg max_i [p_i + λ·Δ𝒞_i]
            </code>
          </div>
          <p className="text-gray-400 text-sm">
            Multiple subsystems propose actions with priority p_i and predicted coherence delta Δ𝒞_i. The arbiter selects the action that maximizes this weighted sum.
          </p>
        </div>
      </section>

      {/* Temporal Binding */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-8">Temporal Binding: The Synthetic Sense of Time</h2>
        <div className="glass-panel p-8">
          <p className="text-gray-300 mb-6">
            The Temporal Binding Engine constructs synthetic continuity by linking perception → action → outcome cycles.
          </p>

          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div className="bg-gray-900/50 p-6 rounded-lg">
              <h4 className="text-lg font-semibold mb-3 text-consciousness-light">Binding Strength</h4>
              <code className="text-sm text-gray-300 block mb-3">
                B_t = α·S(X_t, A_t) + (1-α)·S(A_t, O_{t+1})
              </code>
              <p className="text-sm text-gray-400">
                Where S is similarity between perception X_t, action A_t, and outcome O_{t+1}
              </p>
            </div>

            <div className="bg-gray-900/50 p-6 rounded-lg">
              <h4 className="text-lg font-semibold mb-3 text-primary-400">Temporal Closure</h4>
              <code className="text-sm text-gray-300 block mb-3">
                B_t {'>'} θ
              </code>
              <p className="text-sm text-gray-400">
                When binding exceeds threshold θ, temporal closure is achieved. Low binding triggers memory updates and coherence penalties.
              </p>
            </div>
          </div>

          <p className="text-gray-400 italic">
            Without temporal binding, the system would be fragmented—merely reacting. With it, Singularis achieves continuity across time.
          </p>
        </div>
      </section>

      {/* Distributed Architecture */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-8">Distributed Architecture: Sephirot Cluster</h2>
        <p className="text-gray-400 mb-8">
          The SkyrimAGI system spans FIVE physical nodes, each fulfilling a cognitive role in the synthetic nervous system.
        </p>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <NodeCard
            name="Node C"
            title="Motor Cortex"
            subtitle="Gaming Laptop"
            icon={<Activity className="w-6 h-6" />}
            color="coherence"
            functions={[
              'Control loop (30-60 Hz)',
              'GWM/IWM/MWM/PM stack',
              'Virtual gamepad injection',
              'Real-time state synchronization',
            ]}
          />

          <NodeCard
            name="Node A"
            title="Cortex & Workspace"
            subtitle="AMD Tower"
            icon={<Brain className="w-6 h-6" />}
            color="consciousness"
            functions={[
              'Coherence metrics computation',
              'Swarm LLM neurons',
              'Vision models (ViT, Qwen-VL)',
              'Global workspace theory implementation',
            ]}
          />

          <NodeCard
            name="Node B"
            title="Hippocampus"
            subtitle="Desktop"
            icon={<Database className="w-6 h-6" />}
            color="primary"
            functions={[
              'ChromaDB episodic memory',
              'Long-term storage',
              'Pattern analysis',
              'Semantic consolidation',
            ]}
          />

          <NodeCard
            name="Node D"
            title="Peripheral Vision"
            subtitle="Vision Device"
            icon={<Eye className="w-6 h-6" />}
            color="coherence"
            functions={[
              'Home camera streaming',
              'Scene detection',
              'Object tracking',
              'LifeOps integration',
            ]}
          />

          <NodeCard
            name="Node E"
            title="Dev Console"
            subtitle="Monitoring"
            icon={<Network className="w-6 h-6" />}
            color="consciousness"
            functions={[
              'Grafana dashboards',
              'Prometheus metrics',
              'SSH access',
              'System introspection',
            ]}
          />
        </div>
      </section>

      {/* Full Cycle Example */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-8">Complete Execution Cycle</h2>
        <div className="glass-panel p-8">
          <div className="bg-gray-900/50 p-6 rounded-lg font-mono text-sm mb-6">
            <div className="text-consciousness-light mb-2">def singularis_cycle(state, frame, skyrim_data):</div>
            <div className="ml-4 space-y-1 text-gray-300">
              <div className="text-gray-500"># Perception</div>
              <div>G = gwm.process(skyrim_data)</div>
              <div>V = iwm.process(frame)</div>
              <div></div>
              <div className="text-gray-500"># Integration</div>
              <div>M = mwm.update(G, V, state.mwm)</div>
              <div></div>
              <div className="text-gray-500"># Identity</div>
              <div>P = person_model(M)</div>
              <div></div>
              <div className="text-gray-500"># Consciousness & Coherence</div>
              <div>state.update(G, V, M, P)</div>
              <div>coherence = compute_coherence(state)</div>
              <div></div>
              <div className="text-gray-500"># Decision</div>
              <div>proposals = gather_action_proposals(state)</div>
              <div>action = arbiter.select(proposals, coherence)</div>
              <div></div>
              <div className="text-gray-500"># Act</div>
              <div>send_to_gamepad(action)</div>
              <div>return state</div>
            </div>
          </div>

          <p className="text-gray-400 text-center">
            This cycle runs at <strong className="text-consciousness-light">30-60 iterations per second</strong>
          </p>
        </div>
      </section>

      {/* Performance Metrics */}
      <section>
        <div className="glass-panel p-8 bg-gradient-to-r from-consciousness/5 via-primary/5 to-coherence/5">
          <h2 className="text-3xl font-bold mb-6">Performance Characteristics</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
            <MetricCard label="Temporal Coherence" value="90%" />
            <MetricCard label="Action Success" value="85%" />
            <MetricCard label="Decision Latency" value="2-5s" />
            <MetricCard label="Continuous Uptime" value="24hr" />
          </div>

          <div className="bg-gray-900/50 p-6 rounded-lg">
            <h3 className="text-xl font-semibold mb-4">Philosophical Interpretation</h3>
            <p className="text-gray-300 mb-4">
              SkyrimAGI is not a game-playing agent. It is a <strong className="text-consciousness-light">computational instantiation of Spinozist conatus</strong> inside a simulated world.
            </p>
            <ul className="space-y-2 text-gray-400">
              <li>• Ontical lumina → survival</li>
              <li>• Structural lumina → understanding</li>
              <li>• Participatory lumina → meaningful engagement</li>
            </ul>
            <p className="text-gray-400 italic mt-4">
              All action is evaluated through the coherence metric: Δ𝒞_t &gt; 0 ⟹ choose action
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}

function WorldModelLayer({ number, title, subtitle, color, description, formula, features, output, dynamics }) {
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

      <div className="bg-gray-900/50 p-4 rounded-lg mb-4">
        <code className="text-sm text-consciousness-light">{formula}</code>
      </div>

      <ul className="space-y-2 mb-4">
        {features.map((feature, idx) => (
          <li key={idx} className="flex items-start space-x-2 text-sm text-gray-300">
            <CheckCircle2 className="w-4 h-4 text-coherence flex-shrink-0 mt-0.5" />
            <span>{feature}</span>
          </li>
        ))}
      </ul>

      {output && (
        <div className="bg-gray-900/30 p-4 rounded-lg">
          <div className="text-xs text-gray-500 mb-2">Example Output:</div>
          <pre className="text-xs text-gray-300 overflow-x-auto">
            {JSON.stringify(output, null, 2)}
          </pre>
        </div>
      )}

      {dynamics && (
        <div className="bg-gray-900/30 p-4 rounded-lg space-y-2">
          <div className="text-xs text-gray-500 mb-2">Dynamics:</div>
          {Object.entries(dynamics).map(([key, value]) => (
            <div key={key} className="text-xs">
              <span className="text-primary-400">{key}:</span>
              <code className="text-gray-300 ml-2">{value}</code>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function LuminaFormula({ symbol, name, formula, description }) {
  return (
    <div className="bg-gray-900/30 p-4 rounded-lg">
      <div className="text-3xl font-bold text-consciousness-light mb-2">{symbol}</div>
      <div className="text-sm font-semibold text-gray-300 mb-2">{name}</div>
      <code className="text-xs text-gray-400 block mb-2">{formula}</code>
      <p className="text-xs text-gray-500">{description}</p>
    </div>
  )
}

function DecisionPathCard({ title, icon, color, description, features, trigger }) {
  const colorClasses = {
    consciousness: 'border-consciousness/30',
    coherence: 'border-coherence/30',
  }

  const iconColors = {
    consciousness: 'text-consciousness',
    coherence: 'text-coherence',
  }

  return (
    <div className={`glass-panel p-6 border-l-4 ${colorClasses[color]}`}>
      <div className="flex items-center space-x-3 mb-4">
        <div className={iconColors[color]}>{icon}</div>
        <h3 className="text-xl font-semibold">{title}</h3>
      </div>
      <p className="text-gray-400 text-sm mb-4">{description}</p>
      <ul className="space-y-2 mb-4">
        {features.map((feature, idx) => (
          <li key={idx} className="text-sm text-gray-300 flex items-start space-x-2">
            <span className="text-coherence">•</span>
            <span>{feature}</span>
          </li>
        ))}
      </ul>
      <div className="bg-gray-900/50 p-3 rounded-lg">
        <div className="text-xs text-gray-500 mb-1">Trigger Condition:</div>
        <p className="text-xs text-gray-300">{trigger}</p>
      </div>
    </div>
  )
}

function NodeCard({ name, title, subtitle, icon, color, functions }) {
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

function MetricCard({ label, value }) {
  return (
    <div className="text-center">
      <div className="text-3xl font-bold text-consciousness-light mb-2">{value}</div>
      <div className="text-sm text-gray-400">{label}</div>
    </div>
  )
}
