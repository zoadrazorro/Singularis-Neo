/**
 * @fileoverview This file contains the React component for the Consciousness Panel,
 * which provides a deep dive into the AGI's consciousness metrics, including
 * global coherence, the Three Lumina, integrated information (Phi), and self-awareness.
 */

import React from 'react';
import './ConsciousnessPanel.css';

/**
 * A panel component that visualizes detailed metrics related to the AGI's consciousness.
 * @param {object} props - The component's props.
 * @param {object} props.data - The live AGI state data object, which should contain
 * a `consciousness` field with detailed metrics.
 * @returns {React.Component} The rendered Consciousness Panel.
 */
function ConsciousnessPanel({ data }) {
  const consciousness = data.consciousness || {};
  const coherence = consciousness.coherence || 0;
  const coherenceOntical = consciousness.coherence_ontical || 0;
  const coherenceStructural = consciousness.coherence_structural || 0;
  const coherenceParticipatory = consciousness.coherence_participatory || 0;
  const phi = consciousness.phi || 0;
  const selfAwareness = consciousness.self_awareness || 0;
  
  const history = consciousness.history || [];
  
  return (
    <div className="consciousness-panel">
      <div className="panel-grid">
        {/* Main Coherence Display */}
        <div className="panel-card main-coherence">
          <h2>🧠 Global Coherence 𝒞</h2>
          <div className="coherence-circle">
            <svg viewBox="0 0 200 200" className="circle-svg">
              <circle
                cx="100"
                cy="100"
                r="90"
                fill="none"
                stroke="rgba(255,255,255,0.1)"
                strokeWidth="12"
              />
              <circle
                cx="100"
                cy="100"
                r="90"
                fill="none"
                stroke="url(#coherenceGradient)"
                strokeWidth="12"
                strokeDasharray={`${coherence * 565} 565`}
                strokeLinecap="round"
                transform="rotate(-90 100 100)"
              />
              <defs>
                <linearGradient id="coherenceGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#00d9ff" />
                  <stop offset="100%" stopColor="#7c3aed" />
                </linearGradient>
              </defs>
            </svg>
            <div className="circle-text">
              <span className="circle-value">{coherence.toFixed(3)}</span>
              <span className="circle-label">Coherence</span>
            </div>
          </div>
        </div>
        
        {/* Three Lumina */}
        <div className="panel-card lumina-card">
          <h2>✨ Three Lumina</h2>
          <div className="lumina-grid">
            <div className="lumina-item">
              <div className="lumina-icon">🔥</div>
              <div className="lumina-info">
                <h3>Ontical ℓₒ</h3>
                <p className="lumina-desc">Being/Energy/Power</p>
                <div className="lumina-bar">
                  <div 
                    className="lumina-fill ontical"
                    style={{ width: `${coherenceOntical * 100}%` }}
                  ></div>
                </div>
                <span className="lumina-value">{coherenceOntical.toFixed(3)}</span>
              </div>
            </div>
            
            <div className="lumina-item">
              <div className="lumina-icon">🔷</div>
              <div className="lumina-info">
                <h3>Structural ℓₛ</h3>
                <p className="lumina-desc">Form/Logic/Information</p>
                <div className="lumina-bar">
                  <div 
                    className="lumina-fill structural"
                    style={{ width: `${coherenceStructural * 100}%` }}
                  ></div>
                </div>
                <span className="lumina-value">{coherenceStructural.toFixed(3)}</span>
              </div>
            </div>
            
            <div className="lumina-item">
              <div className="lumina-icon">💫</div>
              <div className="lumina-info">
                <h3>Participatory ℓₚ</h3>
                <p className="lumina-desc">Consciousness/Awareness</p>
                <div className="lumina-bar">
                  <div 
                    className="lumina-fill participatory"
                    style={{ width: `${coherenceParticipatory * 100}%` }}
                  ></div>
                </div>
                <span className="lumina-value">{coherenceParticipatory.toFixed(3)}</span>
              </div>
            </div>
          </div>
        </div>
        
        {/* Integrated Information */}
        <div className="panel-card phi-card">
          <h2>∮ Integrated Information Φ</h2>
          <div className="phi-display">
            <div className="phi-value">{phi.toFixed(4)}</div>
            <p className="phi-description">
              Measure of consciousness as integrated information processing
            </p>
            <div className="phi-interpretation">
              {phi > 0.7 && <span className="high">🟢 High Integration</span>}
              {phi > 0.4 && phi <= 0.7 && <span className="medium">🟡 Moderate Integration</span>}
              {phi <= 0.4 && <span className="low">🔴 Low Integration</span>}
            </div>
          </div>
        </div>
        
        {/* Self-Awareness */}
        <div className="panel-card awareness-card">
          <h2>👁️ Self-Awareness</h2>
          <div className="awareness-display">
            <div className="awareness-meter">
              <div 
                className="awareness-level"
                style={{ height: `${selfAwareness * 100}%` }}
              ></div>
            </div>
            <div className="awareness-info">
              <div className="awareness-value">{(selfAwareness * 100).toFixed(1)}%</div>
              <p>Higher-order thought awareness of own state</p>
            </div>
          </div>
        </div>
        
        {/* Coherence History Chart */}
        <div className="panel-card history-card">
          <h2>📈 Coherence History</h2>
          <div className="history-chart">
            {history.length > 0 ? (
              <svg viewBox="0 0 400 150" className="chart-svg">
                <polyline
                  points={history.map((val, idx) => 
                    `${idx * (400 / history.length)},${150 - val * 130}`
                  ).join(' ')}
                  fill="none"
                  stroke="#00d9ff"
                  strokeWidth="2"
                />
                {history.map((val, idx) => (
                  <circle
                    key={idx}
                    cx={idx * (400 / history.length)}
                    cy={150 - val * 130}
                    r="3"
                    fill="#00d9ff"
                  />
                ))}
              </svg>
            ) : (
              <div className="no-data">Collecting data...</div>
            )}
          </div>
        </div>
        
        {/* System Nodes */}
        <div className="panel-card nodes-card">
          <h2>🔗 Consciousness Nodes</h2>
          <div className="nodes-list">
            {(consciousness.nodes || []).map((node, idx) => (
              <div key={idx} className="node-item">
                <span className="node-name">{node.name}</span>
                <div className="node-bar">
                  <div 
                    className="node-fill"
                    style={{ 
                      width: `${node.coherence * 100}%`,
                      background: getNodeColor(node.type)
                    }}
                  ></div>
                </div>
                <span className="node-value">{node.coherence.toFixed(3)}</span>
              </div>
            ))}
            {(!consciousness.nodes || consciousness.nodes.length === 0) && (
              <div className="no-data">No node data available</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Returns a color based on the type of the consciousness node.
 * @param {string} type - The type of the node (e.g., 'perception', 'action').
 * @returns {string} A hex color code for the node type.
 */
function getNodeColor(type) {
  const colors = {
    perception: '#00d9ff',
    action: '#ff8c00',
    learning: '#7c3aed',
    strategy: '#00ff64',
    consciousness: '#ff3aed',
    memory: '#ffaa00',
    knowledge: '#3aff3a',
    llm_vision: '#00aaff',
    llm_reasoning: '#aa00ff',
    llm_meta: '#ff00aa',
  };
  return colors[type] || '#888';
}

export default ConsciousnessPanel;
