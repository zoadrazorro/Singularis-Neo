/**
 * @fileoverview This file contains the React component for the LLM Panel,
 * which displays the status and activity of the various Large Language Model
 * systems within the AGI.
 */

import React from 'react';
import './LLMPanel.css';

/**
 * A panel component that displays the status of the LLM (Large Language Model) systems.
 * It shows the current architecture mode, the number of active cloud and local models,
 * total API calls, and a breakdown of action sources.
 * @param {object} props - The component's props.
 * @param {object} props.data - The live AGI state data object.
 * @returns {React.Component} The rendered LLM Panel.
 */
function LLMPanel({ data }) {
  const llmStatus = data.llm_status || {};
  const stats = data.stats || {};
  
  const mode = llmStatus.mode || 'none';
  const cloudActive = llmStatus.cloud_active || 0;
  const localActive = llmStatus.local_active || 0;
  const totalCalls = llmStatus.total_calls || 0;
  const activeModels = llmStatus.active_models || [];
  
  const llmActions = stats.llm_actions || 0;
  const rlActions = stats.rl_actions || 0;
  const heuristicActions = stats.heuristic_actions || 0;
  const totalActions = llmActions + rlActions + heuristicActions || 1;
  
  return (
    <div className="llm-panel">
      <h2>🤖 LLM Systems</h2>
      
      <div className="llm-mode-display">
        <div className="mode-header">Architecture Mode</div>
        <div className="mode-name">{formatMode(mode)}</div>
      </div>
      
      <div className="llm-stats-grid">
        <div className="llm-stat-card cloud">
          <div className="stat-icon">☁️</div>
          <div className="stat-value">{cloudActive}</div>
          <div className="stat-label">Cloud LLMs Active</div>
        </div>
        
        <div className="llm-stat-card local">
          <div className="stat-icon">💻</div>
          <div className="stat-value">{localActive}</div>
          <div className="stat-label">Local LLMs Active</div>
        </div>
        
        <div className="llm-stat-card calls">
          <div className="stat-icon">📞</div>
          <div className="stat-value">{totalCalls}</div>
          <div className="stat-label">Total Calls</div>
        </div>
      </div>
      
      <div className="active-models-section">
        <h3>Active Models</h3>
        <div className="models-grid">
          {activeModels.length > 0 ? (
            activeModels.map((model, idx) => (
              <div key={idx} className="model-badge">
                <span className="model-icon">{getModelIcon(model)}</span>
                <span className="model-name">{model}</span>
              </div>
            ))
          ) : (
            <div className="no-data">No LLM models active</div>
          )}
        </div>
      </div>
      
      <div className="action-sources-section">
        <h3>Action Sources</h3>
        <div className="sources-chart">
          <div className="source-bar">
            <div className="source-label">
              <span>🤖 LLM</span>
              <span>{llmActions} ({((llmActions/totalActions)*100).toFixed(1)}%)</span>
            </div>
            <div className="source-progress">
              <div 
                className="source-fill llm"
                style={{ width: `${(llmActions/totalActions)*100}%` }}
              ></div>
            </div>
          </div>
          
          <div className="source-bar">
            <div className="source-label">
              <span>🧠 RL</span>
              <span>{rlActions} ({((rlActions/totalActions)*100).toFixed(1)}%)</span>
            </div>
            <div className="source-progress">
              <div 
                className="source-fill rl"
                style={{ width: `${(rlActions/totalActions)*100}%` }}
              ></div>
            </div>
          </div>
          
          <div className="source-bar">
            <div className="source-label">
              <span>⚙️ Heuristic</span>
              <span>{heuristicActions} ({((heuristicActions/totalActions)*100).toFixed(1)}%)</span>
            </div>
            <div className="source-progress">
              <div 
                className="source-fill heuristic"
                style={{ width: `${(heuristicActions/totalActions)*100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Formats the LLM architecture mode string for display.
 * @param {string} mode - The mode string (e.g., 'hybrid', 'moe').
 * @returns {string} The formatted, human-readable mode name.
 */
function formatMode(mode) {
  const modes = {
    'hybrid': 'Hybrid (Gemini + Claude)',
    'moe': 'Mixture of Experts',
    'parallel': 'Parallel (MoE + Hybrid)',
    'local': 'Local Only',
    'none': 'No LLM Active'
  };
  return modes[mode] || mode.toUpperCase();
}

/**
 * Returns an appropriate icon for a given LLM model name.
 * @param {string} model - The name of the model.
 * @returns {string} An emoji icon representing the model.
 */
function getModelIcon(model) {
  if (model.includes('Gemini')) return '🔷';
  if (model.includes('Claude')) return '🟣';
  if (model.includes('GPT')) return '🟢';
  if (model.includes('Qwen')) return '🔶';
  if (model.includes('Phi')) return '🔵';
  if (model.includes('Nemotron')) return '🟨';
  return '🤖';
}

export default LLMPanel;
