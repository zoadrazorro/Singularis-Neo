/**
 * @fileoverview This file contains the React component for the World Model Panel,
 * which displays the AGI's internal beliefs, goals, and current strategy.
 */

import React from 'react';
import './WorldModelPanel.css';

/**
 * A panel component that displays information about the AGI's internal world model.
 * It shows the current high-level strategy, a list of active goals, and a grid
 * of the AGI's key beliefs about the world state.
 * @param {object} props - The component's props.
 * @param {object} props.data - The live AGI state data object.
 * @returns {React.Component} The rendered World Model Panel.
 */
function WorldModelPanel({ data }) {
  const worldModel = data.world_model || {};
  const beliefs = worldModel.beliefs || {};
  const goals = worldModel.goals || [];
  const strategy = worldModel.strategy || 'explore';
  
  return (
    <div className="world-model-panel">
      <h2>🌍 World Model</h2>
      
      <div className="strategy-display">
        <h3>Current Strategy</h3>
        <div className="strategy-badge">{strategy.toUpperCase()}</div>
      </div>
      
      <div className="goals-section">
        <h3>Active Goals</h3>
        {goals.length > 0 ? (
          <ul className="goals-list">
            {goals.map((goal, idx) => (
              <li key={idx} className="goal-item">{goal}</li>
            ))}
          </ul>
        ) : (
          <div className="no-data">No active goals</div>
        )}
      </div>
      
      <div className="beliefs-section">
        <h3>World Beliefs</h3>
        {Object.keys(beliefs).length > 0 ? (
          <div className="beliefs-grid">
            {Object.entries(beliefs).map(([key, value]) => (
              <div key={key} className="belief-item">
                <span className="belief-key">{key}:</span>
                <span className="belief-value">{String(value)}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="no-data">No beliefs recorded</div>
        )}
      </div>
    </div>
  );
}

export default WorldModelPanel;
