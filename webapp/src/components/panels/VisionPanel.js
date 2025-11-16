/**
 * @fileoverview This file contains the React component for the Vision Panel,
 * which displays information from the AGI's perception and vision systems.
 */

import React from 'react';
import './VisionPanel.css';

/**
 * A panel component that displays the AGI's visual perception and game state data.
 * It shows the current scene classification, detected objects, character vitals
 * (health, magicka, stamina), and other relevant environmental information.
 * @param {object} props - The component's props.
 * @param {object} props.data - The live AGI state data object.
 * @returns {React.Component} The rendered Vision Panel.
 */
function VisionPanel({ data }) {
  const perception = data.perception || {};
  const gameState = data.game_state || {};
  
  const sceneType = perception.scene_type || 'unknown';
  const objects = perception.objects_detected || [];
  const enemiesNearby = perception.enemies_nearby || false;
  const npcsNearby = perception.npcs_nearby || false;
  const lastVisionTime = perception.last_vision_time || 0;
  
  return (
    <div className="vision-panel">
      <h2>👁️ Vision & Perception</h2>
      
      <div className="scene-display">
        <h3>Current Scene</h3>
        <div className="scene-card">
          <div className="scene-icon">{getSceneIcon(sceneType)}</div>
          <div className="scene-name">{formatScene(sceneType)}</div>
        </div>
      </div>
      
      <div className="detection-grid">
        <div className={`detection-card ${enemiesNearby ? 'active danger' : 'inactive'}`}>
          <div className="detection-icon">⚔️</div>
          <div className="detection-label">Enemies</div>
          <div className="detection-status">{enemiesNearby ? 'DETECTED' : 'None'}</div>
        </div>
        
        <div className={`detection-card ${npcsNearby ? 'active friendly' : 'inactive'}`}>
          <div className="detection-icon">🧑</div>
          <div className="detection-label">NPCs</div>
          <div className="detection-status">{npcsNearby ? 'DETECTED' : 'None'}</div>
        </div>
        
        <div className={`detection-card ${gameState.in_combat ? 'active danger' : 'inactive'}`}>
          <div className="detection-icon">💥</div>
          <div className="detection-label">Combat</div>
          <div className="detection-status">{gameState.in_combat ? 'ACTIVE' : 'Peaceful'}</div>
        </div>
        
        <div className={`detection-card ${gameState.in_menu ? 'active neutral' : 'inactive'}`}>
          <div className="detection-icon">📋</div>
          <div className="detection-label">Menu</div>
          <div className="detection-status">{gameState.in_menu ? 'OPEN' : 'Closed'}</div>
        </div>
      </div>
      
      <div className="objects-section">
        <h3>Detected Objects</h3>
        <div className="objects-list">
          {objects.length > 0 ? (
            objects.map((obj, idx) => (
              <div key={idx} className="object-tag">
                {obj}
              </div>
            ))
          ) : (
            <div className="no-data">No objects detected</div>
          )}
        </div>
      </div>
      
      <div className="vision-stats">
        <div className="stat-item">
          <span className="stat-label">Last Vision Update:</span>
          <span className="stat-value">{lastVisionTime > 0 ? `${lastVisionTime.toFixed(2)}s ago` : 'N/A'}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Location:</span>
          <span className="stat-value">{gameState.location || 'Unknown'}</span>
        </div>
      </div>
      
      <div className="character-vitals">
        <h3>Character Vitals</h3>
        <div className="vitals-grid">
          <div className="vital-bar">
            <div className="vital-label">
              <span>Health</span>
              <span>{gameState.health || 100}%</span>
            </div>
            <div className="vital-progress">
              <div 
                className="vital-fill health"
                style={{ width: `${gameState.health || 100}%` }}
              ></div>
            </div>
          </div>
          
          <div className="vital-bar">
            <div className="vital-label">
              <span>Magicka</span>
              <span>{gameState.magicka || 100}%</span>
            </div>
            <div className="vital-progress">
              <div 
                className="vital-fill magicka"
                style={{ width: `${gameState.magicka || 100}%` }}
              ></div>
            </div>
          </div>
          
          <div className="vital-bar">
            <div className="vital-label">
              <span>Stamina</span>
              <span>{gameState.stamina || 100}%</span>
            </div>
            <div className="vital-progress">
              <div 
                className="vital-fill stamina"
                style={{ width: `${gameState.stamina || 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Returns an appropriate emoji icon for a given scene type.
 * @param {string} sceneType - The scene type string (e.g., 'outdoor_wilderness').
 * @returns {string} An emoji icon representing the scene.
 */
function getSceneIcon(sceneType) {
  const icons = {
    'outdoor_wilderness': '🌲',
    'outdoor_town': '🏘️',
    'indoor_dungeon': '⚔️',
    'indoor_building': '🏠',
    'combat': '💥',
    'dialogue': '💬',
    'unknown': '❓'
  };
  return icons[sceneType] || icons['unknown'];
}

/**
 * Formats a scene name string for display (e.g., "outdoor_wilderness" -> "Outdoor Wilderness").
 * @param {string} scene - The scene name string.
 * @returns {string} The formatted scene name.
 */
function formatScene(scene) {
  return scene.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

export default VisionPanel;
