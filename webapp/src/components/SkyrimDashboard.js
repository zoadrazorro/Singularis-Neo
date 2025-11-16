/**
 * @fileoverview This file contains the main component for the real-time AGI dashboard
 * for the Skyrim application. It provides a multi-tab interface for monitoring
 * various subsystems of the AGI.
 */

import React, { useState } from 'react';
import './SkyrimDashboard.css';
import ConsciousnessPanel from './panels/ConsciousnessPanel';
import PerformancePanel from './panels/PerformancePanel';
import ActionPanel from './panels/ActionPanel';
import VisionPanel from './panels/VisionPanel';
import LLMPanel from './panels/LLMPanel';
import WorldModelPanel from './panels/WorldModelPanel';
import StatsPanel from './panels/StatsPanel';
import TimelinePanel from './panels/TimelinePanel';

/**
 * The main component for the Skyrim AGI real-time dashboard.
 * It features a tabbed interface to switch between different monitoring panels.
 * @param {object} props - The component's props.
 * @param {object} props.data - The live data object for the AGI state.
 * @param {boolean} props.connected - The WebSocket connection status.
 * @returns {React.Component} The rendered dashboard component.
 */
function SkyrimDashboard({ data, connected }) {
  const [activeTab, setActiveTab] = useState('overview');
  const [layout, setLayout] = useState('grid'); // 'grid' or 'focus'

  if (!data) {
    return (
      <div className="skyrim-dashboard loading">
        <div className="loading-spinner"></div>
        <p>Waiting for AGI data...</p>
      </div>
    );
  }

  const tabs = [
    { id: 'overview', label: '📊 Overview', icon: '📊' },
    { id: 'consciousness', label: '🧠 Consciousness', icon: '🧠' },
    { id: 'performance', label: '⚡ Performance', icon: '⚡' },
    { id: 'llm', label: '🤖 LLM Systems', icon: '🤖' },
    { id: 'vision', label: '👁️ Vision', icon: '👁️' },
    { id: 'world', label: '🌍 World Model', icon: '🌍' },
    { id: 'timeline', label: '📈 Timeline', icon: '📈' },
  ];

  return (
    <div className="skyrim-dashboard">
      {/* Header with Status */}
      <div className="dashboard-header">
        <div className="header-left">
          <h1>🎮 Skyrim AGI Dashboard</h1>
          <div className="session-info">
            <span className="session-id">Session: {data.session_id?.slice(0, 12) || 'N/A'}</span>
            <span className="cycle-count">Cycle: {data.cycle || 0}</span>
            <span className="uptime">Uptime: {formatDuration(data.uptime || 0)}</span>
          </div>
        </div>
        
        <div className="header-right">
          <div className={`connection-indicator ${connected ? 'connected' : 'disconnected'}`}>
            <span className="pulse-dot"></span>
            <span>{connected ? 'LIVE' : 'DISCONNECTED'}</span>
          </div>
          
          <div className="layout-toggle">
            <button 
              className={layout === 'grid' ? 'active' : ''} 
              onClick={() => setLayout('grid')}
              title="Grid Layout"
            >
              ⊞
            </button>
            <button 
              className={layout === 'focus' ? 'active' : ''} 
              onClick={() => setLayout('focus')}
              title="Focus Layout"
            >
              ⊡
            </button>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="tab-navigation">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="tab-icon">{tab.icon}</span>
            <span className="tab-label">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Main Content Area */}
      <div className={`dashboard-content layout-${layout}`}>
        {activeTab === 'overview' && (
          <OverviewLayout data={data} />
        )}
        
        {activeTab === 'consciousness' && (
          <ConsciousnessPanel data={data} />
        )}
        
        {activeTab === 'performance' && (
          <PerformancePanel data={data} />
        )}
        
        {activeTab === 'llm' && (
          <LLMPanel data={data} />
        )}
        
        {activeTab === 'vision' && (
          <VisionPanel data={data} />
        )}
        
        {activeTab === 'world' && (
          <WorldModelPanel data={data} />
        )}
        
        {activeTab === 'timeline' && (
          <TimelinePanel data={data} />
        )}
      </div>
    </div>
  );
}

/**
 * A component that renders a multi-panel grid view for the 'Overview' tab.
 * @param {object} props - The component's props.
 * @param {object} props.data - The live AGI state data.
 * @returns {React.Component} The rendered overview layout.
 */
function OverviewLayout({ data }) {
  return (
    <div className="overview-layout">
      {/* Top Row - Critical Status */}
      <div className="overview-row row-critical">
        <StatusCard data={data} />
        <ActionCard data={data} />
        <CoherenceCard data={data} />
      </div>

      {/* Middle Row - System Health */}
      <div className="overview-row row-health">
        <LLMStatusCard data={data} />
        <PerformanceCard data={data} />
        <DiversityCard data={data} />
      </div>

      {/* Bottom Row - Details */}
      <div className="overview-row row-details">
        <RecentActionsCard data={data} />
        <MetricsCard data={data} />
      </div>
    </div>
  );
}

/**
 * A mini-card for the overview screen, displaying the AGI's current in-game status.
 * @param {object} props - The component's props.
 * @param {object} props.data - The live AGI state data.
 * @returns {React.Component} The rendered status card.
 */
function StatusCard({ data }) {
  const health = data.game_state?.health || 0;
  const scene = data.perception?.scene_type || 'unknown';
  const inCombat = data.game_state?.in_combat || false;
  
  return (
    <div className="overview-card status-card">
      <h3>🎯 Status</h3>
      <div className="status-grid">
        <div className="status-item">
          <span className="label">Health</span>
          <div className="health-bar">
            <div className="health-fill" style={{ width: `${health}%` }}></div>
            <span className="health-text">{health}%</span>
          </div>
        </div>
        <div className="status-item">
          <span className="label">Scene</span>
          <span className="value">{formatSceneName(scene)}</span>
        </div>
        <div className="status-item">
          <span className="label">Combat</span>
          <span className={`value ${inCombat ? 'danger' : 'safe'}`}>
            {inCombat ? '⚔️ Active' : '🕊️ Peaceful'}
          </span>
        </div>
      </div>
    </div>
  );
}

/**
 * A mini-card for the overview screen, displaying the AGI's current and last actions.
 * @param {object} props - The component's props.
 * @param {object} props.data - The live AGI state data.
 * @returns {React.Component} The rendered action card.
 */
function ActionCard({ data }) {
  const currentAction = data.current_action || 'idle';
  const lastAction = data.last_action || 'none';
  const actionSource = data.action_source || 'unknown';
  
  return (
    <div className="overview-card action-card">
      <h3>🎬 Action</h3>
      <div className="action-display">
        <div className="current-action">{formatActionName(currentAction)}</div>
        <div className="action-details">
          <span className="detail-label">Source:</span>
          <span className="detail-value">{actionSource}</span>
        </div>
        <div className="action-details">
          <span className="detail-label">Previous:</span>
          <span className="detail-value">{formatActionName(lastAction)}</span>
        </div>
      </div>
    </div>
  );
}

/**
 * A mini-card for the overview screen, displaying key consciousness metrics.
 * @param {object} props - The component's props.
 * @param {object} props.data - The live AGI state data.
 * @returns {React.Component} The rendered coherence card.
 */
function CoherenceCard({ data }) {
  const coherence = data.consciousness?.coherence || 0;
  const phi = data.consciousness?.phi || 0;
  const trend = data.consciousness?.trend || 'stable';
  
  return (
    <div className="overview-card coherence-card">
      <h3>🧠 Consciousness</h3>
      <div className="coherence-display">
        <div className="coherence-value">
          <span className="value-large">{coherence.toFixed(3)}</span>
          <span className="value-label">Coherence 𝒞</span>
        </div>
        <div className="coherence-metrics">
          <div className="metric">
            <span className="metric-label">Φ:</span>
            <span className="metric-value">{phi.toFixed(3)}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Trend:</span>
            <span className={`metric-value trend-${trend}`}>
              {getTrendIcon(trend)} {trend}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * A mini-card for the overview screen, displaying the status of LLM systems.
 * @param {object} props - The component's props.
 * @param {object} props.data - The live AGI state data.
 * @returns {React.Component} The rendered LLM status card.
 */
function LLMStatusCard({ data }) {
  const llm = data.llm_status || {};
  const cloudActive = llm.cloud_active || 0;
  const localActive = llm.local_active || 0;
  const totalCalls = llm.total_calls || 0;
  
  return (
    <div className="overview-card llm-status-card">
      <h3>🤖 LLM Systems</h3>
      <div className="llm-grid">
        <div className="llm-item">
          <span className="llm-label">☁️ Cloud</span>
          <span className="llm-value">{cloudActive} active</span>
        </div>
        <div className="llm-item">
          <span className="llm-label">💻 Local</span>
          <span className="llm-value">{localActive} active</span>
        </div>
        <div className="llm-item">
          <span className="llm-label">📞 Calls</span>
          <span className="llm-value">{totalCalls}</span>
        </div>
      </div>
    </div>
  );
}

/**
 * A mini-card for the overview screen, displaying performance metrics.
 * @param {object} props - The component's props.
 * @param {object} props.data - The live AGI state data.
 * @returns {React.Component} The rendered performance card.
 */
function PerformanceCard({ data }) {
  const fps = data.performance?.fps || 0;
  const planTime = data.performance?.planning_time || 0;
  const execTime = data.performance?.execution_time || 0;
  
  return (
    <div className="overview-card performance-card">
      <h3>⚡ Performance</h3>
      <div className="perf-metrics">
        <div className="perf-item">
          <span className="perf-value">{planTime.toFixed(2)}s</span>
          <span className="perf-label">Planning</span>
        </div>
        <div className="perf-item">
          <span className="perf-value">{execTime.toFixed(2)}s</span>
          <span className="perf-label">Execution</span>
        </div>
        <div className="perf-item">
          <span className="perf-value">{fps.toFixed(1)}</span>
          <span className="perf-label">FPS</span>
        </div>
      </div>
    </div>
  );
}

/**
 * A mini-card for the overview screen, displaying action diversity metrics.
 * @param {object} props - The component's props.
 * @param {object} props.data - The live AGI state data.
 * @returns {React.Component} The rendered diversity card.
 */
function DiversityCard({ data }) {
  const diversity = data.diversity || {};
  const score = diversity.score || 0;
  const uniqueActions = diversity.unique_actions || 0;
  const varietyRate = diversity.variety_rate || 0;
  
  return (
    <div className="overview-card diversity-card">
      <h3>🎨 Diversity</h3>
      <div className="diversity-metrics">
        <div className="diversity-score">
          <span className="score-value">{(score * 100).toFixed(0)}%</span>
          <span className="score-label">Diversity Score</span>
        </div>
        <div className="diversity-stats">
          <div>{uniqueActions} unique actions</div>
          <div>{(varietyRate * 100).toFixed(0)}% variety rate</div>
        </div>
      </div>
    </div>
  );
}

/**
 * A mini-card for the overview screen, displaying a list of recent actions.
 * @param {object} props - The component's props.
 * @param {object} props.data - The live AGI state data.
 * @returns {React.Component} The rendered recent actions card.
 */
function RecentActionsCard({ data }) {
  const actions = data.recent_actions || [];
  
  return (
    <div className="overview-card recent-actions-card">
      <h3>📜 Recent Actions</h3>
      <div className="actions-list">
        {actions.slice(0, 5).map((action, idx) => (
          <div key={idx} className="action-item">
            <span className="action-name">{formatActionName(action.name)}</span>
            <span className="action-time">{formatTime(action.timestamp)}</span>
          </div>
        ))}
        {actions.length === 0 && (
          <div className="no-data">No recent actions</div>
        )}
      </div>
    </div>
  );
}

/**
 * A mini-card for the overview screen, displaying key session metrics.
 * @param {object} props - The component's props.
 * @param {object} props.data - The live AGI state data.
 * @returns {React.Component} The rendered metrics card.
 */
function MetricsCard({ data }) {
  const stats = data.stats || {};
  
  return (
    <div className="overview-card metrics-card">
      <h3>📊 Session Metrics</h3>
      <div className="metrics-grid">
        <div className="metric-row">
          <span className="metric-label">Success Rate:</span>
          <span className="metric-value">{(stats.success_rate * 100 || 0).toFixed(1)}%</span>
        </div>
        <div className="metric-row">
          <span className="metric-label">RL Actions:</span>
          <span className="metric-value">{stats.rl_actions || 0}</span>
        </div>
        <div className="metric-row">
          <span className="metric-label">LLM Actions:</span>
          <span className="metric-value">{stats.llm_actions || 0}</span>
        </div>
        <div className="metric-row">
          <span className="metric-label">Heuristic:</span>
          <span className="metric-value">{stats.heuristic_actions || 0}</span>
        </div>
      </div>
    </div>
  );
}

// Helper Functions

/**
 * Formats a duration in seconds into a "Xm Ys" string.
 * @param {number} seconds - The duration in seconds.
 * @returns {string} The formatted duration string.
 */
function formatDuration(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}m ${secs}s`;
}

/**
 * Formats a scene name string for display (e.g., "word_word" -> "Word Word").
 * @param {string} scene - The scene name.
 * @returns {string} The formatted scene name.
 */
function formatSceneName(scene) {
  return scene.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Formats an action name string for display.
 * @param {string} action - The action name.
 * @returns {string} The formatted action name.
 */
function formatActionName(action) {
  return action.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Returns an icon corresponding to a trend string.
 * @param {string} trend - The trend string ('increasing', 'decreasing', 'stable').
 * @returns {string} An emoji icon.
 */
function getTrendIcon(trend) {
  switch (trend) {
    case 'increasing': return '📈';
    case 'decreasing': return '📉';
    case 'stable': return '➡️';
    default: return '•';
  }
}

/**
 * Formats a timestamp into a locale-specific time string.
 * @param {string} timestamp - The ISO timestamp string.
 * @returns {string} The formatted time string.
 */
function formatTime(timestamp) {
  if (!timestamp) return 'N/A';
  const date = new Date(timestamp);
  return date.toLocaleTimeString();
}

export default SkyrimDashboard;
