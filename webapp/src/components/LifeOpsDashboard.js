/**
 * @fileoverview LifeOps Dashboard - Real-time life operations monitoring
 * Displays timeline events, patterns, suggestions, and health metrics
 */

import React from 'react';
import './LifeOpsDashboard.css';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

/**
 * LifeOps Dashboard Component
 * @param {object} props - Component props
 * @param {object} props.data - LifeOps data from WebSocket
 * @param {boolean} props.connected - Connection status
 */
function LifeOpsDashboard({ data, connected }) {
  const {
    timeline_events = [],
    patterns = [],
    suggestions = [],
    health_summary = {},
    productivity_stats = {},
    consciousness_metrics = {},
    last_update,
  } = data;

  // Calculate stats
  const todayEvents = timeline_events.filter(e => {
    const eventDate = new Date(e.timestamp);
    const today = new Date();
    return eventDate.toDateString() === today.toDateString();
  });

  const activeSuggestions = suggestions.filter(s => s.status === 'active');
  const recentPatterns = patterns.slice(0, 5);

  return (
    <div className="lifeops-dashboard">
      {/* Header Stats */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">📅</div>
          <div className="stat-content">
            <div className="stat-value">{todayEvents.length}</div>
            <div className="stat-label">Events Today</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🔍</div>
          <div className="stat-content">
            <div className="stat-value">{patterns.length}</div>
            <div className="stat-label">Patterns Detected</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">💡</div>
          <div className="stat-content">
            <div className="stat-value">{activeSuggestions.length}</div>
            <div className="stat-label">Active Suggestions</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">❤️</div>
          <div className="stat-content">
            <div className="stat-value">{health_summary.health_score || 'N/A'}</div>
            <div className="stat-label">Health Score</div>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="content-grid">
        {/* Timeline Events */}
        <div className="panel timeline-panel">
          <h2>📊 Recent Timeline Events</h2>
          <div className="timeline-list">
            {todayEvents.slice(0, 10).map((event, idx) => (
              <div key={idx} className="timeline-item">
                <div className="timeline-time">
                  {new Date(event.timestamp).toLocaleTimeString()}
                </div>
                <div className="timeline-content">
                  <div className="timeline-type">{getEventIcon(event.type)} {event.type}</div>
                  <div className="timeline-source">{event.source}</div>
                  {event.features && (
                    <div className="timeline-features">
                      {Object.entries(JSON.parse(event.features)).slice(0, 3).map(([key, value]) => (
                        <span key={key} className="feature-tag">
                          {key}: {typeof value === 'number' ? value.toFixed(1) : value}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                <div className="timeline-importance">
                  <div 
                    className="importance-bar" 
                    style={{ width: `${(event.importance || 0.5) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Patterns */}
        <div className="panel patterns-panel">
          <h2>🔍 Detected Patterns</h2>
          <div className="patterns-list">
            {recentPatterns.map((pattern, idx) => (
              <div key={idx} className="pattern-item">
                <div className="pattern-header">
                  <span className="pattern-name">{pattern.name || 'Unnamed Pattern'}</span>
                  <span className="pattern-confidence">
                    {((pattern.confidence || 0) * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="pattern-description">
                  {pattern.description || 'No description available'}
                </div>
                <div className="pattern-meta">
                  <span className="pattern-type">{pattern.type}</span>
                  <span className="pattern-occurrences">
                    {pattern.occurrences || 0} occurrences
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Suggestions */}
        <div className="panel suggestions-panel">
          <h2>💡 AGI Suggestions</h2>
          <div className="suggestions-list">
            {activeSuggestions.slice(0, 5).map((suggestion, idx) => (
              <div key={idx} className={`suggestion-item priority-${suggestion.priority || 'medium'}`}>
                <div className="suggestion-header">
                  <span className="suggestion-icon">{getSuggestionIcon(suggestion.type)}</span>
                  <span className="suggestion-title">{suggestion.title || 'Suggestion'}</span>
                </div>
                <div className="suggestion-message">
                  {suggestion.message}
                </div>
                <div className="suggestion-footer">
                  <span className="suggestion-time">
                    {suggestion.time_slot || 'Now'}
                  </span>
                  <div className="suggestion-actions">
                    <button className="btn-accept">✓ Accept</button>
                    <button className="btn-decline">✗ Decline</button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Health Metrics */}
        <div className="panel health-panel">
          <h2>❤️ Health Summary</h2>
          <div className="health-metrics">
            {health_summary.sleep_hours && (
              <div className="health-metric">
                <div className="metric-icon">😴</div>
                <div className="metric-content">
                  <div className="metric-label">Sleep</div>
                  <div className="metric-value">{health_summary.sleep_hours}h</div>
                </div>
              </div>
            )}
            {health_summary.steps && (
              <div className="health-metric">
                <div className="metric-icon">🚶</div>
                <div className="metric-content">
                  <div className="metric-label">Steps</div>
                  <div className="metric-value">{health_summary.steps.toLocaleString()}</div>
                </div>
              </div>
            )}
            {health_summary.heart_rate && (
              <div className="health-metric">
                <div className="metric-icon">💓</div>
                <div className="metric-content">
                  <div className="metric-label">Heart Rate</div>
                  <div className="metric-value">{health_summary.heart_rate} bpm</div>
                </div>
              </div>
            )}
            {health_summary.active_minutes && (
              <div className="health-metric">
                <div className="metric-icon">🏃</div>
                <div className="metric-content">
                  <div className="metric-label">Active</div>
                  <div className="metric-value">{health_summary.active_minutes} min</div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Productivity Stats */}
        <div className="panel productivity-panel">
          <h2>📈 Productivity</h2>
          <div className="productivity-stats">
            <div className="productivity-stat">
              <div className="stat-label">Tasks Completed</div>
              <div className="stat-value">{productivity_stats.tasks_completed || 0}</div>
            </div>
            <div className="productivity-stat">
              <div className="stat-label">Focus Time</div>
              <div className="stat-value">{productivity_stats.focus_time || 0}h</div>
            </div>
            <div className="productivity-stat">
              <div className="stat-label">Meetings</div>
              <div className="stat-value">{productivity_stats.meetings || 0}</div>
            </div>
            <div className="productivity-stat">
              <div className="stat-label">Calendar Gaps</div>
              <div className="stat-value">{productivity_stats.calendar_gaps || 0}</div>
            </div>
          </div>
        </div>

        {/* Consciousness Metrics */}
        <div className="panel consciousness-panel">
          <h2>🧠 AGI Consciousness</h2>
          <div className="consciousness-metrics">
            <div className="consciousness-metric">
              <div className="metric-label">Integration</div>
              <div className="metric-bar">
                <div 
                  className="metric-fill" 
                  style={{ width: `${(consciousness_metrics.integration || 0) * 100}%` }}
                />
              </div>
              <div className="metric-value">
                {((consciousness_metrics.integration || 0) * 100).toFixed(0)}%
              </div>
            </div>
            <div className="consciousness-metric">
              <div className="metric-label">Temporal Coherence</div>
              <div className="metric-bar">
                <div 
                  className="metric-fill" 
                  style={{ width: `${(consciousness_metrics.temporal || 0) * 100}%` }}
                />
              </div>
              <div className="metric-value">
                {((consciousness_metrics.temporal || 0) * 100).toFixed(0)}%
              </div>
            </div>
            <div className="consciousness-metric">
              <div className="metric-label">Lumen Balance</div>
              <div className="metric-bar">
                <div 
                  className="metric-fill" 
                  style={{ width: `${(consciousness_metrics.lumen_balance || 0) * 100}%` }}
                />
              </div>
              <div className="metric-value">
                {((consciousness_metrics.lumen_balance || 0) * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="lifeops-footer">
        <div className="footer-quote">
          "The unexamined life is not worth living." — Socrates
        </div>
        {last_update && (
          <div className="footer-update">
            Last update: {new Date(last_update).toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  );
}

// Helper functions
function getEventIcon(type) {
  const icons = {
    sleep: '😴',
    exercise: '🏃',
    meeting: '📅',
    task_completed: '✅',
    task_created: '📝',
    heart_rate: '💓',
    steps: '🚶',
    message: '💬',
    fall: '⚠️',
    room_enter: '🚪',
    room_exit: '🚪',
    default: '📍'
  };
  return icons[type] || icons.default;
}

function getSuggestionIcon(type) {
  const icons = {
    focus_block: '🎯',
    quick_win: '⚡',
    break: '☕',
    meeting_prep: '📋',
    context_switch: '🔄',
    energy_alignment: '⚡',
    default: '💡'
  };
  return icons[type] || icons.default;
}

export default LifeOpsDashboard;
