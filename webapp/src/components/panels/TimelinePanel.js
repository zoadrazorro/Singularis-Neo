/**
 * @fileoverview This file contains the React component for the Timeline Panel,
 * which visualizes the AGI's consciousness metrics and recent events over time.
 */

import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './TimelinePanel.css';

/**
 * A panel component that displays a timeline of the AGI's performance and actions.
 * It features a chart showing the history of consciousness metrics (Coherence and Phi)
 * and a list of the most recent events (actions).
 * @param {object} props - The component's props.
 * @param {object} props.data - The live AGI state data object.
 * @returns {React.Component} The rendered Timeline Panel.
 */
function TimelinePanel({ data }) {
  const consciousness = data.consciousness || {};
  const performance = data.performance || {};
  const recentActions = data.recent_actions || [];
  
  const coherenceHistory = consciousness.history || [];
  const performanceHistory = performance.history || [];
  
  // Combine data for timeline chart
  const timelineData = coherenceHistory.map((ch, idx) => ({
    cycle: ch.cycle,
    coherence: ch.coherence,
    phi: ch.phi,
    planning: performanceHistory[idx]?.planning_time || 0
  }));
  
  return (
    <div className="timeline-panel">
      <h2>📈 Timeline</h2>
      
      {timelineData.length > 0 && (
        <div className="timeline-chart">
          <h3>Consciousness Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#444" />
              <XAxis dataKey="cycle" stroke="#aaa" />
              <YAxis stroke="#aaa" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #444' }}
                labelStyle={{ color: '#fff' }}
              />
              <Legend />
              <Line type="monotone" dataKey="coherence" stroke="#9b59b6" strokeWidth={2} name="Coherence" />
              <Line type="monotone" dataKey="phi" stroke="#3498db" strokeWidth={2} name="Φ (Phi)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      
      <div className="event-timeline">
        <h3>Recent Events</h3>
        <div className="events-list">
          {recentActions.slice(-15).reverse().map((action, idx) => (
            <div key={idx} className="event-item">
              <div className="event-marker"></div>
              <div className="event-content">
                <div className="event-action">{formatAction(action.name)}</div>
                <div className="event-meta">
                  <span>Cycle {action.cycle}</span>
                  <span>{formatTime(action.timestamp)}</span>
                </div>
              </div>
            </div>
          ))}
          {recentActions.length === 0 && (
            <div className="no-data">No events yet</div>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Formats an action name string for display (e.g., "move_forward" -> "Move Forward").
 * @param {string} action - The action name string.
 * @returns {string} The formatted action name.
 */
function formatAction(action) {
  return action.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Formats a Unix timestamp into a locale-specific time string.
 * @param {number} timestamp - The Unix timestamp.
 * @returns {string} The formatted time string.
 */
function formatTime(timestamp) {
  if (!timestamp) return '';
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString();
}

export default TimelinePanel;
