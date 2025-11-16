/**
 * @fileoverview This component renders the main dashboard for monitoring the
 * Singularis AGI's learning progress. It displays key metrics, charts, and a
 * table of recent learning chunks.
 */

import React from 'react';
import './Dashboard.css';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

/**
 * The main dashboard component for displaying learning progress.
 * @param {object} props - The component's props.
 * @param {object} props.progress - The learning progress data object received from the WebSocket.
 * @returns {React.Component} The rendered dashboard.
 */
function Dashboard({ progress }) {
  const {
    currentChunk,
    totalChunks,
    chunksCompleted,
    avgTime,
    avgCoherentia,
    ethicalRate,
    recentChunks,
    coherentiaHistory,
    timeHistory,
    isRunning,
    lastUpdate,
  } = progress;

  const progressPercent = (chunksCompleted / totalChunks) * 100;
  const remaining = totalChunks - chunksCompleted;
  const estTimeRemaining = (remaining * avgTime) / 3600; // hours

  return (
    <div className="dashboard">
      {/* Status Banner */}
      <div className="status-banner">
        <div className={`status-indicator ${isRunning ? 'running' : 'stopped'}`}>
          {isRunning ? '🚀 Learning in Progress' : '⏸️ Paused'}
        </div>
        {lastUpdate && (
          <div className="last-update">
            Last update: {new Date(lastUpdate).toLocaleTimeString()}
          </div>
        )}
      </div>

      {/* Progress Overview */}
      <div className="progress-section">
        <h2>Progress Overview</h2>
        <div className="progress-bar-container">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${progressPercent}%` }}
            >
              <span className="progress-text">{progressPercent.toFixed(1)}%</span>
            </div>
          </div>
          <div className="progress-details">
            <span>Chunk {chunksCompleted} / {totalChunks}</span>
            <span>{remaining} remaining</span>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-icon">⏱️</div>
          <div className="metric-content">
            <div className="metric-label">Avg Time/Chunk</div>
            <div className="metric-value">{avgTime.toFixed(1)}s</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">🎯</div>
          <div className="metric-content">
            <div className="metric-label">Avg Coherentia</div>
            <div className="metric-value">{avgCoherentia.toFixed(3)}</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">✅</div>
          <div className="metric-content">
            <div className="metric-label">Ethical Rate</div>
            <div className="metric-value">{(ethicalRate * 100).toFixed(0)}%</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">⏳</div>
          <div className="metric-content">
            <div className="metric-label">Est. Remaining</div>
            <div className="metric-value">{estTimeRemaining.toFixed(1)}h</div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="charts-section">
        <div className="chart-container">
          <h3>Coherentia Over Time</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={coherentiaHistory.slice(-50)}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis 
                dataKey="chunk" 
                stroke="rgba(255,255,255,0.7)"
                label={{ value: 'Chunk', position: 'insideBottom', offset: -5, fill: 'rgba(255,255,255,0.7)' }}
              />
              <YAxis 
                stroke="rgba(255,255,255,0.7)"
                domain={[0, 1]}
                label={{ value: 'Coherentia', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.7)' }}
              />
              <Tooltip 
                contentStyle={{ background: 'rgba(0,0,0,0.8)', border: 'none', borderRadius: '8px' }}
                labelStyle={{ color: 'white' }}
              />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#10b981" 
                strokeWidth={2}
                dot={false}
                name="Coherentia"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h3>Processing Time Per Chunk</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={timeHistory.slice(-50)}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis 
                dataKey="chunk" 
                stroke="rgba(255,255,255,0.7)"
                label={{ value: 'Chunk', position: 'insideBottom', offset: -5, fill: 'rgba(255,255,255,0.7)' }}
              />
              <YAxis 
                stroke="rgba(255,255,255,0.7)"
                label={{ value: 'Time (s)', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.7)' }}
              />
              <Tooltip 
                contentStyle={{ background: 'rgba(0,0,0,0.8)', border: 'none', borderRadius: '8px' }}
                labelStyle={{ color: 'white' }}
              />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#3b82f6" 
                strokeWidth={2}
                dot={false}
                name="Time (s)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Chunks Table */}
      <div className="recent-chunks-section">
        <h3>Recent Chunks</h3>
        <div className="table-container">
          <table className="chunks-table">
            <thead>
              <tr>
                <th>Chunk</th>
                <th>Time (s)</th>
                <th>Coherentia</th>
                <th>Consciousness</th>
                <th>Ethical</th>
              </tr>
            </thead>
            <tbody>
              {recentChunks.slice().reverse().map((chunk, idx) => (
                <tr key={idx}>
                  <td>#{chunk.chunk}</td>
                  <td>{chunk.time.toFixed(1)}</td>
                  <td className="coherentia-cell">{chunk.coherentia.toFixed(3)}</td>
                  <td>{chunk.consciousness.toFixed(3)}</td>
                  <td>
                    <span className={`ethical-badge ${chunk.ethical ? 'ethical' : 'not-ethical'}`}>
                      {chunk.ethical ? '✓ Yes' : '✗ No'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
