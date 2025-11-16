/**
 * @fileoverview This file contains the React component for the Performance Panel,
 * which displays metrics related to system timing, FPS, and overall performance health.
 */

import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './PerformancePanel.css';

/**
 * A panel component that displays performance metrics for the AGI.
 * It includes real-time values for various timings (planning, execution, vision),
 * total cycle time, and frames per second (FPS). It also shows a historical
 * performance chart and a summary of system health.
 * @param {object} props - The component's props.
 * @param {object} props.data - The live AGI state data object.
 * @returns {React.Component} The rendered Performance Panel.
 */
function PerformancePanel({ data }) {
  const performance = data.performance || {};
  const history = performance.history || [];
  
  const planningTime = performance.planning_time || 0;
  const executionTime = performance.execution_time || 0;
  const visionTime = performance.vision_time || 0;
  const totalCycleTime = performance.total_cycle_time || 0;
  const fps = performance.fps || 0;
  
  // Prepare chart data
  const chartData = history.map(h => ({
    cycle: h.cycle,
    planning: h.planning_time || 0,
    execution: h.execution_time || 0,
    vision: h.vision_time || 0,
    total: h.total_cycle_time || 0
  }));
  
  return (
    <div className="performance-panel">
      <h2>⚡ Performance Metrics</h2>
      
      <div className="performance-grid">
        <div className="perf-card">
          <div className="perf-value">{planningTime.toFixed(3)}s</div>
          <div className="perf-label">Planning Time</div>
          <div className="perf-bar">
            <div 
              className="perf-fill planning"
              style={{ width: `${Math.min((planningTime / 1.0) * 100, 100)}%` }}
            ></div>
          </div>
        </div>
        
        <div className="perf-card">
          <div className="perf-value">{executionTime.toFixed(3)}s</div>
          <div className="perf-label">Execution Time</div>
          <div className="perf-bar">
            <div 
              className="perf-fill execution"
              style={{ width: `${Math.min((executionTime / 1.0) * 100, 100)}%` }}
            ></div>
          </div>
        </div>
        
        <div className="perf-card">
          <div className="perf-value">{visionTime.toFixed(3)}s</div>
          <div className="perf-label">Vision Time</div>
          <div className="perf-bar">
            <div 
              className="perf-fill vision"
              style={{ width: `${Math.min((visionTime / 1.0) * 100, 100)}%` }}
            ></div>
          </div>
        </div>
        
        <div className="perf-card highlight">
          <div className="perf-value">{totalCycleTime.toFixed(3)}s</div>
          <div className="perf-label">Total Cycle Time</div>
          <div className="perf-bar">
            <div 
              className="perf-fill total"
              style={{ width: `${Math.min((totalCycleTime / 2.0) * 100, 100)}%` }}
            ></div>
          </div>
        </div>
        
        <div className="perf-card highlight">
          <div className="perf-value">{fps.toFixed(1)}</div>
          <div className="perf-label">FPS</div>
          <div className={`fps-indicator ${fps >= 30 ? 'good' : fps >= 15 ? 'medium' : 'poor'}`}>
            {fps >= 30 ? '🟢' : fps >= 15 ? '🟡' : '🔴'}
          </div>
        </div>
      </div>
      
      {chartData.length > 0 && (
        <div className="performance-chart">
          <h3>Performance History</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#444" />
              <XAxis dataKey="cycle" stroke="#aaa" />
              <YAxis stroke="#aaa" label={{ value: 'Time (s)', angle: -90, position: 'insideLeft' }} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #444' }}
                labelStyle={{ color: '#fff' }}
              />
              <Legend />
              <Line type="monotone" dataKey="planning" stroke="#3498db" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="execution" stroke="#e74c3c" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="vision" stroke="#f39c12" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="total" stroke="#9b59b6" strokeWidth={3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      
      <div className="performance-summary">
        <h3>System Health</h3>
        <div className="health-indicators">
          <div className={`indicator ${planningTime < 0.5 ? 'good' : planningTime < 1.0 ? 'medium' : 'poor'}`}>
            <span className="indicator-label">Planning</span>
            <span className="indicator-status">{planningTime < 0.5 ? 'Optimal' : planningTime < 1.0 ? 'Good' : 'Slow'}</span>
          </div>
          <div className={`indicator ${executionTime < 0.3 ? 'good' : executionTime < 0.6 ? 'medium' : 'poor'}`}>
            <span className="indicator-label">Execution</span>
            <span className="indicator-status">{executionTime < 0.3 ? 'Optimal' : executionTime < 0.6 ? 'Good' : 'Slow'}</span>
          </div>
          <div className={`indicator ${fps >= 30 ? 'good' : fps >= 15 ? 'medium' : 'poor'}`}>
            <span className="indicator-label">Frame Rate</span>
            <span className="indicator-status">{fps >= 30 ? 'Smooth' : fps >= 15 ? 'Acceptable' : 'Choppy'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default PerformancePanel;
