/**
 * @fileoverview This is the main component for the Singularis monitoring web application.
 * It manages the WebSocket connection, handles state for different monitoring modes
 * (learning vs. Skyrim AGI), and renders the appropriate dashboard.
 */

import React, { useState, useEffect } from 'react';
import './App.css';
import Dashboard from './components/Dashboard';
import SkyrimDashboard from './components/SkyrimDashboard';
import LifeOpsDashboard from './components/LifeOpsDashboard';

/**
 * The main application component.
 * @returns {React.Component} The rendered App component.
 */
function App() {
  const [progress, setProgress] = useState(null);
  const [connected, setConnected] = useState(false);
  const [ws, setWs] = useState(null);
  const [mode, setMode] = useState('lifeops'); // 'learning', 'skyrim', or 'lifeops'

  /**
   * Effect hook to manage the WebSocket connection.
   * It establishes a connection when the component mounts or when the `mode` changes.
   * It also handles reconnection logic on disconnection.
   */
  useEffect(() => {
    let reconnectTimeout;
    
    /**
     * Establishes and configures the WebSocket connection.
     */
    const connectWebSocket = () => {
      // Connect to WebSocket with mode parameter
      // Use current hostname to support both local and network access
      const hostname = window.location.hostname;
      let wsUrl;
      if (mode === 'skyrim') {
        wsUrl = `ws://${hostname}:5001?mode=skyrim`;
      } else if (mode === 'lifeops') {
        wsUrl = `ws://${hostname}:5001?mode=lifeops`;
      } else {
        wsUrl = `ws://${hostname}:5001`;
      }
      
      console.log('Connecting to WebSocket:', wsUrl);
      const websocket = new WebSocket(wsUrl);
      
      websocket.onopen = () => {
        console.log('WebSocket connected');
        setConnected(true);
      };
      
      websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setProgress(data);
        } catch (error) {
          console.error('Error parsing message:', error);
        }
      };
      
      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnected(false);
      };
      
      websocket.onclose = () => {
        console.log('WebSocket disconnected');
        setConnected(false);
        // Attempt to reconnect after 3 seconds
        reconnectTimeout = setTimeout(() => {
          console.log('Attempting to reconnect...');
          connectWebSocket();
        }, 3000);
      };
      
      setWs(websocket);
      
      return websocket;
    };
    
    const websocket = connectWebSocket();
    
    // Cleanup function to close the WebSocket connection on component unmount.
    return () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (websocket) {
        websocket.close();
      }
    };
  }, [mode]); // Re-connect when mode changes

  /**
   * Toggles the monitoring mode between 'learning', 'skyrim', and 'lifeops'.
   * This closes the existing WebSocket connection and triggers the useEffect
   * to reconnect with the new mode.
   */
  const toggleMode = () => {
    let nextMode;
    if (mode === 'lifeops') {
      nextMode = 'learning';
    } else if (mode === 'learning') {
      nextMode = 'skyrim';
    } else {
      nextMode = 'lifeops';
    }
    setMode(nextMode);
    setProgress(null); // Clear data when switching
    if (ws) {
      ws.close(); // Close existing connection
    }
  };

  /**
   * Gets the display title for the current mode
   */
  const getModeTitle = () => {
    if (mode === 'skyrim') return 'LifeOps - Skyrim Demo';
    if (mode === 'lifeops') return 'LifeOps';
    return 'LifeOps - Learning Monitor';
  };

  /**
   * Gets the toggle button text for the current mode
   */
  const getToggleText = () => {
    if (mode === 'learning') return '🎮 Skyrim Demo';
    if (mode === 'skyrim') return '🦉 LifeOps Dashboard';
    return '📚 Learning Monitor';
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🧠 Singularis {getModeTitle()}</h1>
        <div className="header-controls">
          <button className="mode-toggle" onClick={toggleMode}>
            {getToggleText()}
          </button>
          <div className="connection-status">
            <span className={`status-dot ${connected ? 'connected' : 'disconnected'}`}></span>
            <span>{connected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>
      </header>
      
      {progress ? (
        mode === 'skyrim' ? (
          <SkyrimDashboard data={progress} connected={connected} />
        ) : mode === 'lifeops' ? (
          <LifeOpsDashboard data={progress} connected={connected} />
        ) : (
          <Dashboard progress={progress} />
        )
      ) : (
        <div className="loading">
          <div className="spinner"></div>
          <p>Connecting to {mode === 'skyrim' ? 'Skyrim AGI' : mode === 'lifeops' ? 'LifeOps' : 'learning process'}...</p>
        </div>
      )}
      
      <footer className="App-footer">
        <p>From ETHICA: "The more the mind understands, the greater its power."</p>
      </footer>
    </div>
  );
}

export default App;
