/**
 * @fileoverview This file sets up a WebSocket and HTTP server for real-time
 * monitoring of the Singularis AGI's learning progress and live state. It
 * serves data from JSON files to a web-based dashboard.
 */

const express = require('express');
const cors = require('cors');
const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 5000;
const WS_PORT = 5001;
const HOST = '0.0.0.0'; // Listen on all network interfaces

// Path to the JSON file containing learning progress data.
const LEARNING_PROGRESS_PATH = path.join(__dirname, '..', 'learning_progress.json');

// Path to the JSON file containing the live state of the Skyrim AGI.
const SKYRIM_STATE_PATH = path.join(__dirname, '..', 'skyrim_agi_state.json');

/**
 * Stores the current, in-memory state of the learning progress.
 * @type {object}
 */
let currentProgress = {
  currentChunk: 0,
  totalChunks: 240,
  chunksCompleted: 0,
  avgTime: 0,
  avgCoherentia: 0,
  ethicalRate: 0,
  recentChunks: [],
  coherentiaHistory: [],
  timeHistory: [],
  isRunning: false,
  lastUpdate: null,
};

/**
 * Parses the learning progress data from the `learning_progress.json` file.
 * @returns {object} An object containing the structured learning progress data.
 */
function parseProgress() {
  try {
    if (!fs.existsSync(LEARNING_PROGRESS_PATH)) {
      return { ...currentProgress, isRunning: false };
    }

    const data = JSON.parse(fs.readFileSync(LEARNING_PROGRESS_PATH, 'utf-8'));
    
    const chunks = data.chunks || [];
    const totalChunks = data.total_chunks || 240;
    const chunksCompleted = data.chunks_completed || chunks.length;
    const avgTime = data.avg_time || 0;
    const avgCoherentia = data.avg_coherentia || 0;
    const ethicalRate = data.ethical_rate || 0;
    
    // Get recent chunks (last 10)
    const recentChunks = chunks.slice(-10);
    
    // Build history for charts
    const coherentiaHistory = chunks.map(c => ({ chunk: c.chunk, value: c.coherentia }));
    const timeHistory = chunks.map(c => ({ chunk: c.chunk, value: c.time }));
    
    return {
      currentChunk: chunksCompleted,
      totalChunks,
      chunksCompleted,
      avgTime,
      avgCoherentia,
      ethicalRate,
      recentChunks,
      coherentiaHistory,
      timeHistory,
      isRunning: chunksCompleted < totalChunks,
      lastUpdate: data.last_update || new Date().toISOString(),
    };
  } catch (error) {
    console.error('Error parsing progress:', error);
    return { ...currentProgress, isRunning: false };
  }
}

/**
 * Parses the live state of the Skyrim AGI from the `skyrim_agi_state.json` file.
 * @returns {object} An object containing the live state data, or an error message
 * if the file is not available.
 */
function parseSkyrimState() {
  try {
    if (!fs.existsSync(SKYRIM_STATE_PATH)) {
      return { 
        available: false,
        message: 'Skyrim AGI not running or state file not found'
      };
    }

    const data = JSON.parse(fs.readFileSync(SKYRIM_STATE_PATH, 'utf-8'));
    
    return {
      available: true,
      ...data,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    console.error('Error parsing Skyrim state:', error);
    return { 
      available: false,
      error: error.message
    };
  }
}

// REST API endpoints

/**
 * @api {get} /api/progress Get Learning Progress
 * @apiName GetProgress
 * @apiGroup API
 *
 * @apiSuccess {Object} progress The current learning progress data.
 */
app.get('/api/progress', (req, res) => {
  const progress = parseProgress();
  res.json(progress);
});

/**
 * @api {get} /api/skyrim Get Skyrim AGI State
 * @apiName GetSkyrimState
 * @apiGroup API
 *
 * @apiSuccess {Object} state The current live state of the Skyrim AGI.
 */
app.get('/api/skyrim', (req, res) => {
  const state = parseSkyrimState();
  res.json(state);
});

/**
 * @api {get} /api/health Health Check
 * @apiName GetHealth
 * @apiGroup API
 *
 * @apiSuccess {String} status The status of the server ('ok').
 * @apiSuccess {String} timestamp The current ISO timestamp.
 */
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Start HTTP server
app.listen(PORT, HOST, () => {
  console.log(`HTTP server running on:`);
  console.log(`  - Local:   http://localhost:${PORT}`);
  console.log(`  - Network: http://<YOUR_LOCAL_IP>:${PORT}`);
  console.log(`\nTo find your local IP, run: ipconfig (Windows) or ifconfig (Mac/Linux)`);
});

// WebSocket server setup
const wss = new WebSocket.Server({ 
  host: HOST,
  port: WS_PORT,
  perMessageDeflate: false,
  clientTracking: true
});

/**
 * Handles new WebSocket connections.
 * Determines the client's requested mode (learning monitor or Skyrim AGI)
 * and sends the appropriate data at regular intervals.
 */
wss.on('connection', (ws, req) => {
  console.log('Client connected');
  
  // Determine which mode to use based on query parameter
  const url = req.url || '';
  const isSkyrimMode = url.includes('mode=skyrim');
  
  console.log(`Mode: ${isSkyrimMode ? 'Skyrim AGI' : 'Learning Monitor'}`);
  
  // Send initial data
  try {
    const initialData = isSkyrimMode ? parseSkyrimState() : parseProgress();
    ws.send(JSON.stringify(initialData));
  } catch (error) {
    console.error('Error sending initial data:', error);
  }
  
  // Set up interval to send updates
  const interval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      try {
        const data = isSkyrimMode ? parseSkyrimState() : parseProgress();
        ws.send(JSON.stringify(data));
      } catch (error) {
        console.error('Error sending update:', error);
      }
    }
  }, isSkyrimMode ? 1000 : 2000); // Skyrim updates every 1s, learning every 2s
  
  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
  });
  
  ws.on('close', () => {
    console.log('Client disconnected');
    clearInterval(interval);
  });
  
  // Send ping to keep connection alive
  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.ping();
    }
  }, 30000);
  
  ws.on('close', () => {
    clearInterval(pingInterval);
  });
});

console.log(`\nWebSocket server running on:`);
console.log(`  - Local:   ws://localhost:${WS_PORT}`);
console.log(`  - Network: ws://<YOUR_LOCAL_IP>:${WS_PORT}`);
console.log('\nMonitoring:');
console.log('  - Learning:', LEARNING_PROGRESS_PATH);
console.log('  - Skyrim AGI:', SKYRIM_STATE_PATH);
