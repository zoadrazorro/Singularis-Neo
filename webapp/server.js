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

// Create data directory if it doesn't exist
const dataDir = path.join(__dirname, '..', 'data');
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
  console.log('Created data directory:', dataDir);
}

// Path to the JSON file containing learning progress data.
const LEARNING_PROGRESS_PATH = path.join(__dirname, '..', 'learning_progress.json');

// Path to the JSON file containing the live state of the Skyrim AGI.
const SKYRIM_STATE_PATH = path.join(__dirname, '..', 'skyrim_agi_state.json');

// Path to the LifeOps data directory
const LIFEOPS_DATA_PATH = path.join(__dirname, '..', 'data', 'life_timeline.json');
const LIFEOPS_PATTERNS_PATH = path.join(__dirname, '..', 'data', 'patterns.json');
const LIFEOPS_SUGGESTIONS_PATH = path.join(__dirname, '..', 'data', 'suggestions.json');

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

/**
 * Parses LifeOps data from timeline, patterns, and suggestions files.
 * @returns {object} An object containing LifeOps monitoring data.
 */
function parseLifeOpsState() {
  try {
    // Read timeline events
    let timeline_events = [];
    if (fs.existsSync(LIFEOPS_DATA_PATH)) {
      const timelineData = JSON.parse(fs.readFileSync(LIFEOPS_DATA_PATH, 'utf-8'));
      timeline_events = timelineData.events || [];
    }

    // Read patterns
    let patterns = [];
    if (fs.existsSync(LIFEOPS_PATTERNS_PATH)) {
      const patternsData = JSON.parse(fs.readFileSync(LIFEOPS_PATTERNS_PATH, 'utf-8'));
      patterns = patternsData.patterns || [];
    }

    // Read suggestions
    let suggestions = [];
    if (fs.existsSync(LIFEOPS_SUGGESTIONS_PATH)) {
      const suggestionsData = JSON.parse(fs.readFileSync(LIFEOPS_SUGGESTIONS_PATH, 'utf-8'));
      suggestions = suggestionsData.suggestions || [];
    }

    // Calculate health summary from recent events
    const healthEvents = timeline_events.filter(e => 
      ['sleep', 'exercise', 'heart_rate', 'steps'].includes(e.type)
    );
    
    const health_summary = {
      sleep_hours: healthEvents.find(e => e.type === 'sleep')?.features?.duration || 0,
      steps: healthEvents.find(e => e.type === 'steps')?.features?.count || 0,
      heart_rate: healthEvents.find(e => e.type === 'heart_rate')?.features?.bpm || 0,
      active_minutes: healthEvents.find(e => e.type === 'exercise')?.features?.duration || 0,
      health_score: calculateHealthScore(healthEvents)
    };

    // Calculate productivity stats
    const productivityEvents = timeline_events.filter(e => 
      ['task_completed', 'meeting', 'work_session'].includes(e.type)
    );
    
    const productivity_stats = {
      tasks_completed: productivityEvents.filter(e => e.type === 'task_completed').length,
      focus_time: productivityEvents.filter(e => e.type === 'work_session')
        .reduce((sum, e) => sum + (e.features?.duration || 0), 0) / 60,
      meetings: productivityEvents.filter(e => e.type === 'meeting').length,
      calendar_gaps: suggestions.filter(s => s.type === 'focus_block').length
    };

    // Mock consciousness metrics (would come from AGI system)
    const consciousness_metrics = {
      integration: 0.78,
      temporal: 0.85,
      lumen_balance: 0.72
    };

    return {
      available: true,
      timeline_events: timeline_events.slice(-50), // Last 50 events
      patterns,
      suggestions,
      health_summary,
      productivity_stats,
      consciousness_metrics,
      last_update: new Date().toISOString()
    };
  } catch (error) {
    console.error('Error parsing LifeOps state:', error);
    return {
      available: false,
      error: error.message,
      timeline_events: [],
      patterns: [],
      suggestions: [],
      health_summary: {},
      productivity_stats: {},
      consciousness_metrics: {},
      last_update: new Date().toISOString()
    };
  }
}

/**
 * Calculates a health score from health events (0-100)
 * @param {Array} healthEvents - Array of health-related events
 * @returns {number} Health score
 */
function calculateHealthScore(healthEvents) {
  if (healthEvents.length === 0) return 0;
  
  let score = 0;
  let factors = 0;
  
  // Sleep score (7-9 hours optimal)
  const sleepEvent = healthEvents.find(e => e.type === 'sleep');
  if (sleepEvent?.features?.duration) {
    const hours = sleepEvent.features.duration;
    const sleepScore = hours >= 7 && hours <= 9 ? 100 : Math.max(0, 100 - Math.abs(8 - hours) * 15);
    score += sleepScore;
    factors++;
  }
  
  // Steps score (10k optimal)
  const stepsEvent = healthEvents.find(e => e.type === 'steps');
  if (stepsEvent?.features?.count) {
    const steps = stepsEvent.features.count;
    const stepsScore = Math.min(100, (steps / 10000) * 100);
    score += stepsScore;
    factors++;
  }
  
  // Heart rate score (60-100 resting optimal)
  const hrEvent = healthEvents.find(e => e.type === 'heart_rate');
  if (hrEvent?.features?.bpm) {
    const bpm = hrEvent.features.bpm;
    const hrScore = bpm >= 60 && bpm <= 100 ? 100 : Math.max(0, 100 - Math.abs(80 - bpm) * 2);
    score += hrScore;
    factors++;
  }
  
  return factors > 0 ? Math.round(score / factors) : 0;
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
 * @api {get} /api/lifeops Get LifeOps State
 * @apiName GetLifeOpsState
 * @apiGroup API
 *
 * @apiSuccess {Object} state The current LifeOps monitoring data.
 */
app.get('/api/lifeops', (req, res) => {
  const state = parseLifeOpsState();
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
  const isLifeOpsMode = url.includes('mode=lifeops');
  
  let mode = 'Learning Monitor';
  if (isSkyrimMode) mode = 'Skyrim AGI';
  if (isLifeOpsMode) mode = 'LifeOps';
  
  console.log(`Mode: ${mode}`);
  
  // Send initial data
  try {
    let initialData;
    if (isSkyrimMode) {
      initialData = parseSkyrimState();
    } else if (isLifeOpsMode) {
      initialData = parseLifeOpsState();
    } else {
      initialData = parseProgress();
    }
    ws.send(JSON.stringify(initialData));
  } catch (error) {
    console.error('Error sending initial data:', error);
  }
  
  // Set up interval to send updates
  const updateInterval = isSkyrimMode ? 1000 : isLifeOpsMode ? 5000 : 2000;
  const interval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      try {
        let data;
        if (isSkyrimMode) {
          data = parseSkyrimState();
        } else if (isLifeOpsMode) {
          data = parseLifeOpsState();
        } else {
          data = parseProgress();
        }
        ws.send(JSON.stringify(data));
      } catch (error) {
        console.error('Error sending update:', error);
      }
    }
  }, updateInterval); // Skyrim: 1s, LifeOps: 5s, Learning: 2s
  
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
console.log(`\nAPI Endpoints:`);
console.log(`  - GET /api/progress  - Learning progress`);
console.log(`  - GET /api/skyrim    - Skyrim AGI state`);
console.log(`  - GET /api/lifeops   - LifeOps data`);
console.log(`  - GET /api/health    - Health check`);
console.log('\nMonitoring:');
console.log('  - Learning:', LEARNING_PROGRESS_PATH);
console.log('  - Skyrim AGI:', SKYRIM_STATE_PATH);
console.log('  - LifeOps:', LIFEOPS_DATA_PATH);
console.log('\nModes: ?mode=skyrim or ?mode=lifeops');
