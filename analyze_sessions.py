"""
This script analyzes the Markdown session logs for the Skyrim AGI to extract key
performance metrics and identify trends over time. It is designed to be run from
the root of the Singularis project directory.

The script performs the following actions:
1.  Locates all session log files (e.g., `sessions/skyrim_agi_*.md`).
2.  Iterates through each file and uses regular expressions to parse key data points,
    including:
    - Session duration and total cycles.
    - Activation of the expert rule system.
    - LLM planning time statistics (average and max).
    - "STUCK" detections and visual similarity scores.
    - System coherence metrics.
3.  Aggregates this data across all sessions.
4.  Prints a "Trends Analysis" report to the console, focusing on the last 15
    sessions to highlight recent changes in performance.
5.  Calculates and displays overall statistics for the entire history of sessions.
6.  Generates and prints key insights and critical warnings based on the analysis,
    such as a high rate of "STUCK" states or a failure of the rule system to activate.

This analysis is crucial for debugging and for understanding the AGI's performance
evolution, helping to pinpoint regressions or confirm improvements.
"""

import re
import glob
from datetime import datetime
import os

# Change to correct directory
os.chdir('d:\\Projects\\Singularis')

sessions = sorted(glob.glob('sessions\\skyrim_agi_*.md'))
print(f'Analyzing {len(sessions)} sessions...\n')

data = []
for session_file in sessions:
    with open(session_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    info = {'file': session_file.split('\\')[-1]}
    
    # Time
    time_match = re.search(r'Start Time.*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', content)
    if time_match:
        info['start_time'] = datetime.strptime(time_match.group(1), '%Y-%m-%d %H:%M:%S')
    
    # Duration and cycles
    duration_match = re.search(r'Duration.*?(\d+\.?\d*) minutes', content)
    cycles_match = re.search(r'Total Cycles.*?(\d+)', content)
    if duration_match:
        info['duration_min'] = float(duration_match.group(1))
    if cycles_match:
        info['cycles'] = int(cycles_match.group(1))
    
    # Rules
    rules_matches = re.findall(r'Applicable Rules: (\d+)', content)
    if rules_matches:
        info['max_rules'] = max(int(r) for r in rules_matches)
        info['total_rules'] = sum(int(r) for r in rules_matches)
    else:
        info['max_rules'] = 0
        info['total_rules'] = 0
    
    # Planning time
    planning_matches = re.findall(r'planning_time.\s*:\s*(\d+\.?\d*)', content)
    if planning_matches:
        times = [float(t) for t in planning_matches]
        info['avg_planning'] = sum(times) / len(times)
        info['max_planning'] = max(times)
    
    # Visual similarity / STUCK
    stuck_count = len(re.findall(r'STUCK', content, re.IGNORECASE))
    similarity_matches = re.findall(r'similarity.\s*[=:]?\s*(\d+\.?\d+)', content, re.IGNORECASE)
    if similarity_matches:
        sims = [float(s) for s in similarity_matches if float(s) <= 1.0]
        if sims:
            info['avg_similarity'] = sum(sims) / len(sims)
            info['max_similarity'] = max(sims)
    info['stuck_count'] = stuck_count
    
    # Coherence
    coherence_matches = re.findall(r'coherence.\s*[=:]?\s*(\d+\.?\d+)', content, re.IGNORECASE)
    if coherence_matches:
        cohs = [float(c) for c in coherence_matches if float(c) <= 1.0]
        if cohs:
            info['avg_coherence'] = sum(cohs) / len(cohs)
    
    data.append(info)

# Print trends
print('='*80)
print('SESSION TRENDS ANALYSIS')
print('='*80)

print('\n1. DURATION & SCALE TRENDS (Last 15 sessions)')
print('-'*80)
for i, d in enumerate(data[-15:], len(data)-14):
    dur = d.get('duration_min', 0)
    cyc = d.get('cycles', 0)
    print(f'{i:2d}. {dur:5.1f}min | {cyc:3d} cycles | {d.get("file", "unknown")[:40]}')

print('\n2. RULE SYSTEM ACTIVATION (Last 15 sessions)')
print('-'*80)
print('Session | Max Rules | Total Rules | File')
print('-'*80)
for i, d in enumerate(data[-15:], len(data)-14):
    max_r = d.get('max_rules', 0)
    tot_r = d.get('total_rules', 0)
    indicator = '⚠️ ZERO' if tot_r == 0 else '✓'
    print(f'{i:2d}     | {max_r:9d} | {tot_r:11d} | {indicator} {d.get("file", "unknown")[:30]}')

print('\n3. PLANNING TIME TRENDS (Last 15 sessions)')
print('-'*80)
for i, d in enumerate(data[-15:], len(data)-14):
    avg_p = d.get('avg_planning', 0)
    max_p = d.get('max_planning', 0)
    if avg_p > 0:
        trend = '🐌' if avg_p > 20 else '⚡' if avg_p < 5 else '→'
        print(f'{i:2d}. {trend} Avg: {avg_p:6.2f}s | Max: {max_p:6.2f}s | {d.get("file", "unknown")[:30]}')

print('\n4. STUCK DETECTION (Last 15 sessions)')
print('-'*80)
for i, d in enumerate(data[-15:], len(data)-14):
    stuck = d.get('stuck_count', 0)
    sim = d.get('max_similarity', 0)
    indicator = '🚫 STUCK!' if stuck > 0 or sim > 0.95 else '✓ moving'
    print(f'{i:2d}. {indicator} | mentions: {stuck:2d} | max sim: {sim:.3f} | {d.get("file", "unknown")[:25]}')

print('\n5. COHERENCE TRENDS (Last 15 sessions)')
print('-'*80)
for i, d in enumerate(data[-15:], len(data)-14):
    coh = d.get('avg_coherence', 0)
    trend = '📈' if coh > 0.4 else '📉' if coh < 0.25 else '→'
    print(f'{i:2d}. {trend} Avg coherence: {coh:.3f} | {d.get("file", "unknown")[:35]}')

# Time-based trends
print('\n6. TEMPORAL TRENDS')
print('-'*80)
print('\nEarly Sessions (1-10):')
early = data[:10]
early_dur = sum(d.get('duration_min', 0) for d in early) / len(early)
early_cycles = sum(d.get('cycles', 0) for d in early) / len(early)
early_planning = sum(d.get('avg_planning', 0) for d in early if 'avg_planning' in d) / max(1, sum(1 for d in early if 'avg_planning' in d))
print(f'  Avg Duration: {early_dur:.1f} min')
print(f'  Avg Cycles: {early_cycles:.1f}')
print(f'  Avg Planning Time: {early_planning:.2f}s')

print('\nMiddle Sessions (20-30):')
if len(data) >= 30:
    middle = data[19:30]
    mid_dur = sum(d.get('duration_min', 0) for d in middle) / len(middle)
    mid_cycles = sum(d.get('cycles', 0) for d in middle) / len(middle)
    mid_planning = sum(d.get('avg_planning', 0) for d in middle if 'avg_planning' in d) / max(1, sum(1 for d in middle if 'avg_planning' in d))
    print(f'  Avg Duration: {mid_dur:.1f} min')
    print(f'  Avg Cycles: {mid_cycles:.1f}')
    print(f'  Avg Planning Time: {mid_planning:.2f}s')

print('\nRecent Sessions (Last 10):')
recent = data[-10:]
recent_dur = sum(d.get('duration_min', 0) for d in recent) / len(recent)
recent_cycles = sum(d.get('cycles', 0) for d in recent) / len(recent)
recent_planning = sum(d.get('avg_planning', 0) for d in recent if 'avg_planning' in d) / max(1, sum(1 for d in recent if 'avg_planning' in d))
print(f'  Avg Duration: {recent_dur:.1f} min')
print(f'  Avg Cycles: {recent_cycles:.1f}')
print(f'  Avg Planning Time: {recent_planning:.2f}s')

# Summary statistics
print('\n' + '='*80)
print('OVERALL STATISTICS')
print('='*80)
total_sessions = len(data)
total_duration = sum(d.get('duration_min', 0) for d in data)
total_cycles = sum(d.get('cycles', 0) for d in data)
sessions_with_rules = sum(1 for d in data if d.get('total_rules', 0) > 0)
sessions_stuck = sum(1 for d in data if d.get('stuck_count', 0) > 0 or d.get('max_similarity', 0) > 0.95)
avg_planning_all = sum(d.get('avg_planning', 0) for d in data if 'avg_planning' in d) / max(1, sum(1 for d in data if 'avg_planning' in d))

print(f'\nTotal Sessions: {total_sessions}')
print(f'Total Duration: {total_duration:.1f} minutes ({total_duration/60:.1f} hours)')
print(f'Total Cycles: {total_cycles}')
print(f'Average Cycles/Session: {total_cycles/total_sessions:.1f}')
print(f'\nSessions with Rules Fired: {sessions_with_rules} / {total_sessions} ({sessions_with_rules/total_sessions*100:.1f}%)')
print(f'Sessions with STUCK Status: {sessions_stuck} / {total_sessions} ({sessions_stuck/total_sessions*100:.1f}%)')
print(f'\nAverage Planning Time: {avg_planning_all:.2f}s')
print(f'Latest 10 Sessions Duration: {sum(d.get("duration_min", 0) for d in data[-10:]):.1f} minutes')

# Key insights
print('\n' + '='*80)
print('KEY INSIGHTS')
print('='*80)

if sessions_with_rules == 0:
    print('\n⚠️  CRITICAL: Zero rule activations across ALL sessions!')
    print('   The expert rule system is not firing. This explains stuck behavior.')

if sessions_stuck > total_sessions * 0.3:
    print(f'\n⚠️  HIGH STUCK RATE: {sessions_stuck/total_sessions*100:.0f}% of sessions show stuck behavior')
    print('   Visual similarity >0.95 or STUCK mentions detected')

if avg_planning_all > 15:
    print(f'\n⚠️  SLOW PLANNING: Average {avg_planning_all:.1f}s planning time')
    print('   LLM reasoning is bottleneck. Rules could provide fast path.')

if recent_planning < early_planning * 0.8:
    print(f'\n✓ IMPROVEMENT: Planning time reduced from {early_planning:.1f}s to {recent_planning:.1f}s')
    print(f'  ({(1 - recent_planning/early_planning)*100:.0f}% faster)')
