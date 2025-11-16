"""
Check learning progress in real-time.

This script provides a simple command-line utility to monitor the progress of a
running learning process by parsing its output log file (`learning_output.txt`).

Usage:
    python check_progress.py

The script reads the log file and uses regular expressions to extract key metrics,
including:
- The number of completed learning chunks out of the total.
- The average time taken per chunk.
- An estimated time remaining for the entire process.
- The average and latest "Coherentia" scores.
- The rate of "Ethical" chunks.

It then prints a formatted summary of these metrics to the console, providing a
quick and easy way to check the status and performance of the learning task
without needing to tail the full log file.
"""

import re
from pathlib import Path

def check_progress():
    """Reads the learning output log and prints a formatted progress summary."""
    
    output_file = Path("learning_output.txt")
    
    if not output_file.exists():
        print("No learning output file found. Is the process running?")
        return
    
    with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract chunk completions
    chunks_done = re.findall(r'\[(\d+)/(\d+)\] Processing.*chunk (\d+)\.\.\.', content)
    ok_lines = re.findall(r'\[OK\] Complete in ([\d.]+)s', content)
    coherentia_lines = re.findall(r'Coherentia: ([\d.]+)', content)
    ethical_lines = re.findall(r'Ethical: (True|False)', content)
    
    if not chunks_done:
        print("Process starting...")
        return
    
    # Get latest chunk info
    last_chunk = chunks_done[-1] if chunks_done else None
    
    if last_chunk:
        current, total, chunk_num = last_chunk
        print(f"=" * 60)
        print(f"LEARNING PROGRESS")
        print(f"=" * 60)
        print(f"Current chunk: {current}/{total}")
        print(f"Progress: {int(current)/int(total)*100:.1f}%")
        print(f"Chunks completed: {len(ok_lines)}")
        
        if ok_lines:
            avg_time = sum(float(t) for t in ok_lines) / len(ok_lines)
            remaining = int(total) - len(ok_lines)
            est_time = (remaining * avg_time) / 3600  # hours
            
            print(f"\nPerformance:")
            print(f"  Avg time/chunk: {avg_time:.1f}s")
            print(f"  Remaining chunks: {remaining}")
            print(f"  Est. time remaining: {est_time:.1f} hours")
        
        if coherentia_lines:
            avg_coherentia = sum(float(c) for c in coherentia_lines) / len(coherentia_lines)
            print(f"\nCoherentia:")
            print(f"  Average: {avg_coherentia:.3f}")
            print(f"  Latest: {coherentia_lines[-1]}")
        
        if ethical_lines:
            ethical_count = sum(1 for e in ethical_lines if e == "True")
            ethical_rate = ethical_count / len(ethical_lines) * 100
            print(f"\nEthical Status:")
            print(f"  Ethical chunks: {ethical_count}/{len(ethical_lines)}")
            print(f"  Ethical rate: {ethical_rate:.0f}%")
        
        print(f"\n" + "=" * 60)
        print(f"Process is running. Check again in a few minutes.")
        print(f"Output file: learning_output.txt")
        print(f"=" * 60)
    else:
        print("Waiting for first chunk to complete...")


if __name__ == "__main__":
    check_progress()
