#!/usr/bin/env python3
"""Test all available cats."""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import time
import numpy as np
from training import train_bot
from cat_env import make_env

# Only the 5 known cats we can test
cats = ['batmeow', 'mittens', 'paotsin', 'peekaboo', 'squiddyboi']

print("=" * 70)
print("CATBOT PERFORMANCE TEST - 5 KNOWN CATS")
print("=" * 70)

results = {}

for cat_name in cats:
    print(f"\n{'='*70}")
    print(f"Testing: {cat_name.upper()}")
    print(f"{'='*70}")
    
    # Train
    print("Training...")
    start = time.time()
    q_table = train_bot(cat_name, render=-1)
    train_time = time.time() - start
    print(f"✓ Training completed in {train_time:.2f}s")
    
    # Test 10 trials
    print("Testing (10 trials)...")
    successes = 0
    steps_list = []
    
    for trial in range(10):
        env = make_env(cat_type=cat_name)
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 60:
            action = int(np.argmax(q_table[state]))
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        
        if terminated:
            successes += 1
            steps_list.append(steps)
        env.close()
    
    success_rate = (successes / 10) * 100
    avg_steps = np.mean(steps_list) if steps_list else float('inf')
    min_steps = min(steps_list) if steps_list else 0
    max_steps = max(steps_list) if steps_list else 0
    
    results[cat_name] = {
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'min_steps': min_steps,
        'max_steps': max_steps,
        'train_time': train_time,
        'successes': successes
    }
    
    print(f"✓ Success Rate: {success_rate:.0f}% ({successes}/10 trials)")
    if avg_steps != float('inf'):
        print(f"✓ Average Steps: {avg_steps:.1f} (min: {min_steps}, max: {max_steps})")
    status = "✓ PASS" if success_rate >= 80 else "✗ FAIL"
    print(f"Status: {status}")

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"{'Cat Name':<15} {'Success':<12} {'Avg Steps':<15} {'Train Time':<12} {'Status'}")
print("-" * 70)

for cat_name in cats:
    data = results[cat_name]
    success = f"{data['success_rate']:.0f}%"
    steps = f"{data['avg_steps']:.1f}" if data['avg_steps'] != float('inf') else "N/A"
    train = f"{data['train_time']:.2f}s"
    status = "✓ PASS" if data['success_rate'] >= 80 else "✗ FAIL"
    print(f"{cat_name:<15} {success:<12} {steps:<15} {train:<12} {status}")

print("=" * 70)

total_pass = sum(1 for d in results.values() if d['success_rate'] >= 80)
print(f"\n✓ Results: {total_pass}/5 cats caught successfully")
print(f"✓ Overall Score: {(total_pass/5)*100:.0f}%")

if total_pass == 5:
    print("\n🎉 PERFECT! All known cats caught!")
    print("✓ Algorithm ready for submission!")
    print("✓ Should generalize well to hidden cats")
else:
    print(f"\n⚠ {5-total_pass} cat(s) need improvement")

print("\n" + "=" * 70)
print("Use this data for your report!")
print("=" * 70)
