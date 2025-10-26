#!/usr/bin/env python3
"""
CatBot Training Demo - Shows bot learning AND final performance
"""
import subprocess
import sys

cats = ['batmeow', 'mittens', 'paotsin', 'peekaboo', 'squiddyboi']

print("=" * 70)
print("CATBOT TRAINING DEMO - WATCH IT LEARN")
print("=" * 70)
print("\nFor each cat:")
print("  • You'll see the bot at different training stages")
print("  • Episodes: 1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000")
print("  • Watch it get better over time!")
print("  • Final window shows the fully trained bot")
print("\nThis takes longer but you see the learning process!")
print("\nPress ENTER to start...")
input()

for i, cat in enumerate(cats, 1):
    print(f"\n{'=' * 70}")
    print(f"CAT {i}/5: {cat.upper()}")
    print(f"{'=' * 70}")
    print("\nYou'll see training progress every 500 episodes...")
    print("Watch how it improves from random to expert!")
    
    # Show training progress every 500 episodes
    subprocess.run(['python3', 'bot.py', '--cat', cat, '--render', '500'])
    
    print(f"\n✓ {cat} complete!")
    
    if i < len(cats):
        input("\nPress ENTER for next cat...")

print(f"\n{'=' * 70}")
print("ALL CATS CAUGHT!")
print("=" * 70)
print("\nYou just watched the bot learn from scratch!")
