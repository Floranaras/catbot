#!/usr/bin/env python3
"""
CatBot Final Demo - Shows only the trained bot catching cats
"""
import subprocess
import sys

cats = ['batmeow', 'mittens', 'paotsin', 'peekaboo', 'squiddyboi']

print("=" * 70)
print("CATBOT FINAL DEMO")
print("=" * 70)
print("\nFor each cat:")
print("  • Training happens in background (no window)")
print("  • Window opens showing trained bot catching the cat")
print("  • Watch it succeed!")
print("\nPress ENTER to start...")
input()

for i, cat in enumerate(cats, 1):
    print(f"\n{'=' * 70}")
    print(f"CAT {i}/5: {cat.upper()}")
    print(f"{'=' * 70}")
    print("\nTraining in background (fast)...")
    
    # Run bot with GUI - only shows final trained bot
    subprocess.run(['python3', 'bot.py', '--cat', cat, '--render', '-1'])
    
    print(f"\n✓ {cat} complete!")
    
    if i < len(cats):
        input("\nPress ENTER for next cat...")

print(f"\n{'=' * 70}")
print("ALL CATS CAUGHT!")
print("=" * 70)
