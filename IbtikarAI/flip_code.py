import pandas as pd

print("="*80)
print("FIXING FLIPPED LABELS")
print("="*80)

# Load the data
df = pd.read_csv("Clean_Normalized.csv")

print(f"\nBEFORE:")
print(f"  Toxic (1): {(df['label']==1).sum():,} ({(df['label']==1).mean()*100:.2f}%)")
print(f"  Non-toxic (0): {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.2f}%)")

# Flip the labels: 0 -> 1, 1 -> 0
df['label'] = 1 - df['label']

print(f"\nAFTER FLIPPING:")
print(f"  Toxic (1): {(df['label']==1).sum():,} ({(df['label']==1).mean()*100:.2f}%)")
print(f"  Non-toxic (0): {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.2f}%)")

# Save
df.to_csv("Clean_Normalized_FIXED.csv", index=False, encoding='utf-8-sig')

print(f"\n✅ Saved to: Clean_Normalized_FIXED.csv")
print("="*80)

# Verify with toxic words
print("\nVERIFYING WITH TOXIC WORDS:")
toxic_texts = df[df['label'] == 1]['text'].values
non_toxic_texts = df[df['label'] == 0]['text'].values

toxic_words = ['خنزير', 'قذر', 'حمار', 'كلب', 'غبي', 'احمق']
for word in toxic_words:
    toxic_count = sum([word in str(t) for t in toxic_texts])
    non_toxic_count = sum([word in str(t) for t in non_toxic_texts])
    ratio = toxic_count / max(non_toxic_count, 1)
    symbol = "✓" if ratio > 1 else "✗"
    print(f"  {symbol} '{word}': Toxic={toxic_count}, Non-toxic={non_toxic_count} (ratio={ratio:.2f})")

print("\n✅ Labels fixed! Now retrain with Clean_Normalized_FIXED.csv")