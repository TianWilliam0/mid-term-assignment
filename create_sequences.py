def create_sequences(text, seq_length=100):
    sequences = []
    for i in range(0, len(text) - seq_length, seq_length):
        seq = text[i:i + seq_length]
        sequences.append(seq)
    return sequences

with open('segmented_harry.txt', 'r', encoding='utf-8') as file:
    segmented_text = file.read()

sequences = create_sequences(segmented_text)

# Save as train data
with open('train_sequences_harry.txt', 'w', encoding='utf-8') as f:
    for seq in sequences:
        f.write(f"{seq}\n")
