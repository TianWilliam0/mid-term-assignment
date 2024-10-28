import jieba

def clean_text(text):
    # Remove special symbols and blank characters
    text = text.replace('\n', '').replace('\r', '').replace('-','')
    return text

def segment_text(text):
    return ' '.join(jieba.cut(text))

with open('harry.txt', 'r', encoding='utf-8') as file:
    raw_text = file.read()

cleaned_text = clean_text(raw_text)
segmented_text = segment_text(cleaned_text)
# print(segmented_text)

# Saving the split text
with open('segmented_harry.txt', 'w', encoding='utf-8') as file:
    file.write(segmented_text)
