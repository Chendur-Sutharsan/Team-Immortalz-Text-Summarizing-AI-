import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 3))

text = """The buffer overflow exploit techniques a hacker uses depends on the architecture and operating system being used by their target. However, the extra data they issue to a program will likely contain malicious code that enables the attacker to trigger additional actions and send new instructions to the application. For example, introducing additional code into a program could send it new instructions that give the attacker access to the organizations IT systems. In the event that an attacker knows a programs memory layout, they may be able to intentionally input data that cannot be stored by the buffer. This will enable them to overwrite memory locations that store executable code and replace it with malicious code that allows them to take control of the program."""

sentences = sent_tokenize(text)
words = [word_tokenize(sentence) for sentence in sentences]

stopwords = set(stopwords.words('english'))
filtered_words = []
for i in range(len(words)):
    filtered_words.append([word for word in words[i] if word.lower() not in stopwords])

flattened_words = [word for sublist in filtered_words for word in sublist]

word_freq = nltk.FreqDist(flattened_words)

sentence_scores = {}
for i in range(len(sentences)):
    score = 0
    for word in filtered_words[i]:
        score += word_freq[word.lower()]
    sentence_scores[i] = score

top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:5]
summary_sentences = []
for i in top_sentences:
    summary_sentences.append(sentences[i])
summary_sentences.pop(-1)

print("----- The given text -----")
print(text)

summary = ' '.join(summary_sentences)
print("----- The Summarized text -----")
print(summary)
