import nltk
from nltk.corpus import brown , words
from collections import defaultdict, deque
import math

class HybridSegmenter:
    def __init__(self, buffer_size=5, ngram_order=3):
        nltk.download('brown', quiet=True)
        
        # Load resources
        self.word_list = set(words.words())
        self.ngram_order = ngram_order
        self.ngram_probs = self.train_ngram_model()
        
        # Dynamic programming cache
        self.cache = defaultdict(float)
        
        # Context window
        self.buffer = deque(maxlen=buffer_size)

    def train_ngram_model(self):
        """Train n-gram model on Brown corpus"""
        tokens = brown.words()
        ngrams = nltk.ngrams(tokens, self.ngram_order)
        fdist = nltk.FreqDist(ngrams)
        total = fdist.N()
        
        return {ngram: (count/total) for ngram, count in fdist.items()}

    def ngram_score(self, context, candidate):
        """Calculate n-gram probability"""
        if len(context) < self.ngram_order-1:
            context = ['<s>']*(self.ngram_order-1 - len(context)) + context
        else:
            context = context[-(self.ngram_order-1):]
            
        ngram = tuple(context + [candidate])
        return math.log(self.ngram_probs.get(ngram, 1e-10))

    def hybrid_cost(self, text, start, end):
        """Combined cost function using n-grams and dictionary"""
        candidate = text[start:end]
        
        # Dictionary score
        dict_score = 2*len(candidate) if candidate in self.word_list else -1
        
        # N-gram score
        context = list(self.buffer)[-(self.ngram_order-1):]
        ngram_score = self.ngram_score(context, candidate)
        
        # Combined score (weights optimized empirically)
        return 0.6*dict_score + 0.4*ngram_score

    def segment_text(self, text):
        n = len(text)
        dp = [-math.inf]*(n+1)
        dp[0] = 0
        path = [0]*(n+1)
        
        for i in range(1, n+1):
            for j in range(max(0, i-15), i):
                score = dp[j] + self.hybrid_cost(text, j, i)
                if score > dp[i]:
                    dp[i] = score
                    path[i] = j
        
        # Reconstruct path
        segments = []
        i = n
        while i > 0:
            j = path[i]
            segments.insert(0, text[j:i])
            self.buffer.append(text[j:i])  # Update context
            i = j
            
        return ' '.join(segments)
