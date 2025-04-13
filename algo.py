import nltk
from nltk.corpus import words, brown
from collections import deque, defaultdict
import math
import re

class OptimizedSegmenter:
    def __init__(self, buffer_size=5, ngram_order=3):
        nltk.download(['words', 'brown'], quiet=True)
        
        # Load resources and convert to lowercase for better matching
        self.word_list = set(w.lower() for w in words.words())
        self.common_words = {'the', 'is', 'are', 'were', 'was', 'be', 'being', 'been',
                           'a', 'an', 'and', 'or', 'but', 'if', 'of', 'at', 'by', 
                           'for', 'with', 'about', 'to', 'from', 'in', 'on', 'off',
                           'out', 'over', 'under', 'before', 'after', 'while', 'until'}
        
        # Character mapping for common errors
        self.char_map = {
            '0': 'o', '3': 'e', '1': 'i', 
            '4': 'a', '5': 's', '7': 't',
            'z': 's', 'L': 'l', 'E': 'e', 'R': 'r'
        }
        
        # Ngram initialization
        self.ngram_order = ngram_order
        self.ngram_probs = self._train_ngram_model()
        self.buffer = deque(maxlen=buffer_size)
        
        # Common word boundaries to enforce
        self.common_prefixes = {'in', 'un', 're', 'de', 'dis', 'over', 'under', 'pre', 'post', 'anti'}
        self.common_suffixes = {'ing', 'ed', 'er', 'tion', 'ment', 'ness', 'ity', 'ly', 'ful', 'able'}

    def preprocess_input(self, char_list):
        """Convert character list to cleaned string"""
        return ''.join([self.char_map.get(c, c.lower()) for c in char_list if c.isalnum()])

    def _train_ngram_model(self):
        """Train language model on Brown corpus"""
        tokens = [w.lower() for w in brown.words()]
        ngrams = nltk.ngrams(tokens, self.ngram_order)
        fdist = nltk.FreqDist(ngrams)
        total = fdist.N()
        return {ngram: (count/total) for ngram, count in fdist.items()}

    def _ngram_score(self, context, candidate):
        """Calculate n-gram probability"""
        if len(context) < self.ngram_order-1:
            context = ['<s>']*(self.ngram_order-1 - len(context)) + context
        else:
            context = context[-(self.ngram_order-1):]
        ngram = tuple(context + [candidate])
        return math.log(self.ngram_probs.get(ngram, 1e-10))

    def _hybrid_cost(self, text, start, end):
        """Combined cost function with bonuses for common words"""
        candidate = text[start:end]
        
        # Base dictionary score
        if candidate in self.word_list:
            # Bonus for common words
            if candidate in self.common_words:
                dict_score = 4*len(candidate)
            else:
                dict_score = 2*len(candidate)
        else:
            dict_score = -1
        
        # N-gram context score
        context = list(self.buffer)[-(self.ngram_order-1):]
        ngram_score = self._ngram_score(context, candidate)
        
        # Combined score with weights
        return 0.7*dict_score + 0.3*ngram_score

    def segment_text(self, text):
        """Primary segmentation algorithm"""
        n = len(text)
        dp = [-math.inf]*(n+1)
        dp[0] = 0
        path = [0]*(n+1)
        
        for i in range(1, n+1):
            for j in range(max(0, i-15), i):
                score = dp[j] + self._hybrid_cost(text, j, i)
                if score > dp[i]:
                    dp[i] = score
                    path[i] = j
        
        # Reconstruct path
        segments = []
        i = n
        while i > 0:
            j = path[i]
            segments.insert(0, text[j:i])
            self.buffer.append(text[j:i])
            i = j
        
        # Apply post-processing to fix common issues
        return self._post_process_segments(segments)
    
    def _post_process_segments(self, segments):
        """Apply fixes for common segmentation errors"""
        result = []
        for segment in segments:
            # Try to split long unknown words
            if len(segment) > 7 and segment not in self.word_list:
                # Try to find common word boundaries
                for i in range(2, len(segment)-2):
                    prefix = segment[:i]
                    suffix = segment[i:]
                    
                    # Check if splitting creates valid words
                    if prefix in self.word_list and suffix in self.word_list:
                        result.append(prefix)
                        result.append(suffix)
                        break
                else:
                    # No valid split found
                    result.append(segment)
            else:
                result.append(segment)
        
        return ' '.join(result)

    def process_characters(self, char_list):
        """Main entry point for character processing"""
        cleaned_text = self.preprocess_input(char_list)
        return self.segment_text(cleaned_text)

def test_segmenter():
    segmenter = OptimizedSegmenter()
    
    test_cases = [
        (["A","p","p","L","E","i","s","r","e","d"], "apple is red"),
        (["z","3","0","z","p","0","z","p","f","k","0","e","g","0","0"], "simple green"),
        (["h","e","L","L","0","w","0","r","L","d"], "hello world"),
        (["t","h","3","q","u","1","c","k","b","r","0","w","n","f","0","x"], "the quick brown fox")
    ]
    
    print("Testing Optimized Segmenter:")
    print("=" * 50)
    for input_chars, expected in test_cases:
        # Process characters
        result = segmenter.process_characters(input_chars)
        print(f"Input: {input_chars}")
        print(f"Result: {result}")
        print(f"Expected: {expected}")
        print("-" * 50)

if __name__ == "__main__":
    test_segmenter()
