from transformers import pipeline
import time
import spacy
from IPython.display import HTML, display
from spacy import displacy
from gensim.models import Doc2Vec
from transformers import AutoTokenizer, AutoModel
from sent2vec.vectorizer import Vectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
from gensim.models.doc2vec import TaggedDocument
import torch
import Levenshtein
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT

class Decorators:
    @staticmethod
    def timing_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if kwargs.get('verbose', False):
                print(f"{func.__name__} took {elapsed_time:.4f} seconds to run.")
            return result
        return wrapper

class TextProcessing:
    """currently supports only Dutch (default) and English (language='en')"""
    def __init__(self, language='nl', size='lg', verbose=False):
        # Load spaCy model for English
        self.verbose = verbose
        if language == "en":
            self.nlp = spacy.load(f"en_core_web_{size}")
            self.prt_tags = ["prt"]
        elif language == "nl":
            self.nlp = spacy.load(f"nl_core_news_{size}")
            self.prt_tags = ["compound:prt"]
    
    @Decorators.timing_decorator
    def highlight_pos_with_lemma_optimized(self, paragraph):
        # Process the paragraph
        doc = self.nlp(paragraph)
        # HTML template for highlighting with lemma in parentheses
        html_output = '<span style="color: {}">{}</span>'

        # Initialize highlighted text
        highlighted_text = ""

        # Mapping of POS tags to colors
        pos_colors = {'N': 'red', 'V': 'blue', 'ADJ': 'yellow', 'ADV': 'purple'}

        # Iterate through tokens in the processed paragraph
        for token in doc:
            # Get the first character of the POS tag (NOUN -> N, VERB -> V, etc.)
            pos_tag_first_char = token.pos_[0]

            # Determine the color based on POS tag
            color = pos_colors.get(pos_tag_first_char, 'black')

            if token.text == token.lemma_:
                highlighted_text += html_output.format(color, f'{token.text}') + " "
            else:
                # Construct the highlighted text with lemma in parentheses
                highlighted_text += html_output.format(color, f'{token.text} ({token.lemma_})') + " "

        # Display the highlighted text without line breaks
        display(HTML(highlighted_text))
    
    def lemmatise(self,sentence):
        doc = self.nlp(sentence)
        return ' '.join([token.lemma_ for token in doc])

    @Decorators.timing_decorator
    def handle_seperable_verbs(self,sentence):           
        doc = self.nlp(sentence)
        dependencies_dict = {token.head.text:token.text for token in doc if token.dep_ in self.prt_tags}
        sentence = ' '.join([token.text for token in doc if token.dep_ not in self.prt_tags])
        for root,prefix in dependencies_dict.items():
            sentence=sentence.replace(root,prefix+root)
        return sentence
    
    def display_dependency(self,sentence):
        doc = self.nlp(sentence)
        displacy.render(doc, style="dep", options={'distance': 100}, jupyter=True)
        

class Embedder:
    def __init__(self, method, model_path=None, suffix=''):
        self.method = method
        self.suffix = suffix
        if suffix!="":
            self.name = f"{method}_{suffix}"
        else:
            self.name = self.method
        self.model_path = model_path
        self.embedding_dict = {}
        self.comparison_type = 'similarity'
        self.nlp = spacy.load('nl_core_news_lg')
        self.initialise_model()

    def initialise_model(self):
        if self.method=='spacy':
            self.model = self.nlp
        elif self.method=='doc2vec':
            self.model = Doc2Vec.load(self.model_path)
        elif self.method=='sent2vec':
            self.model = Vectorizer(pretrained_weights=self.model_path)
        elif	self.method=='sentTF':
            self.model = SentenceTransformer(self.model_path)
        elif self.method == 'TF':
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
        elif self.method == 'wmd':
            self.comparison_type = 'distance'
        elif self.method == 'ld':
            self.comparison_type = 'distance' 
            
    def generate_embedding(self,sentence):
        if self.method == 'spacy':
            tokens = self.model(sentence)
            vector = np.median([word.vector for word in tokens], axis=0)
        elif self.method == 'doc2vec':
            tokens = self.get_tokens(sentence)
            tagged_data = [TaggedDocument(words=tokens, tags=["sentence"])]
            vector = self.model.infer_vector(tagged_data[0].words)
        elif self.method == 'sent2vec':
            if type(sentence) == list:
                self.model.__init__(self.model_path)
                sentences = sentence
                self.model.run(sentences)
                vector = self.model.vectors
            else:
                self.model.__init__(self.model_path)
                sentences = [sentence]
                self.model.run(sentences)
                vector = self.model.vectors[0]
        elif self.method == 'sentTF':
            if type(sentence) == list:
                self.model.__init__(self.model_path)
                sentences = sentence
                vector = self.model.encode(sentences)
            else:
                self.model.__init__(self.model_path)
                sentences = [sentence]
                vector = self.model.encode(sentences)[0]
        elif self.method == 'TF':
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            with torch.no_grad():
                outputs = self.model(input_ids,attention_mask=attention_mask)
                vector = outputs.last_hidden_state.mean(dim=1).numpy()[0]
        return vector
    
    def get_embedding(self, sentence):
        if type(sentence) == str:
            if not self.is_exist(sentence):
                embedding = self.generate_embedding(sentence)
                self.embedding_dict[sentence] = embedding
            else:
                embedding = self.embedding_dict[sentence]
            return embedding
        elif type(sentence) == list:
            embeddings = self.generate_embedding(sentence)
            for sen,embedding in zip(sentence,embeddings):
                self.embedding_dict[sen] = embedding
    
    def is_exist(self,sentence):
        if sentence in self.embedding_dict.keys():
            return True
        else:
            return False
    
    def get_comparison(self,sentence1,sentence2):
        if self.method in ['spacy','doc2vec','sent2vec','sentTF', 'TF']:
            self.vec1 = self.get_embedding(sentence1)
            self.vec2 = self.get_embedding(sentence2)
            measure = cosine_similarity([self.vec1], [self.vec2])[0][0]
        elif self.method == 'wmd':
            tokens1 = self.nlp(sentence1)
            tokens2 = self.nlp(sentence2)
            measure = tokens1.similarity(tokens2)
        elif self.method=='jaccard':
            tokens1 = self.get_tokens(sentence1)
            tokens2 = self.get_tokens(sentence2)
            set1 = set(tokens1)
            set2 = set(tokens2)
            intersection = len(set1.intersection(set2))
            union = len(set1) + len(set2) - intersection
            measure = intersection / union
        elif self.method == 'ld':
            measure = Levenshtein.distance(sentence1, sentence2)
        return measure
    
    def get_tokens(self,sentence):
        doc = self.nlp(sentence)
        tokens = [token.text for token in doc]
        return tokens
    
class KeywordExtractor:
    def __init__(self, method):
        if method=='keybert':
            self.model = KeyBERT()
    def extract(self, text):
        keywords = self.model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None)
        return keywords


    
class Classifier:
    def __init__(self, method, classes, model_path=None, suffix='', multi_label=False):
        self.method = method
        self.suffix = suffix
        self.classes = classes
        if suffix!="":
            self.name = f"{method}_{suffix}"
        else:
            self.name = self.method
        self.model_path = model_path
        self.embedding_dict = {}
        self.comparison_type = 'similarity'
        self.multi_label = multi_label
        self.initialise_model()
        

    def initialise_model(self):
        if self.method=='spacy':
            self.model = self.nlp
        elif self.method == 'TF':
            self.classifier = pipeline("zero-shot-classification", model=self.model_path)
        elif self.method == 'wmd':
            self.comparison_type = 'distance'

    def classify(self, sequence_to_classify):
        output = self.classifier(sequence_to_classify, self.classes, multi_label=self.multi_label)
        return output

