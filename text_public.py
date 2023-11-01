# Sample sentences For quick Validation
sentences = {"Meneer nam het medicijn in":["neemt medicatie in"], 
             "Meneer injecteerde het geneesmiddel zelf":["injectie van medicatie"]}
references = ["injectie van medicatie", "neemt medicatie in"]

result_dict = {'sentence':[], 'reference':[], 'spacy_similarity':[], 'doc2vec_similarity':[],
               'sent2vec_dbert_similarity':[],'sent2vec_mlbert_similarity':[], 'sent2vec_glove_similarity':[],
               'bert_similarity':[],'robert_similarity':[], 'bertje_similarity':[],'jaccard_similarity':[], 
               'wm_distance':[], 'levenshtein_distance':[]}
