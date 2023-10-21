import torch
import string
import nltk
import time
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM
from transformers import AutoModel, AutoTokenizer 

def get_antonyms(word):
    ants = list()

    #Get antonyms from WordNet for this word and any of its synonyms.
    for ss in wn.synsets(word):
        ants.extend([lm.antonyms()[0].name() for lm in ss.lemmas() if lm.antonyms()]) 

    #Get snyonyms of antonyms found in the previous step, thus expanding the list even more.
    syns = list()
    for word in ants:
        for ss in wn.synsets(word):
            syns.extend([lm.name() for lm in ss.lemmas()])

    return sorted(list(set(syns)))

'''
Gets pertainyms of the target word from the WordNet knowledge base.
* pertainyms = words pertaining to the target word (industrial -> pertainym is "industry")
'''
def get_pertainyms(word):
    perts = list()
    for ss in wn.synsets(word):
        perts.extend([lm.pertainyms()[0].name() for lm in ss.lemmas() if lm.pertainyms()]) 
    return sorted(list(set(perts)))

'''
Gets derivationally related forms (e.g. begin -> 'beginner', 'beginning')
'''
def get_related_forms(word):
    forms = list()
    for ss in wn.synsets(word):
        forms.extend([lm.derivationally_related_forms()[0].name() for lm in ss.lemmas() if lm.derivationally_related_forms()]) 
    return sorted(list(set(forms)))

'''
Gets antonyms, hypernyms, hyponyms, holonyms, meronyms, pertainyms, and derivationally related forms of a target word from WordNet.
* hypernym = a word whose meaning includes a group of other words ("animal" is a hypernym of "dog")
* hyponym = a word whose meaning is included in the meaning of another word ("bulldog" is a hyponym of "dog")
* a meronym denotes a part and a holonym denotes a whole: "week" is a holonym of "weekend", "eye" is a meronym of "face", and vice-versa
'''
def get_nyms(word, depth=-1):
    nym_list = ['antonyms', 'hypernyms', 'hyponyms', 'holonyms', 'meronyms', 
                'pertainyms', 'derivationally_related_forms']
    results = list()
    lemmatizer = WordNetLemmatizer()
    word = lemmatizer.lemmatize(word)

    def query_wordnet(getter):
        res = list()
        for ss in wn.synsets(word):
            res_list = [item.lemmas() for item in ss.closure(getter, depth=depth)]
            res_list = [item.name() for sublist in res_list for item in sublist]
            res.extend(res_list)
        return res

    for nym in nym_list:
        if nym=='antonyms':
            results.append(get_antonyms(word))

        elif nym in ['hypernyms', 'hyponyms']:
            getter = eval("lambda s : s."+nym+"()") 
            results.append(query_wordnet(getter))

        elif nym in ['holonyms', 'meronyms']:
            res = list()
            #Three different types of holonyms and meronyms as defined in WordNet
            for prefix in ['part_', 'member_', 'substance_']:
                getter = eval("lambda s : s."+prefix+nym+"()")
                res.extend(query_wordnet(getter))
            results.append(res)

        elif nym=='pertainyms':
            results.append(get_pertainyms(word))

        else:
            results.append(get_related_forms(word))

    results = map(set, results)
    nyms = dict(zip(nym_list, results))
    return nyms

#Converts a part-of-speech tag returned by NLTK to a POS tag from WordNet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''

#Function for clearing up duplicate words (capitalized, upper-case, etc.), stop words, and antonyms from the list of candidates.
def filter_words(target, words, scr, tkn):
    dels = list()
    toks = tkn.tolist()
    nyms = get_nyms(target)
    lemmatizer = WordNetLemmatizer()

    for w in words:
        if w.lower() in words and w.capitalize() in words:
            dels.append(w.capitalize())
        if w.lower() in words and w.upper() in words:
            dels.append(w.upper())
        if w in nltk.corpus.stopwords.words('english') or w in string.punctuation:
            dels.append(w)
        if lemmatizer.lemmatize(w.lower()) in nyms['antonyms']:
            dels.append(w)

    dels = list(set(dels))
    for d in dels:
        del scr[words.index(d)]
        del toks[words.index(d)]
        words.remove(d)

    return words, scr, torch.tensor(toks)

#Calculates the similarity score
def similarity_score(original_output, subst_output, k):
    mask_idx = k
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    weights = torch.div(torch.stack(list(original_output[3])).squeeze().sum(0).sum(0), (12 * 12.0))

    #Calculate the similarittimey score 
    #SIM(x, x'; k) = sum_i^L [ w_{i,k} * cos(h(x_i|x), h(x_i'|x')) ]

    #subst_output = raw_model(sent.reshape(1, sent.shape[0]))
    suma = 0.0
    sent_len = original_output[2][2].shape[1]

    for token_idx in range(sent_len):     
        original_hidden = original_output[2]
        subst_hidden = subst_output[2]

        #Calculate the contextualized representation of the i-th word as a concatenation of RoBERTa's values in its last four layers
        context_original = torch.cat( tuple( [original_hidden[hs_idx][:, token_idx, :] for hs_idx in [1, 2, 3, 4]] ), dim=1)
        context_subst = torch.cat( tuple( [subst_hidden[hs_idx][:, token_idx, :] for hs_idx in [1, 2, 3, 4]] ), dim=1)
        suma += weights[mask_idx][token_idx] * cos_sim(context_original, context_subst)

    substitute_validation = suma
    return substitute_validation


#Calculates the proposal score
def proposal_score(original_score, subst_scores, device):
    subst_scores = torch.tensor(subst_scores).to(device)
    return np.log( torch.div(subst_scores , (1.0 - original_score)).cpu() )


class LexSub():
    lemmatizer = WordNetLemmatizer()
    tokenizer = None
    lm_model = None
    raw_model = None
    device = None

    def __init__(self, MODEL="roberta-base"):
        self.tokenizer = RobertaTokenizer.from_pretrained(MODEL)
        self.lm_model = RobertaForMaskedLM.from_pretrained(MODEL)
        self.raw_model = RobertaModel.from_pretrained(MODEL, output_hidden_states=True, output_attentions=True)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.lm_model = self.lm_model.to(self.device)
        self.raw_model = self.raw_model.to(self.device)
        self.K = 5

    def load_transformers(self):
        return self.tokenizer, self.lm_model, self.raw_model

    #Calculates the proposal scores, substitute validation scores, and then the final score for each candidate word's fit as a substitution.
    def calc_scores(self, scr, sentences, original_output, original_score, mask_index):
        #Get representations of all substitute sentences
        _, _, raw_model = self.load_transformers()
        subst_output = raw_model(sentences)

        prop_score = proposal_score(original_score, scr, self.device)
        substitute_validation = similarity_score(original_output, subst_output, mask_index)

        alpha = 0.003
        final_score = substitute_validation.cpu() + alpha*prop_score
        
        return final_score, prop_score, substitute_validation

    def lexsub_concatenation(self, sentence, target, K=5):
        #Removes the unnecessary punctuation from the input sentence.
        sentence = sentence.replace('-', ' ')
        table = str.maketrans(dict.fromkeys(string.punctuation)) 

        split_sent = nltk.word_tokenize(sentence)
        split_sent = list(map(lambda wrd : wrd.translate(table) if wrd not in string.punctuation else wrd, split_sent))
        original_sent = ' '.join(split_sent)

        #Masks the target word in the original sentence.
        masked_sent = ' '.join(split_sent)
        masked_sent = masked_sent.replace(target, self.tokenizer.mask_token, 1)

        #Get the input token IDs of the input consisting of: the original sentence + separator + the masked sentence.
        input_ids = self.tokenizer.encode(" "+original_sent, " "+masked_sent, add_special_tokens=True)
        try:
            masked_position = input_ids.index(self.tokenizer.mask_token_id)
        except ValueError:
            return []

        original_output = self.raw_model(torch.tensor(input_ids).reshape(1, len(input_ids)).to(self.device))

        #Get the predictions of the Masked LM transformer.
        with torch.no_grad():
            output = self.lm_model(torch.tensor(input_ids).reshape(1, len(input_ids)).to(self.device))
        
        logits = output[0].squeeze().detach().cpu().numpy()
        #logits = torch.squeeze(output[0])

        #Get top guesses: their token IDs, scores, and words.
        mask_logits = logits[masked_position].squeeze()
        top_tokens = torch.topk(torch.from_numpy(mask_logits), k=20, dim=0)[1]
        scores = torch.softmax(torch.from_numpy(mask_logits), dim=0)[top_tokens].tolist()
        words = [self.tokenizer.decode(i.item()).strip() for i in top_tokens]
        
        words, scores, top_tokens = filter_words(target, words, scores, top_tokens)
        assert len(words) == len(scores)

        if len(words) == 0: 
            return []

        #print("GUESSES: ", words)

        #Calculate proposal scores, substitute validation scores, and final scores
        original_score = torch.softmax(torch.from_numpy(mask_logits), dim=0)[masked_position]
        sentences = list()

        for i in range(len(words)):
            subst_word = top_tokens[i]
            input_ids[masked_position] = int(subst_word)
            sentences.append(list(input_ids))

        #print([tokenizer.decode(s) for s in sentences])
        torch_sentences = torch.tensor(sentences).to(self.device)

        finals, props, subval = self.calc_scores(scores, torch_sentences, original_output, original_score, masked_position)
        finals = map(lambda f : float(f), finals)
        props = map(lambda f : float(f), props)
        subval = map(lambda f : float(f), subval)

        if target in words:
            words = [w for w in words if w not in [target, target.capitalize(), target.upper()]] 

        zipped = dict(zip(words, finals))

        ###Remove plurals, wrong verb tenses, duplicate forms, etc.############
        original_pos = nltk.pos_tag(nltk.word_tokenize(original_sent))
        target_index = split_sent.index(target)
        try:
            assert original_pos[target_index][0] == target
        except (AssertionError, IndexError) as e:
            return []
        original_tag = original_pos[target_index][1]

        for i in range(len(words)):
            cand = words[i]
            if cand not in zipped:
                continue
            
            sent = original_sent
            masked_sent = sent.replace(target, cand, 1)

            new_pos = nltk.pos_tag(nltk.word_tokenize(masked_sent))
            new_tag = new_pos[target_index][1]

            #If the word appears in both singular and plural in the candidate list, remove one of them.
            if new_tag.startswith('N') and not new_tag.endswith('S'):
                if (cand+'s' in words or cand+'es' in words) in words and original_tag.endswith('S'):
                    del zipped[cand]
                    continue        
            elif new_tag.startswith('N') and new_tag.endswith('S'):
                if (cand[:-1] in words or cand[:-2] in words) and not original_tag.endswith('S'):
                    del zipped[cand]
                    continue
            
            #If multiple forms of the original word appear in the candidate list, remove them (e.g. begin, begins, began, begun...)
            wntags = ['a', 'r', 'n', 'v']
            for tag in wntags:
                if self.lemmatizer.lemmatize(cand, tag) == self.lemmatizer.lemmatize(target, tag):
                    del zipped[cand]
                    break
        #################        

        #Print sorted candidate words.
        zipped = dict(zipped)
        finish = list(sorted(zipped.items(), key=lambda item: item[1], reverse=True))[:K]
        #print("CANDIDATES:", ["({0}: {1:0.8f})".format(k, v) for k,v in finish])

        #Print any relations between candidates and the original word.
        #words = zipped.keys()
        #nyms = get_nyms(target)
        #nym_output = list()
        #for cand in words:
        #    for k, v in nyms.items():
        #        if self.lemmatizer.lemmatize(cand.lower()) in v:
        #            nym_output.append((cand, k[:-1]))

        #if nym_output:
        #    print("NYMS: ", ["({0}, {1})".format(pair[0], pair[1]) for pair in nym_output])

        return finish
    
    def lexsub_dropout(self, sentence, target, K=5):
        sentence = sentence.replace('-', ' ')
        table = str.maketrans(dict.fromkeys(string.punctuation)) 

        #Remove unnecessary punctuation from the sentence (such as: "GET *free food *coupons!!")
        split_sent = nltk.word_tokenize(sentence)
        split_sent = list(map(lambda wrd : wrd.translate(table) if wrd not in string.punctuation else wrd, split_sent))
        original_sent = ' '.join(split_sent)

        #Get RoBERTa word embeddings for words in the sentence
        original_output = self.raw_model(self.tokenizer.encode(" "+original_sent, return_tensors="pt").to(self.device))
        inputs_embeds = original_output[2][1]

        #The target word to substitute
        target_token_id = self.tokenizer.encode(" "+target)[1]
        input_ids = self.tokenizer.encode(" " + original_sent)
        mask_position = input_ids.index(target_token_id)

        #Set a percentage of randomly selected embedding weights of the target word to 0.
        embedding_dim = 768
        dropout_percent = 0.3
        dropout_amount = round(dropout_percent*embedding_dim)

        #Start timing the experiment.
        start = time.time()

        #Run multiple experiments and then take average because of stochastic nature of choosing indices to dropout (sometimes the predictions are gibberish)
        all_scores = dict()
        all_counts = dict()
        num_iterations = 5
        for it in range(num_iterations):
            #Choose the weight indices to drop out.
            dropout_indices = np.random.choice(embedding_dim, dropout_amount, replace=False)
            inputs_embeds[0, mask_position, dropout_indices] = 0

            #Pass the embeddings where masked word's embedding is partially droppped out to the model 
            with torch.no_grad():
                    output = self.lm_model(inputs_embeds=inputs_embeds.to(self.device))
            logits = output[0].squeeze()

            #Get top guesses
            mask_logits = logits[mask_position]
            top_tokens = torch.topk(mask_logits, k=16, dim=0)[1]
            scores = torch.softmax(mask_logits, dim=0)[top_tokens].tolist()
            words = [self.tokenizer.decode(i.item()).strip() for i in top_tokens]
            
            words, scores, top_tokens = filter_words(target, words, scores, top_tokens)
            assert len(words) == len(scores)

            if len(words) == 0: 
                continue

            #Calculate proposal scores, substitute validation scores, and final scores
            original_score = torch.softmax(mask_logits, dim=0)[target_token_id]
            sentences = list()
            split_sent = nltk.word_tokenize(sentence)

            for i in range(len(words)):
                subst_word = top_tokens[i]
                input_ids[mask_position] = int(subst_word)
                sentences.append(list(input_ids))

            sentences = torch.tensor(sentences).to(self.device)
        
            finals, props, subval = self.calc_scores(scores, sentences, original_output, original_score, mask_position)
            finals = map(lambda f : float(f), finals)
            props = map(lambda f : float(f), props)
            subval = map(lambda f : float(f), subval)

            if target in words:
                words.remove(target)

            #Update total scores and counts in the dictionary
            res = dict(zip(words, finals))
            for w, s in res.items():
                all_scores[w] = all_scores[w] + s if w in all_scores.keys() else s
                all_counts[w] = all_counts[w] + 1 if w in all_counts.keys() else 1

        #Get the average of accumulated scores.
        for w, s in all_scores.items():
            all_scores[w] = s / all_counts[w]
        words, finals = list(all_scores.keys()), list(all_scores.values())

        #Sort the found substitutes by scores and print them out.
        x = dict(zip(words, finals))
        finish = list(sorted(x.items(), key=lambda item: item[1], reverse=True))[:K]
        return finish