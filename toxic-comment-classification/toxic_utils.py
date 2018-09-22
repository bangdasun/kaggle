import re
import string
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

stopwords_collection = ['the', 'is', 'this', 'and', 'are']
stopwords_num = [1] * len(stopwords_collection)
stopwords = dict(zip(stopwords_collection, stopwords_num))

    
def calc_avg_auc(y_true_mat, y_pred_mat):
    """
    Metric calculation: column-wise average AUC
    
    Parameters
    ----------
    y_true_mat: integer matrix
    y_pred_mat: each column is a binary probability
    
    Example
    -------
    >>> y_pred_mat = np.random.rand(train.shape[0], 6)
    >>> y_true_mat = train.iloc[:, 2:].values
    >>> calc_avg_auc(y_true_mat, y_pred_mat)
    """
    
    assert y_true_mat.shape == y_pred_mat.shape
    ncol = y_true_mat.shape[1]
    all_auc = [roc_auc_score(y_true_mat[:, i], y_pred_mat[:, i]) for i in range(ncol)]
    
    return np.mean(all_auc)

def associations(label1, label2, data):
    """
    label2 -> label1: association rules
    
    Parameters
    ----------
    """
    count_1 = np.sum(data.loc[data[label2] == 1, label1])
    count_2 = np.sum(data[label2])
    return count_1 / count_2


# from kaggle data
str_replace_table = {
    r"won't": "will not",
    r"can't": "cannot",
    r" r "  : " are ",
    r" u "  : " you ",
    r" ur " : " your ",
 
    r"n't": " not", 
    r"\\n": "", 
    r"'m" : " am",
    r"'s" : " is",
    
    r"'ve": " have",
    r"'d" : " would", 
    r"'ll": " will", 
    r"'re": " are", 
    
    r"&lt;3"  : " good ",
    r":d"     : " good ",
    r":dd"    : " good ",
    r":p"     : " good ",
    r"8)"     : " good ",
    r":-)"    : " good ",
    r":)"     : " good ",
    r";)"     : " good ",
    r"(-:"    : " good ",
    r"(:"     : " good ",
    r"yay!"   : " good ",
    r"yay"    : " good ",
    r"yaay"   : " good ",
    r"yaaay"  : " good ",
    r"yaaaay" : " good ",
    r"yaaaaay": " good ",
    r":/"     : " bad ",
    r":&gt;"  : " sad ",
    r":')"    : " sad ",
    r":-("    : " bad ",
    r":("     : " bad ",
    r":s"     : " bad ",
    r":-s"    : " bad ",
    r"&lt;3"  : " heart ",
    r":d"     : " smile ",
    r":p"     : " smile ",
    r":dd"    : " smile ",
    r"8)"     : " smile ",
    r":-)"    : " smile ",
    r":)"     : " smile ",
    r";)"     : " smile ",
    r";-)"    : " smile ",
    r"(-:"    : " smile ",
    r"(:"     : " smile ",
    r":/"     : " worry ",
    r":&gt;"  : " angry ",
    r":'\)"   : " sad ",
    r":-\("   : " sad ",
    r":\("    : " sad ",
    r":s"     : " sad ",
    r":-s"    : " sad ",
    
    r"f**k"       : "fuck",
    r"s**t"       : "shit",
    r"n ig ger"   : " nigger ",
    r"nig ger"    : " nigger ",
    r"niggers"    : " nigger ",
    r"!"          : " !",
    r"n i g g e r": " nigger ",
    r" nigga "    : " nigger ",
    r" niggas "   : " nigger ",
    r"jews"       : " jew ",
    r"jewish"     : " jew ",
    r"p i s s"    : "piss",
    r"a**"        : "ass",
    r"fucing"     : " fucking ",
    r"fuckin "    : " fucking ",
    r"fukkin "    : " fucking ",
    r"fukking"    : " fucking ",
    r"fucking"    : " fuck ",
    r"fucked"     : " fuck ",
   
    r"moderfuckn"  : " motherfucker ",
    r"muthafucking": " motherfucker ",
    r" fuk "       : " fuck ",
    r" s hit"      : " shit ",
    r"f\*\*\*"     : "fuck",
    r" kil "       : " kill ",
    r" wil "       : " will ",
    r"fuck"        : " fuck ",
    r"\*\*\*\* you": "fuck you",
    r"y\*\*"       : "you", 
    r"f___"        : "fuck",
    r" ya."        : " you.",
    r" ya?"        : " you?",
    r"s.hit"       : "shit",
    
    r"fuckeeed"     : "fucked",
    r"fagget"       : "faggot",
    r"sh\_t"        : "shit",
    r"Fu ck ing"    : "Fucking",
    r" im "         : " i am",
    r" dont "       : " do not ",
    r"administratin": "administration",
    
    r" dik ": " dick ",
    r" suk ": " suck ",
    r"f{1,}\\s{1,}u{1,}\\s{1,}c{1,}\\s{1,}k{1,}": "fuck",
    
    r"\\t"   : "",
    r"b1tch" : "bitch",
    r"fuk1ng": "fucking",
    r"fcuk"  : "fuck",
    r"fuckah": "fuck",
    r"sh!t"  : "shit",
    r"sh1t"  : "shit",
    r"fuker" : "fucker",
    r"fckin ": "fucking ",
    r"fcker ": "fucker ",
    r"mutha" : "mother",
    r"fucka" : "fucker",
    r"f\*ck" : "fuck",
}

def str_replace(data):
    """
    Normalize string
    
    Parameters
    ----------
    
    """
    data['comment_text'].replace(str_replace_table, regex=True, inplace=True)
    
def repeat_letter(text):
    """
    Detect repeat letter in a text
    
    https://stackoverflow.com/questions/11460397/match-the-same-unknown-character-multiple-times
    """
    
    return len(re.findall(r"([A-Za-z!\*])\1{3,}", text))

def clean_comment(text):
    
    text = text.lower()
    text = re.sub('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'ip', text)
    # text = re.sub('\[\[.*\]', 'username', text)
    return text

def clean_url(text):
    
    text = re.sub('((www.[^\s]+)|(https?://[^\s]+))','url',text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text

def character_range(text): 
    for ch in string.ascii_lowercase[:27]:
        if ch in text:
            template = r"(" + ch + ")\\1{2,}"
            text = re.sub(template, ch, text)
    return text

def clean_punctuation(text):
    punct = re.compile('[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
    text = ' '.join([punct.sub('', token) for token in text.split()])
    return text

def count_regexp_occ(regexp, text):
    return len(re.findall(regexp, text))