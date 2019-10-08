


from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import re as r
from nltk.corpus import stopwords


text="Hello Mr Smith how are you doing today? The weather is great and city is awesome. The sky is pinkish-blue. You shouldn't eat cardboard"
tokenized_text=sent_tokenize(text)
text1=text.lower()


freqword=FreqDist(wordtokenize)

freqword.most_common(6)

freqword.plot(20,cumulative=False)

#removing punctuation

rptext=r.sub('[^\w\s]+','',text1)
wordtokenize=word_tokenize(rptext)


rpwordtoken=word_tokenize(rptext)

#stopword
#import nltk
#nltk.download("stopwords")


swords=set(stopwords.words('english'))
swords

filtlist=[]

for words in rpwordtoken:
    if words not in swords:
        filtlist.append(words)
        
