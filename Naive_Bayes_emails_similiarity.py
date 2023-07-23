from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

train_emails = fetch_20newsgroups(categories = ['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'], subset = 'train', shuffle = True, random_state = 108)

test_emails = fetch_20newsgroups(categories = ['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'], subset = 'test', shuffle = True, random_state = 108)

#creating counter
counter = CountVectorizer()

#fitting data
counter.fit(test_emails.data + train_emails.data)

#counter
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

#creating classifier
classifier = MultinomialNB()

#fitting data
classifier.fit(train_counts, train_emails.target)

#testing accuracy
print(classifier.score(test_counts, test_emails.target))
