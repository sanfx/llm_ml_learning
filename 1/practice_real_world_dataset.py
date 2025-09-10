from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='train')
print("20 Newsgroups:", len(newsgroups.data), "samples,", len(newsgroups.target_names), "classes")
