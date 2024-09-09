import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 10000

# feature 1
email_length = np.random.randint(50, 1000, size=n_samples)

# feature 2 (how many "free" are in the email)
word_free_freq = np.random.randint(0, 20, size=n_samples)

# feature 3 (how many "win" words are in the email)
word_win_freq = np.random.randint(0, 10, size=n_samples)

# feature 4 (how many "click" words are in the email)
word_click_freq = np.random.randint(0, 5, size=n_samples)

# feature 5 (how many suscpicious words are in the email)
suspicious_words_freq = np.random.randint(0, 60, size=n_samples)

is_spam = (word_free_freq + word_win_freq + word_click_freq + suspicious_words_freq > 15).astype(int)


# create a dataset
df = pd.DataFrame({
    "email_lenghth": email_length,
    "word_free_freq": word_free_freq,
    "word_win_freq": word_win_freq,
    "word_click_freq": word_click_freq,
    "suspicious_words_freq": suspicious_words_freq,
    "is_spam": is_spam
})

print(df.head())

df.to_csv("../datasets/spam_email.csv", index=False)

print("Dataset saved successfully")