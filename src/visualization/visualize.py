import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.util import bigrams, trigrams


# Establecer el estilo de Seaborn
sns.set_style("whitegrid")

class DataVisualization:

    def __init__(self, df, column_name):
        self.df = df
        self.column_name = column_name
        self.df['tokens'] = self.df[self.column_name].apply(lambda x: word_tokenize(x.lower()))
        self.df['bigrams'] = self.df['tokens'].apply(lambda x: list(bigrams(x)))
        self.df['trigrams'] = self.df['tokens'].apply(lambda x: list(trigrams(x)))

    def plot_categories_distribution(self):
        plt.figure(figsize=(10, 10))
        category_counts = self.df['sdg'].value_counts()
        plt.pie(category_counts, labels=category_counts.index, autopct=lambda p: '{:.1f}%\n({:,.0f})'.format(p, (p/100)*category_counts.sum()), 
        startangle=90, colors=sns.color_palette('pastel', len(category_counts)))
        plt.title('Distribución de Registros por Categoría')
        plt.legend(title='Categorías', loc="upper left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()

    def plot_length_text_distribution(self):
        self.df['text_length'] = self.df[self.column_name].apply(len)
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.df, x='text_length', bins=50, kde=True, color="skyblue")
        plt.title('Distribución de la Longitud de Textos')
        plt.xlabel('Longitud del Texto')
        plt.ylabel('Cantidad')
        plt.tight_layout()
        plt.show()

    def plot_word_cloud_most_common_words(self):
        word_freq_no_stopwords = Counter([word for tokens in self.df['tokens'] for word in tokens])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_no_stopwords)
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Nube de Palabras')
        plt.show()

    def plot_barh_most_common_words(self, N=10):
        word_freq_no_stopwords = Counter([word for tokens in self.df['tokens'] for word in tokens])
        top_N_words = word_freq_no_stopwords.most_common(N)
        words, counts = zip(*top_N_words)
        colors = ['red'] * N
        plt.figure(figsize=(12, 10))
        plt.barh(words, counts, color=colors)
        plt.xlabel('Frecuencia')
        plt.title(f'Top {N} Palabras Más Comunes')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def plot_barh_longest_words(self, N=5):
        unique_words = set([word.strip() for tokens in self.df['tokens'] for word in tokens])
        sorted_words_by_length = sorted(unique_words, key=len)
        longest_words = sorted_words_by_length[-N:]
        shortest_words = sorted_words_by_length[:N]
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        fig.suptitle('Palabras Más Largas y Más Cortas en el Dataset', fontsize=16)
        ax[0].barh(longest_words, [len(word) for word in longest_words], color='cyan')
        ax[0].set_title(f'{N} Palabras Más Largas')
        ax[0].set_xlabel('Longitud de Palabra')
        ax[1].barh(shortest_words, [len(word) for word in shortest_words], color='magenta')
        ax[1].set_title(F'{N} Palabras Más Cortas')
        ax[1].set_xlabel('Longitud de Palabra')
        ax[1].set_xlim(0, 5)
        plt.tight_layout()
        plt.show()

    def plot_characters_frequency(self):
        alphabet = set('abcdefghijklmnopqrstuvwxyz ')
        char_freq = Counter("".join(self.df[self.column_name]))
        filtered_char_freq = {char: freq for char, freq in char_freq.items() if char.lower() not in alphabet}
        sorted_chars_filtered = sorted(filtered_char_freq.items(), key=lambda x: x[1], reverse=True)
        chars_filtered, char_counts_filtered = zip(*sorted_chars_filtered)
        print(chars_filtered)
        plt.figure(figsize=(15, 10))
        plt.bar(chars_filtered, char_counts_filtered, color='purple')
        plt.xlabel('Carácter')
        plt.ylabel('Frecuencia')
        plt.title('Frecuencia de Caracteres (sin letras del abecedario) en el Dataset')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    
    def plot_most_frequent_n_grams(self, num_bigrams=10, num_trigrams=10):
        bigram_freq = Counter([bigram for bigrams_list in self.df['bigrams'] for bigram in bigrams_list])
        trigram_freq = Counter([trigram for trigrams_list in self.df['trigrams'] for trigram in trigrams_list])
        top_bigrams = bigram_freq.most_common(num_bigrams)
        top_trigrams = trigram_freq.most_common(num_trigrams)
        bigrams, bigram_counts = zip(*top_bigrams)
        bigrams = [" ".join(bigram) for bigram in bigrams]
        trigrams, trigram_counts = zip(*top_trigrams)
        trigrams = [" ".join(bigram) for bigram in trigrams]
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Bigramas y Trigramas más comunes en el Dataset', fontsize=16)
        ax[0].barh(bigrams, bigram_counts, color='green')
        ax[0].set_title(f'Top {num_bigrams} Bigramas Más Comunes')
        ax[0].invert_yaxis()
        ax[0].set_xlabel('Frecuencia')
        ax[0].set_ylabel('Bigramas')
        ax[1].barh(trigrams, trigram_counts, color='orange')
        ax[1].set_title(f'Top {num_trigrams} Trigramas Más Comunes')
        ax[1].invert_yaxis()
        ax[1].set_xlabel('Frecuencia')
        ax[1].set_ylabel('Trigramas')
        plt.tight_layout()
        plt.show()




    
