import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import ImageColorGenerator
from wordcloud import WordCloud
import warnings
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.util import bigrams, trigrams
import numpy as np
import random
warnings.filterwarnings('ignore', category=FutureWarning)
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

    def plot_length_text_distribution_by_category(self):
        plt.figure(figsize=(15, 8))
        sns.violinplot(data=self.df, x='sdg', y='text_length', inner="quartile")
        plt.title('Distribución de la Longitud de Textos por Categoría')
        plt.xlabel('Categoría')
        plt.ylabel('Longitud del Texto')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_most_common_words_by_category(self, N=10):
        # Diccionario para mapear categorías a colores
        category_colors = {
            6: (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
            7: (1.0, 0.4980392156862745, 0.054901960784313725),
            16: (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
        }

        categories = self.df['sdg'].unique()
        for category in categories:
            # Seleccionar color de la categoría actual
            current_color = category_colors.get(category)
            if not current_color:
                continue  # Si la categoría no tiene un color asignado en el diccionario, saltamos al siguiente ciclo

            cat_data = self.df[self.df['sdg'] == category]
            word_freq = Counter([word for tokens in cat_data['tokens'] for word in tokens])
            top_N_words = word_freq.most_common(N)
            words, counts = zip(*top_N_words)
            
            plt.figure(figsize=(12, 8))
            plt.barh(words, counts, color=current_color)  # Usar el color seleccionado
            plt.xlabel('Frecuencia')
            plt.title(f'Top {N} Palabras Más Comunes en Categoría {category}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()


    def plot_most_frequent_bigrams_comparison(self, num_bigrams=10):
        categories = self.df['sdg'].unique()
        fig, ax = plt.subplots(figsize=(15, 10))
        
        for category in categories:
            cat_data = self.df[self.df['sdg'] == category]
            bigram_freq = Counter([bigram for bigrams_list in cat_data['bigrams'] for bigram in bigrams_list])
            top_bigrams = bigram_freq.most_common(num_bigrams)
            bigrams, bigram_counts = zip(*top_bigrams)
            bigrams = [" ".join(bigram) for bigram in bigrams]
            ax.barh(bigrams, bigram_counts, label=category)
        
        ax.set_title(f'Comparación de Top {num_bigrams} Bigramas Más Comunes por Categoría')
        ax.set_xlabel('Frecuencia')
        ax.set_ylabel('Bigramas')
        ax.legend(title='Categorías')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    def plot_most_frequent_trigrams_comparison(self, num_trigrams=10):
        categories = self.df['sdg'].unique()
        num_categories = len(categories)
        
        fig, axes = plt.subplots(nrows=num_categories, figsize=(15, 6*num_categories))
        
        for ax, category in zip(axes, categories):
            cat_data = self.df[self.df['sdg'] == category]
            trigram_freq = Counter([trigram for trigrams_list in cat_data['trigrams'] for trigram in trigrams_list])
            top_trigrams = trigram_freq.most_common(num_trigrams)
            trigrams, trigram_counts = zip(*top_trigrams)
            trigrams = [" ".join(trigram) for trigram in trigrams]
            ax.barh(trigrams, trigram_counts, color=sns.color_palette()[categories.tolist().index(category)])  # Use different colors for each category
            ax.set_title(f'Top {num_trigrams} Trigramas Más Comunes en "{category}"')
            ax.set_xlabel('Frecuencia')
            ax.set_ylabel('Trigramas')
            ax.invert_yaxis()

        plt.tight_layout()
        plt.show()



    def plot_characters_frequency_by_category(self):
        categories = self.df['sdg'].unique()
        alphabet = set('abcdefghijklmnopqrstuvwxyz ')
        for category in categories:
            cat_data = self.df[self.df['sdg'] == category]
            char_freq = Counter("".join(cat_data[self.column_name]))
            filtered_char_freq = {char: freq for char, freq in char_freq.items() if char.lower() not in alphabet}
            sorted_chars_filtered = sorted(filtered_char_freq.items(), key=lambda x: x[1], reverse=True)
            chars_filtered, char_counts_filtered = zip(*sorted_chars_filtered)
            plt.figure(figsize=(15, 8))
            plt.bar(chars_filtered, char_counts_filtered, color=sns.color_palette('spring', len(chars_filtered)))
            plt.xlabel('Carácter')
            plt.ylabel('Frecuencia')
            plt.title(f'Frecuencia de Caracteres en Categoría {category}')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()

    def compare_text_lengths_by_category(self):
        # Diccionario de colores para categorías
        category_colors = {
            6: (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
            7: (1.0, 0.4980392156862745, 0.054901960784313725),
            16: (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
        }

        # Lista de colores basada en el orden de las categorías en el DataFrame
        plot_colors = [category_colors[cat] for cat in self.df['sdg'].unique() if cat in category_colors]

        plt.figure(figsize=(15, 8))
        sns.boxplot(data=self.df, x='sdg', y='text_length', palette=plot_colors)
        plt.title('Comparación de Longitudes de Texto por Categoría')
        plt.xlabel('Categoría')
        plt.ylabel('Longitud del Texto')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_word_clouds_by_category(self):
        categories = self.df['sdg'].unique()

        # Listas de colores similares para cada categoría
        color_map = {
            categories[0]: ["#1E90FF", "#4169E1", "#4682B4", "#5F9EA0"],  # Diferentes tonos de azul
            categories[1]: ["#FFD700", "#FFA500", "#FFB6C1", "#FF4500"],  # Diferentes tonos de amarillo
            categories[2]: ["#7CFC00", "#32CD32", "#3CB371", "#228B22"],  # Diferentes tonos de verde
        }

        fig, axes = plt.subplots(1, len(categories), figsize=(15, 5))

        for ax, category in zip(axes, categories):
            cat_data = self.df[self.df['sdg'] == category]
            word_freq = Counter([word for tokens in cat_data['tokens'] for word in tokens])

            # Función personalizada para elegir colores aleatoriamente de la lista asociada a cada categoría
            def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                return random.choice(color_map[category])

            wordcloud = WordCloud(background_color='white', color_func=color_func).generate_from_frequencies(word_freq)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'Nube de Palabras: {category}')

        plt.tight_layout()
        plt.show()
    
