import re
import spacy
import pandas as pd


class DataProcessor:
    """
    Class for reading, processing, and writing data from the UCI
    `Condition monitoring of hydraulic systems` dataset.
    """
    def __init__(self, raw_data_path):
        self.df = pd.read_excel(raw_data_path)
        self.nlp = spacy.load('es_core_news_sm')
        self.multi_char_map = {'Ã\x81': 'Á', 'Ã¡': 'á', 'Ã‰': 'É', 'Ã©': 'é', 'Ã\x8d': 'Í', 'Ã“': 'Ó', 'Ã³': 'ó', 'Ãš': 'Ú', 'Ãº': 'ú', 'Ã‘': 'Ñ','Ã±':'ñ', 'Ã': 'í'}
        self.single_char_map = {'Á': 'A', 'á': 'a', 'É': 'E', 'é': 'e', 'Í': 'I', 'í': 'i', 'Ó': 'O', 'ó': 'o', 'Ú': 'U', 'ú': 'u', 'ü': 'u', 'energetico':'energia'}
        self.url_pattern = re.compile(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})")
        self.stop_words  = {'serán', 'otra', 'seáis', 'tendría', 'algunas', 'te', 'nuestras', 'sentidos', 'tuvierais', 'tendrán', 'estuviesen', 'además', 'hubiesen', 'tuviésemos', 'esté', 'tenemos', 'seremos', 'también', 'habíamos', 'desde', 'donde', 'estén', 'sentida', 'e', 'habrían', 'como', 'estábamos', 'seríamos', 'estuviésemos', 'fueses', 'antes', 'nuestra', 'estuvierais', 'tuyas', 'estoy', 'tenéis', 'míos', 'fui', 'tus', 'fueras', 'mucho', 'estarás', 'tenía', 'tuvimos', 'serías', 'habidos', 'tengo', 'tengáis', 'mi', 'sentido', 'a', 'habríais', 'tuyo', 'fuiste', 'soy', 'ellos', 'estamos', 'hubieras', 'ella', 'habían', 'seríais', 'contra', 'estaría', 'estéis', 'habrá', 'estaban', 'tendréis', 'mía', 'eso', 'habría', 'él', 'tenga', 'hubiera', 'hayáis', 'ellas', 'pero', 'estuviste', 'se', 'tendrían', 'estar', 'aún', 'la', 'qué', 'asi', 'hubo', 'habrías', 'estuviéramos', 'me', 'estuviera', 'muchos', 'las', 'habíais', 'tendrías', 'tuviese', 'estuve', 'es', 'sean', 'tuvieseis', 'entre', 'había', 'más', 'porque', 'uno', 'fuéramos', 'sentid', 'hube', 'tengas', 'estuvieseis', 'suyo', 'tuviesen', 'hay', 'hubiésemos', 'quienes', 'estas', 'del', 'ante', 'estaré', 'nada', 'cuando', 'tuya', 'mis', 'estada', 'estuvo', 'tú', 'esto', 'por', 'en', 'tuvieran', 'siente', 'esa', 'fueseis', 'vuestra', 'estaba', 'ni', 'seréis', 'haya', 'fuesen', 'fuerais', 'figura', 'ha', 'estabas', 'hubisteis', 'seré', 'o', 'fuisteis', 'estad', 'otras', 'eran', 'tendrá', 'lo', 'sus', 'fuese', 'estará', 'estarían', 'estabais', 'tuyos', 'mismo', 'un', 'habida', 'tuvieron', 'tuve', 'hubieseis', 'fuera', 'sintiendo', 'estaremos', 'eres', 'fue', 'otros', 'estuvimos', 'tendremos', 'hubimos', 'tenidos', 'os', 'fuésemos', 'tendríais', 'tienes', 'estuvieras', 'de', 'sí', 'teníais', 'vuestro', 'tengamos', 'suyas', 'quien', 'estáis', 'habéis', 'nosotros', 'tuvieses', 'para', 'los', 'he', 'estado', 'este', 'estuviese', 'estés', 'algo', 'todo', 'parte', 'nosotras', 'otro', 'tenías', 'tuviste', 'tuvieras', 'les', 'erais', 'suyos', 'poco', 'hayas', 'hayamos', 'teníamos', 'vosotros', 'al', 'esos', 'hasta', 'eras', 'estuvieron', 'fueron', 'hayan', 'habréis', 'hubiese', 'todos', 'seamos', 'muy', 'tuviéramos', 'tuvo', 'suya', 'éramos', 'estuvieses', 'hubieses', 'mas', 'estadas', 'tened', 'habré', 'nos', 'sería', 'han', 'habríamos', 'estaríamos', 'estuvisteis', 'mías', 'ya', 'nuestro', 'no', 'tabla', 'con', 'tendríamos', 'tenían', 'será', 'estando', 'que', 'ti', 'está', 'estados', 'habidas', 'tanto', 'están', 'estarán', 'le', 'tiene', 'teniendo', 'tenida', 'hubieron', 'tenido', 'sin', 'algunos', 'has', 'sea', 'sentidas', 'asimismo', 'durante', 'seas', 'hemos', 'estaréis', 'serás', 'mí', 'habías', 'vuestras', 'habido', 'serían', 'estemos', 'cual', 'hubierais', 'esta', 'hubiste', 'estuvieran', 'somos', 'sois', 'el', 'hubieran', 'yo', 'su', 'vuestros', 'hubiéramos', 'estaríais', 'ese', 'sobre', 'vosotras', 'mío', 'tuviera', 'son', 'tu', 'tendré', 'tienen', 'esas', 'estos', 'embargo', 'y', 'unos', 'una', 'habremos', 'tendrás', 'fuimos', 'estás', 'tuvisteis', 'habiendo', 'era', 'tenidas', 'nuestros', 'estarías', 'tengan', 'habrás', 'fueran', 'habrán', 'ser'}
    
    def process_data(self):
        self.df.Textos_espanol = self.df.Textos_espanol.map(self.preprocess_text)
        self.df.rename(columns={"Textos_espanol":"text"}, inplace = True)

    def preprocess_text(self, text):
        for key, value in self.multi_char_map.items():
            text = text.replace(key, value)
        text = self.url_pattern.sub('', text)
        text = text.replace('-',' ').replace('/',' ')
        text = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', '', text, re.I|re.A)
        text = text.lower().strip()
        tokens = [token.lemma_.split(" ")[0] for token in self.nlp(text) if token.text not in self.stop_words]
        tokens = [token for token in tokens if token not in self.stop_words and len(token)>2]
        text =  ' '.join(tokens)
        for key, value in self.single_char_map.items():
            text = text.replace(key, value)
        text = ' '.join(text.split())
        return text

    def write_data(self, processed_data_path):
        self.df.to_csv(processed_data_path, index=False)