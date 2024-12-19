import contractions
import re
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords

RU_STOPWORDS = stopwords.words("russian")
EN_STOPWORDS = stopwords.words("english")

def preprocess_text(text: str) -> str:
    """
    Функция для предобработки текста.
    """
    if not isinstance(text, str):
        return ''

    # Приводим текст к нижнему регистру
    text = text.lower()

    # Разворачиваем сокращения: текст часто содержит конструкции на англ. языке, вроде "don't" или "won't"
    text = contractions.fix(text)

    # Разделяем "слипшиеся" с пунктуацией слова
    text = re.sub(r'\s*[.,;:!?…-]+\s*', ' ', text)
    
    # Удаляем пунктуацию, не разделяя слова и знаки препинания
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s]', '', text)

    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    # Удаляем стоп-слова
    text = ' '.join([word for word in text.split() if word not in RU_STOPWORDS or word not in EN_STOPWORDS])
    
    # Заменяем дефисы на пробелы 
    text = re.sub(r'(\w+)-(\w+)', r'\1 \2', text)

    # Создаем экземпляр морфологического анализатора
    morph = MorphAnalyzer()

    # Создание стеммера
    stemmer = SnowballStemmer("english", ignore_stopwords=True)

    # Токенизируем текст для более точного разделения на слова
    words = word_tokenize(text, language='russian')

    # Инициализация списка для нормализованных слов
    processed_words = []

    for word in words:
        # Определяем язык слова
        if bool(re.fullmatch(r'[a-zA-Z]+', word)):
            # Выделим основу слова
            parsed_word = stemmer.stem(word)
            processed_words.append(parsed_word)             
        else:
            # Применяем лемматизацию
            parsed_word = morph.parse(word)[0]
            processed_words.append(parsed_word.normal_form)
        
    # Соединяем обработанные слова в строку
    return ' '.join(processed_words)