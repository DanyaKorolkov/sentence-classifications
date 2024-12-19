import contractions
import unicodedata
import inflect
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re

def preprocess_text(text: str) -> str:
    """
    Функция для предобработки текста.
    """
    if not isinstance(text, str):
        return ''

    # Приводим текст к нижнему регистру
    text = text.lower()

    # Разворачиваем сокращения: текст часто содержит конструкции вроде "don't" или "won't", поэтому развернём подобные сокращения
    text = contractions.fix(text)

    # Преобразование символов с диакритическими знаками к ASCII-символам: используем функцию normalize из модуля unicodedata и преобразуем символы с диакритическими знаками к ASCII-символам
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Удаляем пунктуацию, не разделяя слова и знаки препинания
    text = re.sub(r'[^\w\s]', '', text)

    # Записываем числа прописью: 4 превращается в "four"
    temp = inflect.engine()
    words = []
    for word in text.split():
        if word.isdigit():
            words.append(temp.number_to_words(word))
        else:
            words.append(word)
    text = ' '.join(words)

    # Токенизируем текст для более точного разделения на слова
    text = word_tokenize(text, language='english')

    # Фильтрация слов, состоящих только из подчеркиваний
    words = [word for word in text if not re.match(r'^_+$', word)]

    # Создание стеммера
    stemmer = SnowballStemmer("english", ignore_stopwords=True)

    # Инициализация списка для нормализованных слов
    processed_words = []

    for word in words:
        # Выделим основу слова
        word = stemmer.stem(word)
        processed_words.append(word)

    # Соединяем обработанные слова в строку
    return ' '.join(processed_words)