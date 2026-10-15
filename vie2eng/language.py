"""Language detection and bidirectional routing without model dependencies."""

import re


VI_CHARS_REGEX = re.compile(
    r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ'
    r'ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]',
    re.IGNORECASE
)

VI_COMMON_WORDS = {
    'và', 'của', 'là', 'có', 'được', 'trong', 'với', 'một', 'các', 'người',
    'không', 'đã', 'này', 'cho', 'những', 'khi', 'thì', 'mà', 'ở', 'từ',
    'tôi', 'bạn', 'chúng', 'anh', 'chị', 'em', 'họ', 'gì', 'thế', 'nào',
    'nhiều', 'rất', 'về', 'lại', 'như', 'đến', 'ra', 'phải', 'biết', 'sẽ',
    'cũng', 'làm', 'sau', 'trên', 'qua', 'năm', 'ngày', 'nói', 'theo', 'hơn',
    'việt', 'nam', 'bình', 'xin', 'chào', 'cảm', 'ơn', 'khỏe'
}

EN_COMMON_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'hello', 'how', 'are', 'is', 'love', 'fast', 'final', 'exam', 'very'
}

VI_UNACCENTED_WORDS = {
    'xin', 'chao', 'cam', 'on', 'tam', 'biet', 'toi', 'ban', 'chung',
    'anh', 'chi', 'em', 'ho', 'gi', 'the', 'nao', 'troi', 'dep', 'hoc',
    'sinh', 'vien', 'yeu', 'rat', 'nhieu', 'khong', 'duoc', 'nguoi',
    'viet', 'nam', 'khi', 'nao', 'o', 'dau', 'lam', 'sao'
}


def detect_language(text: str) -> str:
    """Detect if text is Vietnamese ('vi') or English ('en')."""
    if not text.strip():
        return 'en'

    # 1. Check for specific Vietnamese diacritics
    if VI_CHARS_REGEX.search(text):
        return 'vi'

    # 2. Check word tokens against dictionary
    words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
    if not words:
        return 'en'

    vi_matches = sum(1 for w in words if w in VI_COMMON_WORDS or w in VI_UNACCENTED_WORDS)
    en_matches = sum(1 for w in words if w in EN_COMMON_WORDS)

    if vi_matches > en_matches:
        return 'vi'
    return 'en'
