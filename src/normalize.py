import unicodedata

# 선박명 정규화
# 같은 이름을 비교할 때 겉모양만 다른 부분을 통일하는 과정
def normalize_name(name) -> str:

    # None이나 결측값이면 -> 빈 문자열 반환
    # isinstance(name, str) : name이 str이면 True, 아니면 False
    if not isinstance(name, str):
        return "" 

    # 유니코드 형태 통일
    # 영문은 대문자로 통일
    # 공백, 하이픈, 괄호, 슬래시 같은 기호 제거
    name = unicodedata.normalize("NFKC", name).upper()

    return "".join(ch for ch in name if ch.isalnum())