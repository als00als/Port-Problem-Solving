def is_korean(name) -> bool:

    # isinstance :  어떤 객체가 특정 클래스나 데이터 타입의 인스턴스인지 확인하여 
    # 참(True) 또는 거짓(False)을 반환하는 내장 함수
    if not isinstance(name, str):
        return False
    
    return any("가" <= char <= "힣" for char in name)

def calculate_score(true_ids, predicted_ids, vessel_names) -> dict:
    # 한글 여부는 이름에 [가=힣]이 하나라도 포함되는지로 판별
    # 전체 4개 중 3개 정답
    # 비한글 이름 2개 모두 정답
    # 최종 가중 점수가 0.7 * korean + 0.3 * non_korean으로 계산되는지
    # 한글 행이나 비한글 행이 없는 경우 오류가 나지 않는지
    
    total_correct = 0
    total_count = 0
    non_korean_correct = 0
    non_korean_count = 0
    korean_correct = 0
    korean_count = 0
    
    for true_id, predict_id, vessel_name in zip(true_ids, predicted_ids, vessel_names):
        total_count += 1
        correct = true_id == predict_id
        korean_name = is_korean(vessel_name)

        if correct:
            total_correct += 1

        if korean_name :
            korean_count += 1
            if correct :
                korean_correct += 1
        else:
            non_korean_count += 1
            if correct:
                non_korean_correct += 1

    # 전체정확도
    overall = total_correct / total_count if total_count else 0.0

    # 한글 선박명 정확도
    korean = (korean_correct / korean_count if korean_count else 0.0)

    # 비한글 선박명 정확도
    non_korean = (non_korean_correct / non_korean_count if non_korean_count else 0.0)

    weighted = 0.7 * korean + 0.3 * non_korean
    
    return {
        "overall": overall,
        "korean": korean,
        "non_korean": non_korean,
        "weighted": round(weighted, 3),
    }