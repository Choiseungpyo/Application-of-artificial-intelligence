import pandas as pd

ADULT_KEYWORDS = [
    "adult",
    "hentai",
    "eroge",
    "explicit",
    "sexual"
]

def is_adult_content(row: pd.Series) -> bool:
    """
    Pandas DataFrame의 행(row) 데이터를 받아 성인 콘텐츠 여부를 판별합니다.
    검사 대상 필드: title, genres, tags, description (존재하는 경우에만)
    대소문자 차이, 공백, None 값에 안전하게 동작합니다.
    """
    # 안전하게 문자열로 변환하는 헬퍼 함수
    def _safe_str(val):
        if pd.isna(val) or val is None:
            return ""
        return str(val).strip().lower()

    # 확인할 필드들
    fields_to_check = ['title', 'genres', 'tags', 'description']
    
    # 텍스트 병합
    combined_text = " ".join([
        _safe_str(row.get(field)) 
        for field in fields_to_check 
        if field in row.index
    ])
    
    # 키워드 검사
    return any(keyword in combined_text for keyword in ADULT_KEYWORDS)

def filter_adult_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    원본 DataFrame에서 Adult 계열 데이터를 제외한 새 DataFrame을 반환합니다.
    사용 예시:
        df = pd.read_csv("metadata.csv")
        filtered_df = filter_adult_data(df)
        filtered_df.to_csv("filtered_metadata.csv", index=False)
    """
    if df.empty:
        return df
        
    # is_adult_content가 False인 행만 유지
    mask = df.apply(lambda row: not is_adult_content(row), axis=1)
    filtered_df = df[mask].copy()
    
    print(f"[필터링 완료] 원본 데이터 수: {len(df)} -> 필터 적용 후: {len(filtered_df)}")
    print(f"제외된 Adult 데이터 수: {len(df) - len(filtered_df)}")
    
    return filtered_df
