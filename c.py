import random

# 파일 이름 범위 지정
start_idx = 0
end_idx = 100

# 임의 문장 리스트
dummy_sentences = [
    "안녕하세요", "반가워요", "좋은 하루 보내세요", "오늘 날씨 어때요?",
    "점심은 드셨어요?", "잘 지내시죠?", "지금 뭐 하고 계세요?", "도움이 필요하신가요?",
    "연락 주셔서 감사합니다", "편안한 하루 되세요", "무엇을 도와드릴까요?",
    "곧 다시 연락드릴게요", "다음에 또 뵈어요", "건강 조심하세요", "재밌는 이야기 해드릴까요?",
    "저는 AI입니다", "지금 바로 시작할까요?", "잠시만 기다려 주세요", "감사합니다",
    "축하드려요", "좋은 소식이 있네요", "다 잘될 거예요", "희망을 가지세요", "믿고 있어요"
]

# 결과 저장
output_lines = []

for i in range(start_idx, end_idx + 1):
    filename = f"1_{i:04d}.wav"
    text = random.choice(dummy_sentences)
    output_lines.append(f"{filename}|{text}")

# 결과 출력 or 저장
output_text = "\n".join(output_lines)
print(output_text)

# 저장하고 싶다면 아래 코드 추가
# with open("wav_text_pairs.txt", "w", encoding="utf-8") as f:
#     f.write(output_text)
