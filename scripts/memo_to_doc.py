#!/usr/bin/env python3
"""
음성 메모 텍스트를 마크다운 문서로 변환하는 스크립트
다양한 템플릿(메모, 회의록, 아이디어)을 지원합니다.
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# 문서 템플릿 정의
TEMPLATES = {
    "memo": {
        "name": "메모",
        "structure": """\
# {title}

- **날짜**: {date}
- **작성 방법**: 음성 메모 (자동 변환)

---

## 내용

{content}

---

> 이 문서는 음성 메모에서 자동 생성되었습니다. ({timestamp})
""",
    },
    "meeting": {
        "name": "회의록",
        "structure": """\
# 회의록: {title}

- **날짜**: {date}
- **작성 방법**: 음성 녹음 자동 변환

---

## 참석자

- (참석자 정보를 입력하세요)

## 안건

- (안건을 정리하세요)

## 논의 내용

{content}

## 결정사항

- (결정사항을 정리하세요)

## 액션 아이템

| 담당자 | 내용 | 기한 |
|--------|------|------|
| - | - | - |

---

> 이 회의록은 음성 녹음에서 자동 생성되었습니다. ({timestamp})
""",
    },
    "idea": {
        "name": "아이디어 노트",
        "structure": """\
# 아이디어: {title}

- **날짜**: {date}
- **작성 방법**: 음성 메모 (자동 변환)

---

## 핵심 아이디어

(첫 문단이 핵심 아이디어를 요약합니다)

## 배경

(아이디어의 배경과 맥락)

## 상세 내용

{content}

## 다음 단계

- [ ] (구체적인 다음 행동을 정리하세요)

---

> 이 문서는 음성 메모에서 자동 생성되었습니다. ({timestamp})
""",
    },
}


def clean_text(text: str) -> str:
    """텍스트 정리: 불필요한 공백, 반복 제거"""
    # 연속 빈 줄을 하나로
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 줄 앞뒤 공백 정리
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def format_paragraphs(text: str) -> str:
    """텍스트를 문단 단위로 포맷팅"""
    lines = text.split("\n")
    formatted = []
    current_paragraph = []

    for line in lines:
        if line.strip() == "":
            if current_paragraph:
                formatted.append(" ".join(current_paragraph))
                formatted.append("")
                current_paragraph = []
        else:
            current_paragraph.append(line.strip())

    if current_paragraph:
        formatted.append(" ".join(current_paragraph))

    return "\n".join(formatted)


def generate_title(text: str) -> str:
    """텍스트에서 자동으로 제목 생성"""
    # 첫 문장을 제목으로 사용 (최대 30자)
    first_line = text.strip().split("\n")[0].strip()
    # 문장 끝 부호 제거
    first_line = re.sub(r"[.。!?！？]+$", "", first_line)

    if len(first_line) > 30:
        # 30자에서 가장 가까운 단어 경계에서 자름
        truncated = first_line[:30]
        # 한국어는 단어 경계가 공백
        last_space = truncated.rfind(" ")
        if last_space > 10:
            truncated = truncated[:last_space]
        return truncated + "..."

    if not first_line:
        return f"음성 메모 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    return first_line


def convert_to_document(
    text: str,
    template: str = "memo",
    title: str = None,
) -> str:
    """
    텍스트를 마크다운 문서로 변환합니다.

    Args:
        text: 변환할 텍스트
        template: 템플릿 이름 (memo, meeting, idea)
        title: 문서 제목 (None이면 자동 생성)

    Returns:
        마크다운 형식 문서
    """
    if template not in TEMPLATES:
        print(f"알 수 없는 템플릿: {template}")
        print(f"사용 가능: {', '.join(TEMPLATES.keys())}")
        sys.exit(1)

    # 텍스트 정리
    cleaned = clean_text(text)
    formatted = format_paragraphs(cleaned)

    # 제목 생성
    if title is None:
        title = generate_title(cleaned)

    # 날짜/시간
    now = datetime.now()
    date_str = now.strftime("%Y년 %m월 %d일 (%a)")
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # 템플릿 적용
    doc = TEMPLATES[template]["structure"].format(
        title=title,
        date=date_str,
        content=formatted,
        timestamp=timestamp_str,
    )

    return doc


def main():
    parser = argparse.ArgumentParser(description="음성 메모 → 마크다운 문서 변환")
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="입력 텍스트 파일 경로 (기본: stdin)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="출력 마크다운 파일 경로 (기본: stdout)",
    )
    parser.add_argument(
        "--title", "-t",
        default=None,
        help="문서 제목 (기본: 자동 생성)",
    )
    parser.add_argument(
        "--template",
        choices=list(TEMPLATES.keys()),
        default="memo",
        help="문서 템플릿 (기본: memo)",
    )
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="사용 가능한 템플릿 목록 표시",
    )
    args = parser.parse_args()

    # 템플릿 목록 표시
    if args.list_templates:
        print("사용 가능한 템플릿:")
        for key, tmpl in TEMPLATES.items():
            print(f"  {key}: {tmpl['name']}")
        sys.exit(0)

    # 입력 텍스트 읽기
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"파일을 찾을 수 없습니다: {args.input}")
            sys.exit(1)
        text = input_path.read_text(encoding="utf-8")
    else:
        if sys.stdin.isatty():
            print("텍스트를 입력하세요 (Ctrl+D로 종료):")
        text = sys.stdin.read()

    if not text.strip():
        print("입력 텍스트가 비어 있습니다.")
        sys.exit(1)

    # 문서 생성
    document = convert_to_document(text, args.template, args.title)

    # 출력
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(document, encoding="utf-8")
        print(f"문서 생성 완료: {args.output}")
        print(f"  템플릿: {TEMPLATES[args.template]['name']}")
        print(f"  크기: {len(document)}자")
    else:
        print(document)


if __name__ == "__main__":
    main()
