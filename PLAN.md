# K-pop 아이돌 그룹의 생존 분석 — "7년 징크스"는 실재하는가?

> 21501755 AI and Data Analysis Practice · Final Tutorial 기획서
> 마감: 2026-12-21(월) 16:00 · 제출물: GitHub Pages URL 1개

## 1. 한 줄 요약

1990년대부터 2020년대까지 데뷔한 한국 아이돌 그룹 약 600개의 데뷔·해체 기록을 Wikipedia에서 수집해,
**의학 통계에서 쓰는 생존 분석(Survival Analysis)** 으로 그룹의 "수명"을 분석하고,
데뷔 시점 정보만으로 5년 후 활동 여부를 예측하는 머신러닝 모델을 만든다.

## 2. 왜 이 주제인가 (Motivation 항목 대응)

- **일반적 중요성**: K-pop은 연 매출 수조 원 규모 산업이고, 그룹 해체는 팬·소속사·멤버 모두에게 큰 사건. 7년 표준계약 만료 시점에 해체가 몰린다는 "7년 징크스"는 업계 통념이지만 숫자로 검증된 적이 거의 없다.
- **데이터 사이언스 관점**: 아직 활동 중인 그룹은 "언제 해체할지 모른다"(right-censoring). 이런 데이터는 평균 수명 같은 단순 통계로 다루면 틀린다. 생존 분석이 정확히 이 문제를 풀기 위해 만들어진 도구이고, 의학 밖에서 적용한 사례를 보여주는 것이 교육적 가치.

## 3. 연구 질문과 가설

| # | 질문 | 가설 (H1) | 검정 방법 |
|---|------|-----------|-----------|
| Q1 | 아이돌 그룹의 "수명" 분포는 어떤 모양인가? | 7년 근처에 해체 위험(hazard)이 뾰족하게 솟는다 | Kaplan-Meier 생존곡선 + hazard 추정 |
| Q2 | 대형 기획사 그룹이 더 오래 살아남는가? | 빅4(SM·YG·JYP·HYBE) 소속 그룹의 생존곡선이 유의하게 높다 | Log-rank test |
| Q3 | 보이그룹과 걸그룹의 생존률이 다른가? | 걸그룹의 해체 위험이 더 높다 (통념 검증) | Log-rank test |
| Q4 | 멤버 수가 많으면 해체 위험이 높은가? | 멤버 수와 해체 위험은 양의 상관 | Cox 비례위험 모형 |
| Q5 | 세대(1~5세대)별로 생존 패턴이 달라졌는가? | 최근 세대는 7년 징크스가 약해졌다 (계약·재계약 관행 변화) | 세대별 KM 곡선 비교 |
| Q6 | 데뷔 시점 정보만으로 5년 후 활동 여부를 예측할 수 있나? | 랜덤보다 유의하게 나은 예측이 가능하다 | 분류 모델 (로지스틱, 랜덤포레스트, 그래디언트부스팅) |

## 4. 데이터

### 4.1 출처
| 데이터 | 출처 | 수집 방식 | 비고 |
|--------|------|-----------|------|
| 그룹 목록 + 데뷔 연도 | Wikipedia `List of South Korean idol groups (1990s/2000s/2010s/2020s)` | MediaWiki API (`action=parse`) | 약 600개 그룹. "Wikipedia 문서가 있는 그룹만" → 선택 편향은 한계로 명시 |
| 활동 기간, 소속사, 멤버, 장르 | 각 그룹 Wikipedia 문서의 infobox | MediaWiki API, 문서당 1회 요청 | `years_active` 필드에 해체·휴식·재결합이 모두 기록됨 |
| 성별(보이/걸그룹) | 문서의 Wikipedia 카테고리 | 위와 같은 요청에 포함 | `South Korean boy bands` / `South Korean girl groups` |
| 인기 지표(보조) | Wikipedia 페이지뷰 API, 문서 길이 | REST API | 데뷔 시점에는 알 수 없는 정보라 예측 모델엔 쓰지 않고 EDA에만 사용 |

크롤링이 아닌 **공식 API**를 사용하므로 약관·차단 문제가 없고, 누구나 스크립트를 다시 실행해 재현할 수 있다 (Code 항목의 reproducible).

### 4.2 변수 설계
- `debut_year`, `debut_date` — 리스트 페이지의 연도 섹션 + infobox
- `end_year` — `years_active`의 마지막 종료 연도. "present"면 결측 → **검열(censored) = 1**
- `duration_years` — 관측 종료(해체 or 2026년 현재) − 데뷔
- `event` — 해체 여부 (1 = 해체, 0 = 활동 중/검열)
- `gender` — boy / girl / mixed
- `first_label`, `big4` — 데뷔 소속사, 빅4 여부
- `n_members_debut` — 데뷔 멤버 수 (current + past)
- `generation` — 1세대(~2002), 2세대(2003–2011), 3세대(2012–2017), 4세대(2018–2022), 5세대(2023~)
- `had_hiatus`, `reunited` — `years_active`에 구간이 2개 이상이면 True

### 4.3 데이터의 함정 (튜토리얼에서 직접 다룰 것)
- "해체" 정의: 공식 해체 vs 무기한 활동중단 vs 멤버 재계약 불발 — Wikipedia `years_active`가 끝났으면 해체로 간주하고, 재결합은 별도 플래그로 처리
- 생존자 편향: 유명하지 않은 그룹은 Wikipedia 문서가 없어 데이터에 안 들어온다 → 결론은 "문서가 있을 정도의 그룹 중에서"로 한정
- 최근 데뷔 그룹은 관측 기간이 짧다 → 생존 분석이 바로 이 문제를 처리함 (이게 생존 분석을 쓰는 이유라는 점을 강조)

## 5. 노트북 구성 (루브릭 6개 항목 매핑)

```
00  제목 · 이름 · 목차
01  Introduction — 7년 징크스란? 왜 데이터로 봐야 하나?         [Motivation]
02  Survival Analysis 5분 입문 — 검열, KM 곡선, log-rank, Cox   [Understanding, Other resources]
03  Data Collection — MediaWiki API로 600개 그룹 수집            [Code]
04  Data Cleaning — years_active 파싱, 소속사 정규화, 결측 처리   [Code, Prose]
05  EDA — 연도별 데뷔 수, 세대 분포, 소속사 분포, 멤버 수 분포     [Understanding]
06  KM 생존곡선 — 전체 / 성별 / 빅4 / 세대                       [핵심 그림]
07  Hazard 곡선 — 7년에 정말 피크가 있나?                        [핵심 결과]
08  Log-rank test — 집단 간 차이 검정                           [Hypothesis testing]
09  Cox 비례위험 모형 — 어떤 요인이 해체 위험을 높이나             [Hypothesis testing]
10  ML — 데뷔 정보로 5년 생존 예측 (train/test, 교차검증, 특성중요도) [Machine learning]
11  Discussion — 결과 해석, 한계, "7년 징크스"에 대한 결론         [Prose]
12  Further Reading — 생존분석 교재, lifelines 문서, K-pop 산업 보고서 [Other resources]
```

분량 목표: 산문 2,000단어 이상(영어), 코드 250줄 이상 — 요구치(1,500 / 150)의 약 1.5배.

## 6. 기술 스택

- Python 3.11 · `pandas` · `requests` · `matplotlib` / `seaborn` · `lifelines`(생존 분석) · `scikit-learn`
- Jupyter Notebook → `jupyter nbconvert --to html` → GitHub Pages (`username.github.io`)

## 7. 일정 (오늘 2026-10-30 기준, 마감까지 7주)

| 주차 | 기간 | 할 일 |
|------|------|-------|
| 1 | 10/30 – 11/05 | 데이터 수집 스크립트 완성, 원본 데이터 확보 (`data/raw/`) |
| 2 | 11/06 – 11/12 | 정제 · 변수 생성 · EDA 노트북 섹션 03~05 |
| 3 | 11/13 – 11/19 | 생존 분석 섹션 06~09, 핵심 그림 확정 |
| 4 | 11/20 – 11/26 | ML 섹션 10, 모델 비교 |
| 5 | 11/27 – 12/03 | 산문 작성 (01, 02, 11, 12), 참고 링크 정리 |
| 6 | 12/04 – 12/10 | GitHub Pages 배포, 디자인 다듬기, 모바일에서 읽어보기 |
| 7 | 12/11 – 12/18 | 리뷰 · 수정 · 여유분. **12/18까지 최종 제출** (마감 3일 전) |

## 8. 폴더 구조

```
AI데분 프로젝트/
├── PLAN.md                  ← 이 문서
├── requirements.txt
├── scripts/
│   └── collect_data.py      ← Wikipedia API 수집 스크립트 (재현 가능)
├── data/
│   ├── raw/                 ← API 응답 원본 (JSON), 재실행 시 캐시로 사용
│   └── processed/           ← 분석용 CSV
├── notebooks/
│   └── kpop_survival.ipynb  ← 최종 튜토리얼
└── docs/                    ← GitHub Pages로 올릴 HTML export
```
