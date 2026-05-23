# Advanced Features — AI Writing Assistant

## Tone adjustment without full regeneration

Apply style changes to existing content without running the full 4-step pipeline:

```python
@app.post("/adjust-tone")
async def adjust_tone(content: str, new_style: str, original_style: str = "professional"):
    llm = app.state.llm
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Rewrite the following text from {original_style} to {new_style} style. Preserve all facts and structure."),
        ("user", "{text}"),
    ])
    result = await (prompt | llm | parser).ainvoke({"text": content})
    return {"adjusted": result, "word_count": len(result.split())}
```

## Reading level analysis

```python
# reading_level.py
import re

def flesch_kincaid_grade(text: str) -> float:
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()
    syllable_count = sum(_count_syllables(w) for w in words)

    if not sentences or not words:
        return 0.0

    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = syllable_count / len(words)
    return 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59

def _count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:")
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def analyze_text(text: str) -> dict:
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    grade = flesch_kincaid_grade(text)
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
        "reading_grade_level": round(grade, 1),
        "estimated_reading_minutes": round(len(words) / 238, 1),
    }
```

## Key points extraction

```python
@app.post("/extract-key-points")
async def extract_key_points(content: str, n_points: int = 5):
    from pydantic import BaseModel

    class KeyPoints(BaseModel):
        points: list[str]
        one_line_summary: str

    llm = app.state.llm
    structured_llm = llm.with_structured_output(KeyPoints)
    prompt = f"Extract {n_points} key points and a one-line summary from:\n\n{content}"
    result = await structured_llm.ainvoke(prompt)
    return result.model_dump()
```

## SEO metadata generation

```python
@app.post("/generate-meta")
async def generate_meta(content: str, topic: str):
    from pydantic import BaseModel

    class SEOMeta(BaseModel):
        title: str
        meta_description: str
        keywords: list[str]
        slug: str

    llm = app.state.llm
    structured_llm = llm.with_structured_output(SEOMeta)
    prompt = (
        f"Generate SEO metadata for this content about: {topic}\n\n"
        f"Content (first 500 chars): {content[:500]}\n\n"
        f"Rules: title ≤ 60 chars, meta_description ≤ 160 chars, 5–8 keywords, slug is URL-safe."
    )
    result = await structured_llm.ainvoke(prompt)
    return result.model_dump()
```

---

[[02-implementation]] | [[04-evaluation]]
