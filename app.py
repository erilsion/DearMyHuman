import io
import time
import random
import hashlib
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import streamlit as st
from PIL import Image, ImageOps

from google import genai
from google.genai import types
from google.genai.errors import ClientError

DEFAULT_IMAGE_PATH = "images/default_pet_image.png"

# =========================================================
# Streamlit Config
# =========================================================
st.set_page_config(page_title="Dear, My Human", page_icon="🐾", layout="centered")
st.title("🐾 Dear, My Human")
st.caption("반려동물이 주인님께 편지를 가져왔어요.")

# =========================================================
# Secrets / Client
# =========================================================
API_KEY = st.secrets.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY가 설정되지 않았어요. (.streamlit/secrets.toml 또는 Streamlit Cloud Secrets)")
    st.stop()

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(api_version="v1"),
)

# 텍스트 생성 모델 (가볍고 빠른 모델)
LETTER_MODEL = "gemini-2.0-flash"

# 이미지 생성 (가능하면) + 폴백
IMAGE_MODEL_PRIMARY = "gemini-2.5-flash-image"
IMAGE_MODEL_FALLBACK = "imagen-4.0-generate-001"

# =========================================================
# Concurrency / Rate limiting helpers (for multi-user safety)
# =========================================================
@st.cache_resource
def get_api_semaphore():
    # 동시 API 호출 수 제한 (해커톤/Streamlit Cloud에서는 2 정도가 적당)
    return threading.Semaphore(2)

# (선택) 요청 간 최소 간격(너무 빠른 연타 방지)
@st.cache_resource
def get_rate_gate():
    # 최근 호출 시간을 저장해서 과도한 스파이크 완화
    return {"last_call_ts": 0.0}

def throttle_min_interval(min_interval_sec: float = 0.35):
    gate = get_rate_gate()
    now = time.time()
    dt = now - gate["last_call_ts"]
    if dt < min_interval_sec:
        time.sleep(min_interval_sec - dt)
    gate["last_call_ts"] = time.time()

def call_with_backoff(fn, max_tries=5, base=1.2):
    """
    429 RESOURCE_EXHAUSTED일 때만 지수 백오프로 재시도.
    """
    for i in range(max_tries):
        try:
            throttle_min_interval(0.20)  # 너무 짧은 시간 연속 호출 완화
            return fn()
        except ClientError as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                sleep_s = base * (2 ** i) + random.uniform(0, 0.6)
                time.sleep(sleep_s)
                continue
            raise
    # retries exhausted
    raise ClientError(429, {"error": {"message": "429 RESOURCE_EXHAUSTED (retries exceeded)"}})

# =========================================================
# Session State
# =========================================================
if "generated_image_bytes" not in st.session_state:
    st.session_state.generated_image_bytes = None
if "letter_text" not in st.session_state:
    st.session_state.letter_text = None
if "ready" not in st.session_state:
    st.session_state.ready = False
if "image_error" not in st.session_state:
    st.session_state.image_error = None
if "user_image_bytes" not in st.session_state:
    st.session_state.user_image_bytes = None
if "pet_name" not in st.session_state:
    st.session_state.pet_name = None

# 재호출 방지용
if "last_request_key" not in st.session_state:
    st.session_state.last_request_key = None

# 입력값 보관(이미지 버튼 눌렀을 때 정확히 다시 쓰려고)
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None

# =========================================================
# Helpers
# =========================================================
@dataclass
class PetInputs:
    name: str
    species: str
    personality: str
    age: str
    actions: str
    worries: str
    owner_message: str

def _safe_strip(x: Optional[str]) -> str:
    return (x or "").strip()

def make_request_key(inputs: PetInputs, image_bytes: bytes = b"") -> str:
    """
    같은 입력이면 같은 결과를 재사용하기 위한 키.
    이미지 bytes 포함 -> 사진까지 같을 때만 동일 처리.
    """
    h = hashlib.sha256()
    payload = "|".join([
        inputs.name, inputs.species, inputs.personality, inputs.age,
        inputs.actions, inputs.worries, inputs.owner_message
    ]).encode("utf-8")
    h.update(payload)
    h.update(image_bytes)
    return h.hexdigest()

def build_letter_prompt(inputs: PetInputs) -> str:
    # 기본값 가이드
    personality = _safe_strip(inputs.personality) or "아직 잘 모르겠지만 사랑이 많은"
    age = _safe_strip(inputs.age) or "어린"
    actions = _safe_strip(inputs.actions) or "함께 시간을 보내 주는 것"
    worries = _safe_strip(inputs.worries) or "요즘 마음이 조금 바빠 보이는 것"
    owner_message = _safe_strip(inputs.owner_message) or "항상 고마워."
    species = _safe_strip(inputs.species)
    species_line = f"- 반려동물 종류: {species} (가능하면 분위기/표현에 은은하게만 반영하고 단정하지 말 것)\n" if species else ""

    prompt = f"""
[반려동물 편지 모드 지침]
너는 이제 '{inputs.name}'(이)라는 반려동물이다.
너는 편지를 요청한 주인을 순수하게 사랑한다.
입력된 정보(이름/성격/나이/주인이 자주 해준 행동/걱정거리/주인의 말)를 바탕으로
주인에게 보내는 '짧은 손편지'를 작성하라.

[글의 톤/말투 규칙]
- '{personality}' 성격과 '{age}' 나이를 반영해 말투를 자연스럽게 정한다.
- 지나치게 유치하거나 과장된 아기말(“쨔쨔”, “앙”)은 피한다.
- 공감/위로/고마움이 중심이되, 밝은 희망으로 끝낸다.
- 사과가 필요하면 짧게, 하지만 죄책감을 과도하게 자극하지 않는다.

[내용 규칙]
- {species_line}- 주인이 자주 해준 행동: {actions} → 고마움을 구체적으로 표현한다.
- 걱정거리(고민): {worries} → 주인을 안심시키거나 함께 해결하자는 제안을 한다.
- 주인이 하고 싶은 말: {owner_message} → 다정하게 받아주고 따뜻하게 답한다.
- ‘의학/진단’처럼 단정 짓지 말고, 일반적인 조언 수준으로 부드럽게 말한다.

[출력 형식(반드시 지킬 것)]
1) 첫 줄: "주인님께," 또는 "OO에게," 같은 호칭(한 줄)
2) 본문: 3~6문장. 줄바꿈을 1~2번 넣어 손편지 느낌을 낸다.
3) 마무리 한 줄: 애정 표현(한 줄)
4) PS 한 줄: 짧고 귀엽게(한 줄)

[길이 제한]
- 전체 600자 이내(공백 포함)
- 위 규칙/지침/메타 설명을 출력에 포함하지 말 것. 오직 편지 본문만 출력.
""".strip()
    return prompt

def build_image_prompt(inputs: PetInputs) -> str:
    personality = _safe_strip(inputs.personality) or "cute and warm"
    age = _safe_strip(inputs.age) or "young"
    species = _safe_strip(inputs.species)
    species_hint = f'The pet is a "{species}".' if species else "The pet is a household pet."
    return f"""
Using the uploaded pet photo as reference, generate an illustration-like image.
{species_hint}
Scene: The pet "{inputs.name}" is returning home holding a letter in its mouth.
Mood: warm, cute, wholesome, cozy.
Style: soft illustration, clean composition, friendly lighting.
Details: reflect "{personality}" vibe and "{age}" age impression subtly.
Rules: NO text, NO letters readable, NO watermark, NO logos.
""".strip()

def load_default_image_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None

def clamp_text(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"

def reset_result_state():
    st.session_state.generated_image_bytes = None
    st.session_state.letter_text = None
    st.session_state.ready = False
    st.session_state.image_error = None
    st.session_state.pet_name = None
    st.session_state.last_inputs = None
    st.session_state.last_request_key = None

# =========================================================
# API calls (with safety)
# =========================================================
def generate_letter_text(prompt: str) -> str:
    def _do():
        resp = client.models.generate_content(
            model=LETTER_MODEL,
            contents=prompt,
        )
        return (resp.text or "").strip()

    sem = get_api_semaphore()
    with sem:
        return call_with_backoff(_do)

def generate_image_with_fallback(image_prompt: str, user_image_bytes: bytes) -> Tuple[Optional[bytes], Optional[str]]:
    """
    이미지 생성은 '옵션'. 실패해도 텍스트 UX는 끊기지 않도록
    (bytes=None, error=...) 형태로 반환.
    """
    generated_image_bytes = None
    image_error = None

    sem = get_api_semaphore()

    # 1) Primary: 이미지 참고 포함 생성 시도
    try:
        def _do_primary():
            return client.models.generate_content(
                model=IMAGE_MODEL_PRIMARY,
                contents=[
                    image_prompt,
                    types.Part.from_bytes(data=user_image_bytes, mime_type="image/png"),
                ],
            )

        with sem:
            resp_img = call_with_backoff(_do_primary, max_tries=3, base=1.0)

        # 응답에서 이미지 바이트 추출(방어적으로)
        for c in getattr(resp_img, "candidates", []) or []:
            content = getattr(c, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue
            for p in parts:
                inline = getattr(p, "inline_data", None)
                data = getattr(inline, "data", None) if inline else None
                if data:
                    generated_image_bytes = data
                    break
            if generated_image_bytes:
                break

    except Exception as e:
        image_error = f"primary image model failed: {e}"

    # 2) Fallback: Imagen - 모델 여러 개 자동 시도
    if generated_image_bytes is None:
        IMAGEN_FALLBACK_MODELS = [
            IMAGE_MODEL_FALLBACK,
            "imagen-4.0-generate-001",
            "imagen-3.0-generate-002",
            "imagen-3.0-generate-001",
        ]
        tried = set()

        for m in IMAGEN_FALLBACK_MODELS:
            if m in tried:
                continue
            tried.add(m)
            try:
                def _do_imagen():
                    return client.models.generate_images(model=m, prompt=image_prompt)

                with sem:
                    resp_imagen = call_with_backoff(_do_imagen, max_tries=3, base=1.0)

                generated_image_bytes = resp_imagen.generated_images[0].image.image_bytes
                break
            except Exception as e:
                image_error = (image_error or "") + f"\nimagen fallback failed ({m}): {e}"
                generated_image_bytes = None

    return generated_image_bytes, image_error

# =========================================================
# UI Inputs
# =========================================================
with st.form("pet_form"):
    st.subheader("반려동물 정보 입력")

    uploaded = st.file_uploader("사진 첨부 (선택)", type=["png", "jpg", "jpeg"])
    name = st.text_input("이름 (필수)", placeholder="예: 해피")
    species_choice = st.selectbox(
        "반려동물 종류 (선택)",
        ["선택 안 함", "강아지", "고양이", "토끼", "햄스터", "앵무새", "도마뱀", "거북이", "기타(직접 입력)"],
    )
    species_custom = ""
    if species_choice == "기타(직접 입력)":
        species_custom = st.text_input("어떤 반려동물인가요? (예: 페럿, 고슴도치, 물고기)", placeholder="예: 페럿")
    personality = st.text_input("성격", placeholder="예: 겁 많지만 애교 많음 / 츤데레 / 활발함")
    age = st.text_input("나이", placeholder="예: 3살 / 7개월")
    actions = st.text_area("주인이 자주 해준 행동", placeholder="예: 산책 자주 해줌, 간식 챙겨줌, 안아줌")
    worries = st.text_area("걱정거리(고민)", placeholder="예: 분리불안이 있는 것 같아 걱정돼")
    owner_message = st.text_area("하고 싶은 말(주인이 반려동물에게)", placeholder="예: 요즘 바빠서 미안해. 그래도 사랑해!")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        submitted = st.form_submit_button("✨ 편지 가져오게 하기", use_container_width=True)
    with col_b:
        cleared = st.form_submit_button("🧹 결과 지우기", use_container_width=True)

if cleared:
    reset_result_state()

if submitted:
    # 입력 검증
    if not _safe_strip(name):
        st.warning("이름은 꼭 넣어주세요! (나머지는 비워도 괜찮아요!)")
        st.stop()

    # 사용자 이미지 로드 + bytes 저장(대체 표시용)
    user_image_bytes = b""  # ✅ 사진 없을 수도 있으니 기본값
    if uploaded is not None:
        user_image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        max_side = 1024
        user_image.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        user_image.save(buf, format="PNG")
        user_image_bytes = buf.getvalue()

        st.image(user_image, caption="업로드한 사진", use_container_width=True)
    else:
        st.info("사진 없이도 편지를 만들 수 있어요 🐾 (그림 기능은 사진이 있을 때만 가능해요)")

    # 종(반려동물 종류) 최종 문자열 결정
    if species_choice == "선택 안 함":
        species_final = ""
    elif species_choice == "기타(직접 입력)":
        species_final = _safe_strip(species_custom)
    else:
        species_final = species_choice

    inputs = PetInputs(
        name=_safe_strip(name),
        species=_safe_strip(species_final),
        personality=_safe_strip(personality),
        age=_safe_strip(age),
        actions=_safe_strip(actions),
        worries=_safe_strip(worries),
        owner_message=_safe_strip(owner_message),
    )

    request_key = make_request_key(inputs, user_image_bytes)

    # 이미 같은 입력으로 결과가 있으면 재호출 방지
    if st.session_state.ready and st.session_state.last_request_key == request_key and st.session_state.letter_text:
        st.info("이미 편지를 가져왔어요! 아래에서 확인해줘 🐾")
        st.stop()

    # 새 요청 시작 -> 결과 초기화(텍스트는 새로 만들 거라)
    st.session_state.generated_image_bytes = None
    st.session_state.image_error = None
    st.session_state.ready = False

    st.session_state.user_image_bytes = user_image_bytes
    st.session_state.pet_name = inputs.name
    st.session_state.last_inputs = inputs
    st.session_state.last_request_key = request_key

    letter_prompt = build_letter_prompt(inputs)

    with st.spinner(f"{inputs.name}: 편지를 가져오고 있어요! 잠시만 기다려주세요~"):
        try:
            # 텍스트는 무조건
            letter_text = generate_letter_text(letter_prompt)
            letter_text = clamp_text(letter_text, 600)
        except Exception:
            st.warning("지금 요청이 몰려서 편지를 가져오지 못했어요 🥲 10~30초 후에 다시 눌러줘!")
            st.stop()

    st.session_state.letter_text = letter_text
    st.session_state.ready = True

# =========================================================
# Results
# =========================================================
if st.session_state.ready:
    pet_name = st.session_state.pet_name or "반려동물"
    st.subheader("📮 반려동물이 편지를 가져왔어요!")

    # 이미지 표시
    if st.session_state.generated_image_bytes:
        st.image(st.session_state.generated_image_bytes, use_container_width=True)
    else:
        st.info("우선 편지를 먼저 가져왔어요. (그림은 선택하면 바로 그려줄게요. 🐾)")
        if st.session_state.user_image_bytes:
            st.image(
                st.session_state.user_image_bytes,
                caption="대신, 제 사진을 보여줄게요!",
                use_container_width=True,
            )
        else:
            default_bytes = load_default_image_bytes(DEFAULT_IMAGE_PATH)
        if default_bytes:
            st.image(default_bytes, caption="멍멍! 제가 편지를 배달하러 왔어요. 🐾", use_container_width=True)
        else:
            st.info("기본 이미지 파일이 없어서 표시할 수 없어요. images/default_pet_image.png 경로를 확인해주세요!")

        # 이미지 생성은 선택 버튼으로만!
        if st.button("🖼️ 그림도 같이 받을래요 (선택)", use_container_width=True):
            if not st.session_state.last_inputs or not st.session_state.user_image_bytes:
                st.warning("입력 정보가 없어서 그림을 만들 수 없어요. 다시 한 번 제출해주세요!")
                st.stop()

            with st.spinner(f"{pet_name}: 그림을 그리는 중이에요..."):
                img_prompt = build_image_prompt(st.session_state.last_inputs)
                img_bytes, img_err = generate_image_with_fallback(
                    image_prompt=img_prompt,
                    user_image_bytes=st.session_state.user_image_bytes,
                )

            st.session_state.generated_image_bytes = img_bytes
            st.session_state.image_error = img_err
            st.rerun()

        # 개발용 로그(심사 때는 접혀있어서 깔끔)
        if st.session_state.image_error:
            with st.expander("이미지 생성 로그(개발용)"):
                st.code(st.session_state.image_error)

    # 편지는 무조건 제공
    if st.button("💌 편지받기", use_container_width=True):
        st.subheader("편지")
        st.write(st.session_state.letter_text or "")
