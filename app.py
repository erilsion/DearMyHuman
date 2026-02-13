import io
import time
import random
import hashlib
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import streamlit as st
from PIL import Image, ImageOps

if ENABLE_IMAGE:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage

from google import genai
from google.genai import types
from google.genai.errors import ClientError

DEFAULT_IMAGE_PATH = "images/default_pet_image.png"
ENABLE_IMAGE = False

# =========================================================
# Streamlit Config
# =========================================================
st.set_page_config(page_title="Dear, My Human", page_icon="🐾", layout="centered")
st.title("🐾 Dear, My Human")
st.caption("반려동물이 주인님께 편지를 가져왔어요.")

# =========================================================
# Secrets / Clients
# =========================================================
API_KEY = st.secrets.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY가 설정되지 않았어요. (.streamlit/secrets.toml 또는 Streamlit Cloud Secrets)")
    st.stop()

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(api_version="v1"),
)

LETTER_MODEL = "gemini-2.0-flash"
VISION_MODEL = "gemini-2.0-flash"

GCP_PROJECT_ID = st.secrets.get("GCP_PROJECT_ID")
GCP_LOCATION = st.secrets.get("GCP_LOCATION", "us-central1")
IMAGEN_GENERATE_MODEL = st.secrets.get("IMAGEN_GENERATE_MODEL", "imagen-3.0-generate-002")
IMAGEN_EDIT_MODEL = st.secrets.get("IMAGEN_EDIT_MODEL", "imagen-3.0-edit-001")

if not GCP_PROJECT_ID:
    st.error("GCP_PROJECT_ID가 설정되지 않았어요. (secrets.toml / Streamlit Cloud Secrets)")
    st.stop()

@st.cache_resource
def get_imagen_models():
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    gen = ImageGenerationModel.from_pretrained(IMAGEN_GENERATE_MODEL)

    edit = None
    try:
        edit = ImageGenerationModel.from_pretrained(IMAGEN_EDIT_MODEL)
    except Exception:
        edit = None

    return gen, edit

# =========================================================
# Concurrency / Rate limiting
# =========================================================
@st.cache_resource
def get_api_semaphore():
    return threading.Semaphore(2)

@st.cache_resource
def get_rate_gate():
    return {"last_call_ts": 0.0}

def throttle_min_interval(min_interval_sec: float = 0.35):
    gate = get_rate_gate()
    now = time.time()
    dt = now - gate["last_call_ts"]
    if dt < min_interval_sec:
        time.sleep(min_interval_sec - dt)
    gate["last_call_ts"] = time.time()

def call_with_backoff(fn, max_tries=5, base=1.2):
    for i in range(max_tries):
        try:
            throttle_min_interval(0.20)
            return fn()
        except ClientError as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                sleep_s = base * (2 ** i) + random.uniform(0, 0.6)
                time.sleep(sleep_s)
                continue
            raise
    raise ClientError(429, {"error": {"message": "429 RESOURCE_EXHAUSTED (retries exceeded)"}})

# =========================================================
# Session State
# =========================================================
for k, v in {
    "generated_image_bytes": None,
    "letter_text": None,
    "ready": False,
    "image_error": None,
    "user_image_bytes": b"",
    "pet_name": None,
    "last_request_key": None,
    "last_inputs": None,
    "generation_seed": 0,  # 같은 입력이라도 seed 바꾸면 새 결과
    "last_generation_seed": None,  # 디버그/표시용(선택)
}.items():
    if k not in st.session_state:
        st.session_state[k] = v
if "generation_seed" not in st.session_state:
    st.session_state.generation_seed = 0
if "regenerate_requested" not in st.session_state:
    st.session_state.regenerate_requested = False

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

def make_request_key(inputs: PetInputs, image_bytes: bytes = b"", seed: int = 0) -> str:
    h = hashlib.sha256()
    payload = "|".join([
        inputs.name, inputs.species, inputs.personality, inputs.age,
        inputs.actions, inputs.worries, inputs.owner_message,
        str(seed),  # ✅ seed 포함
    ]).encode("utf-8")
    h.update(payload)
    h.update(image_bytes)
    return h.hexdigest()

def clamp_text(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"

def load_default_image_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None

def build_letter_prompt(inputs: PetInputs, seed: int) -> str:
    personality = _safe_strip(inputs.personality) or "아직 잘 모르겠지만 사랑이 많은"
    age = _safe_strip(inputs.age) or "어린"
    actions = _safe_strip(inputs.actions) or "함께 시간을 보내 주는 것"
    worries = _safe_strip(inputs.worries) or "요즘 마음이 조금 바빠 보이는 것"
    owner_message = _safe_strip(inputs.owner_message) or "항상 고마워."
    species = _safe_strip(inputs.species)
    species_line = f"- 반려동물 종류: {species} (가능하면 분위기/표현에 은은하게만 반영하고 단정하지 말 것)\n" if species else ""

    return f"""
[반려동물 편지 모드 지침]
너는 이제 '{inputs.name}'(이)라는 반려동물이다.
너는 편지를 요청한 주인을 순수하게 사랑한다.
입력된 정보(이름/성격/나이/주인이 자주 해준 행동/걱정거리/주인의 말)를 바탕으로
주인에게 보내는 '짧은 손편지'를 작성하라.

[글의 톤/말투 규칙]
- '{personality}' 성격과 '{age}' 나이를 반영해 말투를 자연스럽게 정한다.
- 지나치게 유치하거나 과장된 아기말(“쨔쨔”, “앙”)은 피한다.
- 공감/위로/고마움이 중심이되, 밝은 희망으로 끝낸다.

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
- 위 지침을 출력하지 말 것. 오직 편지 본문만 출력.

[INTERNAL_VARIATION_METADATA]
VARIATION_SEED: {seed}
- Do NOT output this metadata.
- Do NOT mention "seed" or "variation".

""".strip()

def analyze_pet_photo_to_visual_desc(user_image_bytes: bytes) -> str:
    if not user_image_bytes:
        return ""

    vision_prompt = """
You are extracting visual facts from a pet photo for identity consistency in illustration.

CRITICAL RULES:
- Describe ONLY what is clearly visible in the image.
- Do NOT guess, assume, infer, or embellish.
- If a detail is unclear or not visible, use null.
- Do NOT mention breed unless it is unmistakably obvious.
- Do NOT use subjective or emotional language.
- Do NOT write full sentences except where specified.
- Return ONLY valid JSON. No markdown. No commentary.

Think like a visual inspector, not a storyteller.

JSON schema (must match exactly):

{
  "species_visible": "dog|cat|rabbit|bird|reptile|rodent|other|unknown",
  "size_visible": "very_small|small|medium|large|unknown",
  "coat_or_feather": {
    "primary_color": "...",
    "secondary_color": "...",
    "pattern": "solid|bicolor|tricolor|spotted|striped|patchy|unknown",
    "texture": "short|medium|long|wiry|curly|hairless|unknown"
  },
  "face": {
    "face_shape": "round|oval|long|flat|unknown",
    "snout_length": "short|medium|long|unknown",
    "eye_shape": "round|almond|unknown",
    "eye_color": "...",
    "nose_color": "...",
    "distinctive_markings": ["..."]
  },
  "ears": {
    "position": "upright|floppy|semi-floppy|unknown",
    "size_relative": "small|medium|large|unknown"
  },
  "tail": {
    "length": "short|medium|long|unknown",
    "shape": "straight|curled|unknown"
  },
  "pose": {
    "body_position": "standing|sitting|lying|unknown",
    "head_direction": "forward|left|right|unknown"
  }
}
""".strip()

    def _do():
        resp = client.models.generate_content(
            model=VISION_MODEL,
            contents=[
                vision_prompt,
                types.Part.from_bytes(
                    data=user_image_bytes,
                    mime_type="image/png"
                ),
            ],
        )
        return (resp.text or "").strip()

    sem = get_api_semaphore()
    with sem:
        try:
            return call_with_backoff(_do, max_tries=3, base=1.0)
        except Exception:
            return ""


def build_image_prompt(
    inputs: PetInputs,
    pet_visual_desc: str = "",
    memory_cues: str = "",
    seed: int = 0
) -> str:
    species = _safe_strip(inputs.species)
    visual = pet_visual_desc.strip()
    visual_line = (
        "Use the reference photo as the PRIMARY source of truth.\n"
        "The structured visual traits below are STRICT constraints for identity consistency.\n"
        "Do NOT override the photo or these traits.\n"
        f"Pet appearance reference:\n{visual}\n"
    ) if visual else ""
    species_hint = f'The pet is a "{species}".' if species else "The pet is a household pet."

    background_block = """
Background (must not be plain):
- Add a soft watercolor environment wash, NOT a flat solid color background.
- Include 1–2 simple recognizable elements related to delivery: a small mailbox, a doorstep/porch, or a cozy home interior silhouette.
- Keep it low-detail and pastel so the pet remains the focus.
""".strip()

    memory = (memory_cues or "").strip()
    memory_block = ""
    if memory:
        memory_block = f"""
    Background vignettes (must be visible):
    Place THREE small, separate daily-life vignettes BEHIND the pet.
    They should look like small, simple watercolor vignettes (no frames, no panels).
    Each vignette corresponds to one bullet below.

    Layout rules:
    - Keep the pet centered and in the foreground.
    - Put the three vignettes around the pet (left / right / upper).
    - Vignettes are smaller and slightly faded so the pet remains the focus.
    - Do NOT merge them into one abstract wash; they must be distinguishable as three mini scenes.

    Vignette list:
    {memory}

    Hard rules:
    - Do NOT add readable text anywhere.
    - No watermark, no logo.
    """.strip()

    return f"""
Create a single, cute illustration (not photo-realistic).
{species_hint}
{visual_line}

Scene:
The pet "{inputs.name}" is a mail carrier,
wearing a tiny postman uniform and hat,
carrying a letter in its mouth as if delivering it to the owner.
The uniform is adapted to the animal body (harness-like, cape-like), not human clothing.

Anatomy rules:
- The pet must follow natural anatomy for its species.
- Do NOT add human arms, hands, or humanoid body parts.
- Do NOT add extra limbs beyond what the animal naturally has.
- The pet remains fully animal-like (not humanoid or bipedal).
- The letter is held in the mouth or beak (not hands).

{background_block}

{memory_block}

Mood: warm, wholesome, cozy, friendly, reassuring.
Style:
hand-painted watercolor illustration,
storybook / children's book style,
warm pastel color palette,
soft brush strokes and gentle textures,
very light watercolor wash background with simple environment hints (not a flat solid color),
The three background vignettes should be simpler and lighter than the main pet,
soft outlines, no harsh ink lines.
Lighting: soft natural light, gentle shadows.
Rules: NO readable text, NO watermark, NO logo.
Variation seed: {seed}
""".strip()


def reset_result_state():
    st.session_state.generated_image_bytes = None
    st.session_state.letter_text = None
    st.session_state.ready = False
    st.session_state.image_error = None
    st.session_state.pet_name = None
    st.session_state.last_inputs = None
    st.session_state.last_request_key = None
    st.session_state.user_image_bytes = b""

def build_memory_triptych(inputs: PetInputs, letter_text: str, seed: int = 0) -> str:
    """
    Imagen background에 넣을 '추억 3장면'을 영어로 3줄로 생성.
    - actions 기반 1줄
    - worries 기반 1줄 (어둡지 않게)
    - letter 분위기/핵심 감정 기반 1줄
    """
    actions = _safe_strip(inputs.actions)
    worries = _safe_strip(inputs.worries)
    letter_text = (letter_text or "").strip()

    prompt = f"""
    You create 3 background vignette ideas for a single illustration.
    Return EXACTLY 3 bullet lines in English (each line starts with "- ").

    Goal:
    - Each bullet MUST describe one small, simple daily-life watercolor vignette.
    - The vignettes will appear BEHIND the pet as 3 separate small mini scenes (no frames).
    - Keep them friendly, wholesome, and recognizable with 1~2 concrete objects.

    Rules:
    - No readable text, no quotes, no signage.
    - No scary/dark content.
    - Keep each line short (max ~12 words).
    - Mention 1~2 objects per vignette (e.g., leash, bowl, bed, lamp, toothbrush).

    Inputs:
    - Owner actions: {actions}
    - Worries (keep hopeful): {worries}
    - Letter text (mood only): {letter_text}
    Variation seed: {seed}
    """.strip()


    def _do():
        resp = client.models.generate_content(model=VISION_MODEL, contents=prompt)
        text = (resp.text or "").strip()

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        bullets = [ln for ln in lines if ln.startswith("- ")]
        if len(bullets) >= 3:
            return "\n".join(bullets[:3])
        # fallback: 그냥 앞 3줄을 - 로 붙여서라도 반환
        return "\n".join([f"- {ln.lstrip('- ').strip()}" for ln in lines[:3]])

    sem = get_api_semaphore()
    with sem:
        try:
            return call_with_backoff(_do, max_tries=3, base=1.0)
        except Exception:
            return ""

# =========================================================
# API calls
# =========================================================
def generate_letter_text(prompt: str) -> str:
    def _do():
        resp = client.models.generate_content(model=LETTER_MODEL, contents=prompt)
        return (resp.text or "").strip()

    sem = get_api_semaphore()
    with sem:
        return call_with_backoff(_do)

def generate_image_with_vertex_imagen(
    imagen_prompt: str,
    user_image_bytes: bytes,
) -> Tuple[Optional[bytes], Optional[str]]:
    gen_model, edit_model = get_imagen_models()
    sem = get_api_semaphore()
    image_error = None

    # A) Edit (image-conditioned) if possible
    if user_image_bytes and edit_model is not None:
        try:
            def _do_edit():
                base = VertexImage(image_bytes=user_image_bytes)
                out = edit_model.edit_image(
                    base_image=base,
                    prompt=imagen_prompt,
                    number_of_images=1,
                )
                return out

            with sem:
                out = call_with_backoff(_do_edit, max_tries=3, base=1.0)

            img0 = out.images[0]
            img_bytes = getattr(img0, "_image_bytes", None) or getattr(img0, "image_bytes", None)
            if img_bytes:
                return img_bytes, None
            image_error = "imagen edit returned no image bytes."

        except Exception as e:
            image_error = f"imagen edit failed: {e}"

    # B) Generate (text-to-image)
    try:
        def _do_gen():
            out = gen_model.generate_images(
                prompt=imagen_prompt,
                number_of_images=1,
            )
            return out

        with sem:
            out = call_with_backoff(_do_gen, max_tries=3, base=1.0)

        img0 = out.images[0]
        img_bytes = getattr(img0, "_image_bytes", None) or getattr(img0, "image_bytes", None)
        return img_bytes, image_error

    except Exception as e:
        image_error = (image_error or "") + f"\nimagen generate failed: {e}"
        return None, image_error

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
        species_custom = st.text_input("어떤 반려동물인가요?", placeholder="예: 페럿")

    personality = st.text_input("성격", placeholder="예: 겁 많지만 애교 많음 / 츤데레 / 활발함")
    age = st.text_input("나이", placeholder="예: 3살 / 7개월")
    actions = st.text_area("주인이 자주 해준 행동", placeholder="예: 산책 자주 해줌, 간식 챙겨줌, 안아줌")
    worries = st.text_area("걱정거리(고민)", placeholder="예: 분리불안이 있는 것 같아 걱정돼")
    owner_message = st.text_area("하고 싶은 말(주인이 반려동물에게)", placeholder="예: 요즘 바빠서 미안해. 그래도 사랑해!")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        submitted = st.form_submit_button("✨ 편지 가져오게 하기", width="stretch")
    with col_b:
        cleared = st.form_submit_button("🧹 결과 지우기", width="stretch")

if cleared:
    reset_result_state()

should_generate = submitted or st.session_state.regenerate_requested
if should_generate:
    if not _safe_strip(name):
        st.warning("이름은 꼭 넣어주세요! (나머지는 비워도 괜찮아요!)")
        st.stop()

    user_image_bytes = b""
    if uploaded is not None:
        user_image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        user_image.thumbnail((1024, 1024))
        buf = io.BytesIO()
        user_image.save(buf, format="PNG")
        user_image_bytes = buf.getvalue()
        st.image(user_image, caption="업로드한 사진", width="stretch")
    else:
        st.info("지금은 편지만 제공 중이에요🐾 (이미지 기능은 추후 추가 예정!)")

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

    request_key = make_request_key(inputs, user_image_bytes, seed=st.session_state.generation_seed)
    st.session_state.last_generation_seed = st.session_state.generation_seed

    if (st.session_state.last_request_key == request_key
            and st.session_state.letter_text
            and not st.session_state.regenerate_requested):
        st.info("이미 편지를 가져왔어요! 아래에서 확인해주세요🐾")
        st.stop()

    st.session_state.generated_image_bytes = None
    st.session_state.image_error = None
    st.session_state.ready = False

    st.session_state.user_image_bytes = user_image_bytes
    st.session_state.pet_name = inputs.name
    st.session_state.last_inputs = inputs
    st.session_state.last_request_key = request_key

    letter_prompt = build_letter_prompt(inputs, seed=st.session_state.generation_seed)

    with st.spinner(f"{inputs.name}: 편지를 작성하고 있어요! 잠시만 기다려주세요~ (시간이 조금 걸릴 수 있어요!)"):
        # 0) 기본값(항상 초기화)
        st.session_state.generated_image_bytes = None
        st.session_state.image_error = None
        st.session_state.ready = False
        memory_cues = ""

        # 1) 편지 생성(여기서 실패하면 전체 중단)
        try:
            letter_text = generate_letter_text(letter_prompt)
            st.session_state.letter_text = clamp_text(letter_text, 600)
        except Exception:
            st.warning("...편지 실패...")
            st.stop()

        # 1.5) 메모리 큐(실패해도 계속)
        try:
            memory_cues = build_memory_triptych(
                inputs=st.session_state.last_inputs,
                letter_text=st.session_state.letter_text,
                seed=st.session_state.generation_seed
            )
        except Exception:
            memory_cues = ""

        # 2) 이미지 생성(실패해도 편지는 유지)
        if ENABLE_IMAGE and user_image_bytes:
            try:
                pet_desc = analyze_pet_photo_to_visual_desc(user_image_bytes)
                img_prompt = build_image_prompt(
                    st.session_state.last_inputs,
                    pet_visual_desc=pet_desc,
                    memory_cues=memory_cues,
                    seed=st.session_state.generation_seed
                )
                img_bytes, img_err = generate_image_with_vertex_imagen(
                    imagen_prompt=img_prompt,
                    user_image_bytes=user_image_bytes,
                )

                st.session_state.generated_image_bytes = img_bytes
                st.session_state.image_error = img_err

                if img_bytes is None and not img_err:
                    st.session_state.image_error = "image generation returned no image (unknown reason)"
            except Exception as e:
                st.session_state.generated_image_bytes = None
                st.session_state.image_error = f"auto image generation failed: {e}"

        # 3) 결과 준비 완료(편지라도 있으면 ready)
        st.session_state.ready = True
        st.session_state.regenerate_requested = False

# =========================================================
# Results
# =========================================================
if st.session_state.ready:
    pet_name = st.session_state.pet_name or "반려동물"
    st.subheader("📮 반려동물이 편지를 가져왔어요!")
    col_r1, col_r2 = st.columns([1, 1])
    with col_r1:
        if st.button("🔄 지금과 비슷한 이미지와 편지로 다시 만들고 싶어요.", width="stretch"):
            st.session_state.generation_seed += 1
            st.session_state.regenerate_requested = True
            st.rerun()

    with col_r2:
        if st.button("🎲 느낌이 아예 다른 이미지와 편지를 받아보고 싶어요!", width="stretch"):
            st.session_state.generation_seed += random.randint(5, 30)
            st.session_state.regenerate_requested = True
            st.rerun()

    # 1) 이미지 표시
    if st.session_state.generated_image_bytes:
        st.image(st.session_state.generated_image_bytes, width="stretch")
    else:
        default_bytes = load_default_image_bytes(DEFAULT_IMAGE_PATH)
        if default_bytes:
            st.image(default_bytes, caption="멍멍! 배달부가 편지를 배달하러 왔어요🐾", width="stretch")
        else:
            st.info("기본 이미지 파일이 없어요. images/default_pet_image.png 경로를 확인해주세요!")

    # (선택) 개발용 로그
    if st.session_state.image_error:
        with st.expander("이미지 생성 로그(개발용)"):
            st.code(st.session_state.image_error)

    # 2) 편지 보기
    if st.button("💌 편지받기", width="stretch"):
        st.subheader("편지")
        st.write(st.session_state.letter_text or "")
