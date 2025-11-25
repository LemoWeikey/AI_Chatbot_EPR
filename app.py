import streamlit as st
import sys
import os
import time
import asyncio
import base64
from datetime import datetime
from io import BytesIO
import importlib.util

# New features
from gtts import gTTS
from fpdf import FPDF

# Set page config with professional legal theme
st.set_page_config(
    page_title="Trợ Lý Pháp Lý EPR Pro",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🎨 UI & CSS STYLING (Eco-Friendly Design)
# ==========================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    /* Main Background - Eco-Friendly Gradient */
    .stApp {
        background: linear-gradient(135deg, #fefce8 0%, #f0fdf4 50%, #ecfeff 100%);
        background-attachment: fixed;
    }

    /* Typography */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        color: #047857; /* Forest Green */
        font-weight: 700;
    }
    
    p, div, span {
        font-family: 'Inter', sans-serif;
        color: #44403c; /* Warm Stone */
    }

    /* Header Styling - Eco Glassmorphism */
    .main-header {
        background: linear-gradient(135deg, #047857 0%, #10b981 50%, #0ea5e9 100%);
        padding: 2.5rem;
        border-radius: 1.5rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(4, 120, 87, 0.3);
        position: relative;
        overflow: hidden;
        animation: gradientShift 8s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: #d1fae5 !important;
        margin-top: 0.5rem;
        font-size: 1.2rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }

    /* Chat Messages - Glassmorphism */
    .chat-container {
        max-width: 850px;
        margin: 0 auto;
    }

    .user-message {
        background: rgba(220, 252, 231, 0.6); /* Green with transparency */
        backdrop-filter: blur(10px);
        border: 1px solid rgba(187, 247, 208, 0.5);
        border-radius: 1.5rem 1.5rem 0.5rem 1.5rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        box-shadow: 0 8px 32px rgba(4, 120, 87, 0.1);
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease;
    }
    
    .user-message:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(4, 120, 87, 0.15);
    }

    .assistant-message {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(229, 231, 235, 0.5);
        border-left: 4px solid #10b981; /* Emerald Accent */
        border-radius: 0.5rem 1.5rem 1.5rem 1.5rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease;
    }
    
    .assistant-message:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Source Documents - Enhanced Eco Theme */
    .source-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #86efac;
        border-radius: 1rem;
        padding: 1.25rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        box-shadow: 0 4px 16px rgba(4, 120, 87, 0.1);
        transition: all 0.3s ease;
    }
    
    .source-box:hover {
        border-color: #4ade80;
        box-shadow: 0 6px 24px rgba(4, 120, 87, 0.15);
    }
    
    .source-title {
        color: #047857; /* Forest Green */
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1rem;
    }

    /* Sidebar - Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-right: 2px solid rgba(16, 185, 129, 0.2);
    }
    
    .sidebar-header {
        color: #047857;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        font-family: 'Outfit', sans-serif;
    }

    /* Buttons - Eco Theme */
    .stButton>button {
        border-radius: 0.75rem;
        font-weight: 600;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
        border: none;
        background: linear-gradient(135deg, #047857 0%, #10b981 100%);
        color: white;
        box-shadow: 0 4px 16px rgba(4, 120, 87, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 24px rgba(4, 120, 87, 0.3);
        background: linear-gradient(135deg, #065f46 0%, #059669 100%);
    }
    
    .stButton>button:active {
        transform: translateY(0) scale(0.98);
    }

    /* Download Button */
    .stDownloadButton>button {
        border-radius: 0.75rem;
        font-weight: 600;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.2);
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.3);
    }

    /* Badges - Nature Inspired */
    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .badge:hover {
        transform: scale(1.1);
        animation: none;
    }
    
    .badge-blue { 
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
        box-shadow: 0 2px 8px rgba(30, 64, 175, 0.2);
    }
    
    .badge-green { 
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        color: #047857;
        box-shadow: 0 2px 8px rgba(4, 120, 87, 0.2);
    }
    
    .badge-red { 
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        box-shadow: 0 2px 8px rgba(153, 27, 27, 0.2);
    }

    /* Input Fields */
    .stTextInput>div>div>input,
    .stChatInput>div>div>input {
        border-radius: 1rem;
        border: 2px solid #d1fae5;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextInput>div>div>input:focus,
    .stChatInput>div>div>input:focus {
        border-color: #10b981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
    }

    /* Expander - Enhanced */
    .streamlit-expanderHeader {
        background: rgba(240, 253, 244, 0.6);
        border-radius: 0.75rem;
        font-weight: 600;
        color: #047857;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(220, 252, 231, 0.8);
    }

    /* Toggle Switch */
    .stCheckbox {
        font-family: 'Inter', sans-serif;
    }

    /* Spinner */
    .stSpinner>div {
        border-top-color: #10b981 !important;
    }

    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
        border-radius: 0.75rem;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
        border-radius: 0.75rem;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 0.75rem;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f0fdf4;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }

</style>
""", unsafe_allow_html=True)

# ==========================================
# 🛠️ UTILITY FUNCTIONS
# ==========================================

@st.cache_resource
def load_chatbot_core():
    """Load and initialize the chatbot core module"""
    try:
        spec = importlib.util.spec_from_file_location(
            "epr_chatbot_core",
            os.path.join(os.path.dirname(__file__), "epr_chatbot_core.py")
        )
        core_module = importlib.util.module_from_spec(spec)
        sys.modules["epr_chatbot_core"] = core_module
        
        # Redirect stdout to capture init logs
        import io
        from contextlib import redirect_stdout, redirect_stderr
        output_buffer = io.StringIO()
        
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            spec.loader.exec_module(core_module)
            
        return core_module
    except Exception as e:
        st.error(f"❌ Core System Error: {e}")
        return None

def text_to_speech(text):
    """Convert text to speech using gTTS"""
    try:
        tts = gTTS(text=text, lang='vi')
        fp = BytesIO()
        tts.write_to_fp(fp)
        return fp
    except Exception as e:
        st.warning(f"TTS Error: {e}")
        return None

def create_pdf(chat_history):
    """Generate PDF from chat history"""
    pdf = FPDF()
    pdf.add_page()
    
    # Add Unicode font - use Helvetica as default (works everywhere)
    font_family = "Helvetica"
    
    # Header
    pdf.set_font(font_family, '', 16)
    pdf.cell(200, 10, text="EPR Legal Assistant - Chat Export", align='C')
    pdf.ln(10)
    
    pdf.set_font(font_family, '', 12)
    
    for msg in chat_history:
        role = "Nguoi dung" if msg["role"] == "user" else "Tro ly"
        content = msg["content"]
        
        # Clean up content - remove emojis and special characters
        content = content.replace("🌱", "").replace("✅", "").replace("⚠️", "")
        content = content.replace("🌿", "").replace("🌍", "").replace("♻️", "")
        content = content.replace("📜", "").replace("🏭", "").replace("👤", "")
        content = content.replace("📚", "").replace("📄", "").replace("🎤", "")
        content = content.replace("🗣️", "").replace("▌", "")
        
        # Safety fallback for Helvetica (standard font doesn't support Unicode well)
        try:
            content = content.encode('latin-1', 'replace').decode('latin-1')
        except:
            content = content.encode('ascii', 'ignore').decode('ascii')
        
        # Role
        pdf.set_font(font_family, 'B', 12)
        pdf.set_text_color(4, 120, 87) # Forest Green
        pdf.cell(0, 10, text=f"{role}:")
        pdf.ln()
        
        # Content
        pdf.set_font(font_family, '', 11)
        pdf.set_text_color(0, 0, 0) # Black
        pdf.multi_cell(0, 8, text=content)
            
        pdf.ln(5)
        
    return bytes(pdf.output())

# ==========================================
# 🚀 APP INITIALIZATION
# ==========================================

# Load Chatbot Core
if "chatbot_core" not in st.session_state:
    with st.spinner("Initializing Legal Knowledge Base..."):
        st.session_state.chatbot_core = load_chatbot_core()

chatbot_core = st.session_state.chatbot_core

if not chatbot_core:
    st.stop()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = False

# ==========================================
# 📱 SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Cài Đặt</div>', unsafe_allow_html=True)
    
    # Voice Toggle
    st.session_state.voice_enabled = st.toggle("Bật Phản Hồi Giọng Nói 🗣️", value=st.session_state.voice_enabled)
    
    st.markdown("---")
    
    # Export Chat
    st.markdown('<div class="sidebar-header">📂 Thao Tác</div>', unsafe_allow_html=True)
    if st.button("🗑️ Xóa Cuộc Trò Chuyện", use_container_width=True):
        st.session_state.messages = []
        chatbot_core.clear_memory()
        st.rerun()
        
    if len(st.session_state.messages) > 0:
        if st.download_button(
            label="📄 Xuất File PDF",
            data=create_pdf(st.session_state.messages),
            file_name=f"EPR_TroChuyenPhapLy_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        ):
            st.success("Đã sẵn sàng xuất file!")

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.85rem; color: #78716c; padding: 1rem; background: rgba(240, 253, 244, 0.5); border-radius: 0.75rem; margin-top: 1rem;">
        <strong style="color: #047857;">🌿 Trợ Lý Pháp Lý EPR Pro</strong><br>
        Phiên bản 2.0.0 | Thân Thiện Môi Trường
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 🏠 MAIN INTERFACE
# ==========================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🌿 Trợ Lý Pháp Lý EPR Pro</h1>
    <p>🌍 Tư Vấn Luật Môi Trường Bằng AI Cho Tương Lai Bền Vững</p>
</div>
""", unsafe_allow_html=True)

# Welcome Message
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem; color: #44403c;">
        <h3 style="color: #047857; font-size: 2rem; margin-bottom: 1rem;">🌱 Chào Mừng Đến Với Trợ Lý Pháp Lý EPR!</h3>
        <p style="font-size: 1.1rem; color: #78716c; max-width: 700px; margin: 0 auto 2rem;">Người đồng hành AI đáng tin cậy của bạn trong việc tìm hiểu các quy định về Trách nhiệm Mở rộng của Nhà sản xuất (EPR) tại Việt Nam. Cùng nhau, chúng ta xây dựng một tương lai bền vững hơn. ♻️</p>
        <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin-top: 2rem;">
            <span class="badge badge-green">🌿 Luật Môi Trường</span>
            <span class="badge badge-blue">📜 Tra Cứu Điều Luật</span>
            <span class="badge badge-green">♻️ Hướng Dẫn Tái Chế</span>
            <span class="badge badge-blue">🏭 Hỗ Trợ Tuân Thủ</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Chat History Display
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 Bạn</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Assistant Message with Metadata
        metadata = message.get("metadata", {})
        
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🌿 Trợ Lý</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Display Source Documents if available
        if metadata.get("documents"):
            with st.expander("📚 Tài Liệu Pháp Lý Tham Khảo"):
                for i, doc in enumerate(metadata["documents"], 1):
                    doc_meta = doc.get("metadata", {})
                    
                    # Build the title with Chương, Mục, and Điều
                    title_parts = []
                    
                    # Add Chương (Chapter) if available
                    if doc_meta.get('Chuong'):
                        chuong = doc_meta.get('Chuong')
                        chuong_name = doc_meta.get('Chuong_Name', '')
                        if chuong_name:
                            title_parts.append(f"📖 {chuong}: {chuong_name}")
                        else:
                            title_parts.append(f"📖 {chuong}")
                    
                    # Add Mục (Section) if available
                    if doc_meta.get('Muc'):
                        muc = doc_meta.get('Muc')
                        muc_name = doc_meta.get('Muc_Name', '')
                        if muc_name:
                            title_parts.append(f"📑 {muc}: {muc_name}")
                        else:
                            title_parts.append(f"📑 {muc}")
                    
                    # Add Điều (Article) - check if it already contains "Điều" prefix
                    dieu = doc_meta.get('Dieu', 'N/A')
                    dieu_name = doc_meta.get('Dieu_Name', 'Không rõ')
                    
                    # If Dieu already starts with "Điều", don't add it again
                    if str(dieu).startswith('Điều'):
                        if dieu_name and dieu_name != 'Không rõ':
                            title_parts.append(f"📄 {dieu}: {dieu_name}")
                        else:
                            title_parts.append(f"📄 {dieu}")
                    else:
                        if dieu_name and dieu_name != 'Không rõ':
                            title_parts.append(f"📄 Điều {dieu}: {dieu_name}")
                        else:
                            title_parts.append(f"📄 Điều {dieu}")
                    
                    # Join all parts with line breaks
                    title_html = "<br>".join(title_parts)
                    
                    st.markdown(f"""
                    <div class="source-box">
                        <div class="source-title">
                            {title_html}
                        </div>
                        <div style="margin-top: 0.5rem; color: #44403c;">
                            {doc.get('page_content', '')[:300]}...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Audio Player if Voice Enabled
        if st.session_state.voice_enabled and message.get("audio"):
            st.audio(message["audio"], format="audio/mp3")

# ==========================================
# 💬 INPUT AREA
# ==========================================

from openai import OpenAI

# Initialize OpenAI Client
try:
    client = OpenAI()
except Exception as e:
    client = None
    print(f"OpenAI Client Init Error: {e}")

# Audio Input (New Feature)
audio_value = st.audio_input("🎤 Nhập Bằng Giọng Nói")
user_input = st.chat_input("Nhập câu hỏi pháp lý của bạn tại đây...")

# Handle Input
final_input = None

if user_input:
    final_input = user_input
elif audio_value:
    if not client:
        st.error("⚠️ Chưa khởi tạo OpenAI Client. Vui lòng kiểm tra API Key của bạn.")
    else:
        # Check if we already processed this audio to prevent infinite loop
        audio_bytes = audio_value.getvalue()
        if "last_audio_bytes" not in st.session_state or st.session_state.last_audio_bytes != audio_bytes:
            with st.spinner("🎤 Đang xử lý giọng nói..."):
                try:
                    # Create a file-like object
                    audio_file = BytesIO(audio_bytes)
                    audio_file.name = "voice_input.wav" 
                    
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=audio_file,
                        language="vi"
                    )
                    text = transcript.text.strip()
                    
                    # Filter known Whisper hallucinations (silence/noise interpretations)
                    IGNORED_PHRASES = [
                        "Cảm ơn các bạn đã theo dõi.",
                        "Cảm ơn các bạn đã xem video.",
                        "Xin chào và hẹn gặp lại.",
                        "Chúc các bạn thành công.",
                        "MBC",
                        "Subtitles by",
                        "Amara.org"
                    ]
                    
                    is_hallucination = any(phrase in text for phrase in IGNORED_PHRASES)
                    
                    if not text or is_hallucination or len(text) < 2:
                        st.warning("⚠️ Không nghe rõ. Vui lòng nói lại.")
                    else:
                        final_input = text
                        st.success(f"🗣️ Đã nghe: {final_input}")
                        time.sleep(1) # Let user see the text
                        
                    st.session_state.last_audio_bytes = audio_bytes # Mark as processed
                    
                except Exception as e:
                    st.error(f"Lỗi giọng nói: {e}")

if final_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": final_input})
    
    # Rerun to show user message immediately
    st.rerun()

# Processing Logic (Triggered by rerun if last message is user)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_msg = st.session_state.messages[-1]["content"]
    
    # Get Chat History
    chat_history = chatbot_core.get_full_chat_history(max_exchanges=3)
    
    # Placeholder for streaming
    with st.container():
        response_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # State container for async updates
        chat_state = {
            "full_response": "",
            "documents_used": [],
            "source_type": None
        }
        
        # Async Loop
        async def run_chat():
            # Show "Thinking" status
            status_placeholder.markdown("`🌿 Đang phân tích tài liệu pháp lý...`")
            
            async for update in chatbot_core.optimized_chatbot_pipeline(last_user_msg, chat_history):
                update_type = update.get('type')
                
                if update_type == 'status':
                    status_placeholder.markdown(f"`{update.get('message')}`")
                    
                elif update_type == 'response_chunk':
                    chunk = update.get('chunk', '')
                    chat_state["full_response"] += chunk
                    response_placeholder.markdown(f"""
                    <div class="assistant-message">
                        <strong>🌿 Trợ Lý</strong><br>
                        {chat_state["full_response"]}▌
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif update_type == 'response_complete':
                    chat_state["full_response"] = update.get('text', chat_state["full_response"])
                    chat_state["documents_used"] = update.get('documents', [])
                    chat_state["source_type"] = update.get('source')
            
            status_placeholder.empty()
            response_placeholder.markdown(f"""
            <div class="assistant-message">
                <strong>🌿 Trợ Lý</strong><br>
                {chat_state["full_response"]}
            </div>
            """, unsafe_allow_html=True)
            
        # Run Async
        asyncio.run(run_chat())
        
        # Extract results
        full_response = chat_state["full_response"]
        documents_used = chat_state["documents_used"]
        source_type = chat_state["source_type"]
        
        # Prepare Metadata
        metadata = {
            "documents": [
                {
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                    "page_content": doc.page_content if hasattr(doc, 'page_content') else str(doc)
                }
                for doc in documents_used
            ],
            "source": source_type
        }
        
        # Generate Audio if enabled
        audio_data = None
        if st.session_state.voice_enabled:
            with st.spinner("Đang tạo phản hồi âm thanh..."):
                audio_fp = text_to_speech(full_response[:500]) # Limit length for speed
                if audio_fp:
                    audio_data = audio_fp.getvalue()
        
        # Save to Session State
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "metadata": metadata,
            "audio": audio_data
        })
        
        # Save to Memory
        try:
            chatbot_core.conversation_memory.save_context(
                {"input": last_user_msg},
                {"generation": full_response}
            )
        except Exception as e:
            pass
            
        st.rerun()
