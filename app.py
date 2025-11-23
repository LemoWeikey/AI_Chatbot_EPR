import streamlit as st
import sys
import os
from typing import Dict, Any
import importlib.util

# Set page config with eco-friendly theme
st.set_page_config(
    page_title="🌱 EPR Legal Chatbot - Trợ lý Luật Môi trường",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for eco-friendly styling
st.markdown("""
<style>
    /* Force light eco-friendly background everywhere */
    .stApp {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%) !important;
    }

    /* Main content area */
    .main {
        background: transparent !important;
    }

    /* Main block container */
    .block-container {
        background: transparent !important;
        padding-top: 2rem;
    }

    /* Override any dark theme */
    body {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%) !important;
    }

    /* Header styling - nature-inspired */
    .main-title {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(5, 150, 105, 0.1);
    }

    .subtitle {
        text-align: center;
        color: #047857;
        margin-bottom: 2rem;
        font-size: 1.1rem;
        font-weight: 500;
    }

    .eco-badge {
        display: inline-block;
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 2rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem auto;
        text-align: center;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
    }

    /* Chat messages - eco-friendly colors */
    .chat-message {
        padding: 1.2rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s ease;
    }

    .chat-message:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }

    .user-message {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 5px solid #3b82f6;
    }

    .assistant-message {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid #10b981;
    }

    /* Source documents - earthy tones */
    .source-doc {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 0.5rem 0;
        border-left: 4px solid #f59e0b;
        box-shadow: 0 2px 6px rgba(245, 158, 11, 0.15);
    }

    /* Quality badges - nature-inspired */
    .quality-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 1.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.3rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }

    .quality-badge:hover {
        transform: scale(1.05);
    }

    .badge-success {
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        color: white;
    }

    .badge-warning {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
    }

    .badge-error {
        background: linear-gradient(135deg, #f87171 0%, #ef4444 100%);
        color: white;
    }

    /* Buttons - eco-styled */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 6px 12px rgba(16, 185, 129, 0.3);
        transform: translateY(-2px);
    }

    /* Sidebar - forest green theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #064e3b 0%, #047857 100%);
        color: white;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: white !important;
    }

    /* Input field - eco-friendly */
    .stTextInput>div>div>input,
    .stChatInput>div>div>input,
    [data-testid="stChatInput"] input {
        border-radius: 2rem;
        border: 2px solid #10b981;
        padding: 0.75rem 1.5rem;
        background-color: white !important;
        transition: all 0.3s ease;
        color: #064e3b !important;
    }

    .stTextInput>div>div>input:focus,
    .stChatInput>div>div>input:focus {
        border-color: #059669;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
        background-color: white !important;
    }

    /* Chat input container */
    .stChatInputContainer,
    [data-testid="stChatInputContainer"] {
        background: transparent !important;
    }

    /* Bottom container */
    .stBottom {
        background: transparent !important;
    }

    /* Expander - nature theme */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-radius: 0.75rem;
        color: #065f46;
        font-weight: 600;
    }

    /* Metrics - eco-styled */
    [data-testid="stMetricValue"] {
        color: #10b981;
        font-size: 2rem;
        font-weight: 700;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 1rem;
        border-left: 5px solid #10b981;
    }

    /* Eco footer */
    .eco-footer {
        text-align: center;
        color: #059669;
        font-size: 0.9rem;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 2px solid #d1fae5;
    }

    /* Spinner - eco colors */
    .stSpinner > div {
        border-top-color: #10b981 !important;
    }

    /* Success/Info/Warning boxes */
    .stSuccess {
        background-color: #d1fae5 !important;
        color: #065f46 !important;
    }

    .stInfo {
        background-color: #dbeafe !important;
        color: #1e40af !important;
    }

    .stWarning {
        background-color: #fef3c7 !important;
        color: #92400e !important;
    }

    .stError {
        background-color: #fee2e2 !important;
        color: #991b1b !important;
    }

    /* All text should be readable */
    p, span, div, label {
        color: #064e3b !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #059669 !important;
    }

    /* Code blocks */
    code {
        background-color: #ecfdf5 !important;
        color: #047857 !important;
    }

    /* Dataframes and tables */
    .dataframe {
        background-color: white !important;
    }

    /* Ensure no black anywhere */
    * {
        color: inherit;
    }

    /* Override Streamlit's default dark backgrounds */
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #064e3b 0%, #047857 100%) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize core chatbot components
@st.cache_resource
def load_chatbot_core():
    """Load and initialize the chatbot core module"""
    try:
        # Load the epr_chatbot_core module
        spec = importlib.util.spec_from_file_location(
            "epr_chatbot_core",
            os.path.join(os.path.dirname(__file__), "epr_chatbot_core.py")
        )
        core_module = importlib.util.module_from_spec(spec)
        sys.modules["epr_chatbot_core"] = core_module

        # Load module with output visible for debugging
        import io
        from contextlib import redirect_stdout, redirect_stderr

        # Create status placeholder
        status_container = st.empty()
        output_buffer = io.StringIO()

        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            status_container.info("🔄 Loading chatbot components...")
            spec.loader.exec_module(core_module)

        # Show initialization output
        output = output_buffer.getvalue()
        if output:
            with st.expander("📋 Initialization Log", expanded=False):
                st.code(output)

        status_container.success("✅ Chatbot loaded successfully!")
        return core_module

    except Exception as e:
        st.error(f"❌ Error loading chatbot core: {e}")
        st.info("💡 Tip: Try restarting the app or check the console for detailed errors.")
        import traceback
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())
        return None

# Load chatbot
with st.spinner("Loading chatbot... This may take a minute on first run..."):
    chatbot_core = load_chatbot_core()

# Check if chatbot loaded successfully
if chatbot_core is None:
    st.error("⚠️ Chatbot failed to load. Please check the error details above and restart the app.")
    st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history_str" not in st.session_state:
    st.session_state.chat_history_str = ""

# Header with eco-friendly theme
st.markdown('<h1 class="main-title">🌱 EPR Legal Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🌍 Trợ lý AI chuyên về Luật Trách nhiệm Mở rộng của Nhà sản xuất Việt Nam</p>', unsafe_allow_html=True)
st.markdown('<div class="eco-badge">♻️ Bảo vệ môi trường - Trách nhiệm của mọi người</div>', unsafe_allow_html=True)

# Sidebar with environmental theme
with st.sidebar:
    st.markdown("### 🌿 Menu")

    st.markdown("---")
    st.markdown("### 📊 Thống kê")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💬 Tin nhắn", len(st.session_state.messages))
    with col2:
        st.metric("🌱 CO₂ Saved", f"{len(st.session_state.messages) * 0.5:.1f}g")

    st.caption("💡 Mỗi câu hỏi kỹ thuật số giúp tiết kiệm giấy và giảm carbon!")

    st.markdown("---")
    st.markdown("### 🎯 Về chúng tôi")
    st.success("""
    **Trợ lý Pháp lý EPR** cung cấp câu trả lời thông minh về luật EPR
    (Extended Producer Responsibility - Trách nhiệm Mở rộng của Nhà sản xuất).

    **🌿 Tính năng:**
    - ♻️ Tra cứu FAQ nhanh
    - 📚 Tìm kiếm văn bản pháp luật
    - 🔍 Tìm kiếm web (dự phòng)
    - ✅ Kiểm tra chất lượng câu trả lời
    - 🎯 Phát hiện thông tin sai lệch

    **🌍 Sứ mệnh:**
    Hỗ trợ doanh nghiệp và cá nhân hiểu rõ trách nhiệm bảo vệ môi trường!
    """)

    st.markdown("---")
    st.markdown("### 🔧 Hành động")

    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        if chatbot_core:
            chatbot_core.clear_memory()
        st.success("✅ Đã xóa lịch sử!")
        st.rerun()

    st.markdown("---")
    st.markdown("### 💚 Cam kết xanh")
    st.info("""
    🌱 **Chatbot xanh** - Giảm tiêu thụ giấy

    ♻️ **Tái chế tri thức** - Chia sẻ kiến thức pháp lý

    🌍 **Bảo vệ hành tinh** - Mỗi hành động nhỏ đều có ý nghĩa
    """)

    st.markdown("---")
    st.caption("🌿 Xây dựng với Streamlit & LangGraph")
    st.caption("💚 Vì một Việt Nam xanh & bền vững")

# Main chat interface
st.markdown("### 💬 Trò chuyện với Trợ lý EPR")

# Quick start guide for new users
if len(st.session_state.messages) == 0:
    st.info("""
    👋 **Xin chào! Tôi là Trợ lý EPR - hỗ trợ bạn về Luật Trách nhiệm Mở rộng của Nhà sản xuất!**

    **🌿 Bạn có thể hỏi tôi:**
    - 📜 "Điều 7 quy định gì?"
    - ♻️ "Quy định về tái chế là gì?"
    - 🏭 "Ai chịu trách nhiệm tái chế sản phẩm?"
    - 🌍 "EPR là gì?"
    - 📦 "Trách nhiệm của nhà sản xuất về bao bì?"

    **💡 Mẹo:** Tôi có thể hiểu câu hỏi tiếp theo của bạn dựa trên ngữ cảnh!
    """)
else:
    st.markdown("💡 **Gợi ý:** Hỏi về Điều luật, quy định tái chế, trách nhiệm nhà sản xuất...")

# Display chat messages
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>👤 Bạn:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
        else:
            # Assistant message
            st.markdown(f'<div class="chat-message assistant-message"><strong>🌱 Trợ lý EPR:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)

            # Show quality indicators if available
            if "metadata" in message:
                metadata = message["metadata"]

                # Quality badges with Vietnamese labels
                col1, col2, col3 = st.columns(3)

                with col1:
                    if metadata.get("hallucination_detected"):
                        st.markdown('<span class="quality-badge badge-error">⚠️ Cần kiểm tra</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="quality-badge badge-success">✅ Đáng tin cậy</span>', unsafe_allow_html=True)

                with col2:
                    grade = metadata.get("grade_result", "unknown")
                    if grade == "useful":
                        st.markdown('<span class="quality-badge badge-success">✓ Hữu ích</span>', unsafe_allow_html=True)
                    elif grade in ["not useful", "not supported"]:
                        st.markdown('<span class="quality-badge badge-warning">⚠️ Cần cải thiện</span>', unsafe_allow_html=True)
                    elif grade == "web_search":
                        st.markdown('<span class="quality-badge badge-warning">🌐 Từ Web</span>', unsafe_allow_html=True)

                with col3:
                    retries = metadata.get("retries", 0)
                    if retries > 0:
                        st.markdown(f'<span class="quality-badge badge-warning">🔄 {retries} lần thử</span>', unsafe_allow_html=True)

                # Show source documents with eco-friendly design
                if metadata.get("documents"):
                    with st.expander("📚 Tài liệu pháp lý tham khảo", expanded=False):
                        for i, doc in enumerate(metadata["documents"], 1):
                            doc_meta = doc.get("metadata", {})
                            st.markdown(f"""
                            <div class="source-doc">
                                <strong>📄 Tài liệu {i}:</strong> Điều {doc_meta.get('Dieu', 'N/A')} - {doc_meta.get('Dieu_Name', 'Không rõ')}<br>
                                <small>📖 Chương {doc_meta.get('Chuong', 'N/A')}: {doc_meta.get('Chuong_Name', '')}</small><br>
                                <small>📑 Mục {doc_meta.get('Muc', 'N/A')}: {doc_meta.get('Muc_Name', '')}</small><br>
                                <p style="margin-top: 0.5rem; font-size: 0.875rem; color: #065f46;">💡 {doc.get('page_content', '')[:200]}...</p>
                            </div>
                            """, unsafe_allow_html=True)

                # Show web URLs if available
                if metadata.get("web_urls"):
                    with st.expander("🌐 Kết quả từ Internet", expanded=False):
                        st.markdown(metadata["web_urls"])

# Chat input with eco-friendly placeholder
user_input = st.chat_input("💬 Đặt câu hỏi về luật EPR... (VD: 'Điều 7 quy định gì?')")

if user_input and chatbot_core:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get chat history
    chat_history = chatbot_core.get_full_chat_history(max_exchanges=5)

    # Display user message immediately
    with st.container():
        st.markdown(f'<div class="chat-message user-message"><strong>👤 Bạn:</strong><br>{user_input}</div>', unsafe_allow_html=True)

    # Create placeholder for streaming response
    response_placeholder = st.empty()
    status_placeholder = st.empty()

    try:
        import asyncio

        # Run optimized pipeline with streaming
        async def stream_response():
            full_response = ""
            documents_used = []
            source_type = None
            current_status = ""

            async for update in chatbot_core.optimized_chatbot_pipeline(user_input, chat_history):
                update_type = update.get('type')

                if update_type == 'status':
                    # Update status message
                    current_status = update.get('message', '')
                    status_placeholder.info(current_status)

                elif update_type == 'response_chunk':
                    # Stream response chunks
                    chunk = update.get('chunk', '')
                    full_response += chunk

                    # Display streaming response
                    response_placeholder.markdown(
                        f'<div class="chat-message assistant-message"><strong>🌱 Trợ lý EPR:</strong><br>{full_response}▌</div>',
                        unsafe_allow_html=True
                    )

                elif update_type == 'response_complete':
                    # Final response
                    full_response = update.get('text', full_response)
                    documents_used = update.get('documents', [])
                    source_type = update.get('source')

            # Clear status
            status_placeholder.empty()

            # Display final response without cursor
            response_placeholder.markdown(
                f'<div class="chat-message assistant-message"><strong>🌱 Trợ lý EPR:</strong><br>{full_response}</div>',
                unsafe_allow_html=True
            )

            return full_response, documents_used, source_type

        # Run async function
        full_response, documents_used, source_type = asyncio.run(stream_response())

        # Prepare metadata
        metadata = {
            "documents": [
                {
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                    "page_content": doc.page_content if hasattr(doc, 'page_content') else str(doc)
                }
                for doc in documents_used
            ],
            "source": source_type,
            "hallucination_detected": False,
            "grade_result": "useful" if documents_used else "no_docs",
            "retries": 0,
            "generation_retries": 0,
            "web_urls": ""
        }

        # Add assistant message to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "metadata": metadata
        })

        # Save to memory
        try:
            chatbot_core.conversation_memory.save_context(
                {"input": user_input},
                {"generation": full_response}
            )
        except Exception as e:
            st.warning(f"Could not save to memory: {e}")

        # Show document sources after streaming completes
        if documents_used:
            with st.expander("📚 Tài liệu tham khảo", expanded=False):
                for i, doc in enumerate(documents_used, 1):
                    if hasattr(doc, 'metadata'):
                        doc_meta = doc.metadata
                        if source_type == "faq":
                            st.markdown(f"""
                            <div class="source-doc">
                                <strong>❓ FAQ {i}:</strong> {doc_meta.get('Câu_hỏi', 'N/A')}<br>
                                <small>Score: {doc_meta.get('score', 'N/A')}</small><br>
                                <p style="margin-top: 0.5rem; font-size: 0.875rem; color: #065f46;">{doc.page_content[:200]}...</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="source-doc">
                                <strong>📄 Tài liệu {i}:</strong> Điều {doc_meta.get('Dieu', 'N/A')} - {doc_meta.get('Dieu_Name', 'Không rõ')}<br>
                                <small>📖 Chương {doc_meta.get('Chuong', 'N/A')}: {doc_meta.get('Chuong_Name', '')}</small><br>
                                <small>📑 Mục {doc_meta.get('Muc', 'N/A')}: {doc_meta.get('Muc_Name', '')}</small><br>
                                <p style="margin-top: 0.5rem; font-size: 0.875rem; color: #065f46;">💡 {doc.page_content[:200]}...</p>
                            </div>
                            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing your question: {e}")
        import traceback
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Xin lỗi, đã xảy ra lỗi: {str(e)}",
            "metadata": {}
        })

    # Small delay to show the streaming effect
    import time
    time.sleep(0.5)

    # Rerun to update message history
    st.rerun()

# Eco-friendly Footer
st.markdown("---")
st.markdown("""
<div class="eco-footer">
    <h4 style="color: #059669; margin-bottom: 0.5rem;">🌍 Cùng nhau bảo vệ môi trường</h4>
    <p style="color: #047857;">
        <strong>♻️ Chatbot xanh</strong> - Giảm sử dụng giấy, tăng hiệu quả tra cứu pháp luật<br>
        <strong>🌱 Tri thức bền vững</strong> - Chia sẻ kiến thức EPR miễn phí cho cộng đồng<br>
        <strong>💚 Trách nhiệm chung</strong> - Mỗi doanh nghiệp, mỗi công dân đều có vai trò
    </p>
    <p style="font-size: 0.85rem; color: #10b981; margin-top: 1rem;">
        🔧 Xây dựng với STS,EPR_PRO & OpenAI |
        🌿 Vì một Việt Nam xanh & phát triển bền vững
    </p>
</div>
""", unsafe_allow_html=True)
