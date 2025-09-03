import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from duckduckgo_search import DDGS
import re
import json
import os
from datetime import datetime

# Thiết lập trang
st.set_page_config(page_title="Chatbot Tư vấn Thiết bị Y tế", page_icon="🏥")

# Đường dẫn file lưu lịch sử
HISTORY_FILE = "medical_equipment_chat_history.json"

# Khởi tạo lịch sử hội thoại
if "history" not in st.session_state:
    st.session_state.history = []
    
    # Tải lịch sử từ file nếu tồn tại
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                st.session_state.history = json.load(f)
        except Exception as e:
            st.error(f"Lỗi khi tải lịch sử: {e}")
            st.session_state.history = []

# Hàm lưu lịch sử vào file
def save_chat_history():
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Lỗi khi lưu lịch sử: {e}")

# Tải models với caching
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tải model chuyên về thiết bị y tế
    medical_tokenizer = AutoTokenizer.from_pretrained('./models/mechanic_model')
    medical_model = AutoModelForCausalLM.from_pretrained(
        './models/mechanic_model'
    ).to(device)
    
    return medical_tokenizer, medical_model, device

# Hàm tìm kiếm DuckDuckGo
def search_duckduckgo(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='vn-vn', max_results=max_results))
            return results
    except Exception as e:
        st.error(f"Lỗi khi tìm kiếm: {e}")
        return []

# Hàm xác định xem có cần tìm kiếm không
def needs_search(query):
    # Các từ khóa cho thấy cần tìm kiếm thông tin mới
    search_keywords = [
        'mới nhất', 'hiện nay', 'gần đây', 'cập nhật', 
        'thông tin', 'tìm kiếm', 'tra cứu', 'hướng dẫn mới'
    ]
    
    # Kiểm tra nếu câu hỏi chứa từ khóa tìm kiếm
    return any(keyword in query.lower() for keyword in search_keywords)

# Hàm tạo bối cảnh từ lịch sử
def generate_conversation_context(history):
    if not history:
        return "Chưa có lịch sử trò chuyện trước đó."
    
    context = "Lịch sử trò chuyện trước đây:\n"
    for i, (user_msg, bot_msg) in enumerate(history[-5:]):  # Lấy 5 tin nhắn gần nhất
        context += f"Người dùng: {user_msg}\n"
        context += f"Chuyên gia: {bot_msg}\n"
    
    return context

# Hàm tạo phản hồi với lịch sử hội thoại và tìm kiếm
def generate_response(user_input, tokenizer, model, history, device):
    # Kiểm tra xem có cần tìm kiếm thông tin không
    search_results = []
    if needs_search(user_input):
        with st.spinner("Đang tìm kiếm thông tin mới nhất..."):
            search_query = f"thiết bị y tế {user_input} sửa chữa bảo trì"
            search_results = search_duckduckgo(search_query)
    
    # Tạo bối cảnh từ lịch sử
    conversation_context = generate_conversation_context(history)
    
    # Thêm kết quả tìm kiếm vào prompt nếu có
    search_context = ""
    if search_results:
        search_context = "Thông tin tìm kiếm được từ internet:\n"
        for i, result in enumerate(search_results):
            search_context += f"{i+1}. {result['title']}: {result['body']}\n"
        search_context += "\n"
    
    system_prompt = f"""Bạn là chuyên gia kỹ thuật về sửa chữa, bảo trì và kiểm tra thiết bị y tế. 
Hãy trả lời câu hỏi dựa trên kiến thức chuyên môn về:
- Đo lường và phân tích các thông số điện (điện áp, dòng điện, công suất)
- Chuẩn đoán sự cố thiết bị y tế
- Quy trình bảo trì phòng ngừa
- An toàn điện trong thiết bị y tế

Sử dụng thông tin từ lịch sử trò chuyện trước đây để cung cấp câu trả lời phù hợp và cá nhân hóa.

{conversation_context}

{search_context}

Câu hỏi: {user_input}

Trả lời:"""
    
    inputs = tokenizer(system_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024,
            num_beams=5,
            early_stopping=True,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Loại bỏ phần prompt đã có trong câu trả lời
    response = response.replace(system_prompt, "").strip()
    
    # Thêm nguồn tham khảo nếu có kết quả tìm kiếm
    if search_results:
        response += "\n\n---\n*Thông tin được tham khảo từ các nguồn tìm kiếm trực tuyến*"
    
    return response

# Giao diện chính
st.title("🏥 Chatbot Tư vấn Kỹ thuật Thiết bị Y tế")
st.write("Chatbot hỗ trợ tư vấn sửa chữa, bảo trì và kiểm tra thiết bị y tế dựa trên các thông số kỹ thuật")

# Tải models
medical_tokenizer, medical_model, device = load_models()
models_loaded = True

# Hiển thị lịch sử hội thoại
for user_msg, bot_msg in st.session_state.history:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.write(bot_msg)

# Nhận input từ người dùng
user_input = st.chat_input("Nhập vấn đề về thiết bị y tế...")

if user_input and models_loaded:
    # Hiển thị câu hỏi của người dùng
    with st.chat_message("user"):
        st.write(user_input)

    # Tạo phản hồi bằng model
    with st.spinner("Đang phân tích vấn đề..."):
        history_messages = [(user, bot) for user, bot in st.session_state.history]
        response = generate_response(user_input, medical_tokenizer, medical_model, history_messages, device)
    
    # Hiển thị phản hồi
    with st.chat_message("assistant"):
        st.write(response)
    
    # Lưu vào lịch sử
    st.session_state.history.append((user_input, response))
    
    # Tự động lưu lịch sử vào file
    save_chat_history()
else:
    response = ""

# Nút xóa lịch sử hội thoại
if st.button("Xóa lịch sử hội thoại"):
    st.session_state.history = []
    # Xóa file lịch sử
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    st.rerun()

# Hiển thị thông tin debug
with st.expander("Thông tin kỹ thuật (cho chuyên gia)"):
    if user_input and models_loaded:
        st.write("**Câu hỏi:**", user_input)
        st.write("**Phản hồi:**", response)
    
    # Hiển thị tổng số tin nhắn trong lịch sử
    st.write(f"**Tổng số tin nhắn trong lịch sử:** {len(st.session_state.history)}")
    
    # Nút xuất lịch sử
    if st.button("Xuất lịch sử trò chuyện"):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            st.download_button(
                label="Tải xuống lịch sử",
                data=f,
                file_name="history_backup.json",
                mime="application/json"
            )