import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from duckduckgo_search import DDGS
import json
from dotenv import load_dotenv
import os
import time

# Load biến môi trường từ file .env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
model_name = "google/gemma-3-1b-it"

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

# Tải models với caching và quản lý bộ nhớ
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"Đang sử dụng thiết bị: {device}")

    try:
        # Tải tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=hf_token
        )
        
        # Kiểm tra bộ nhớ GPU khả dụng
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.info(f"GPU Memory: {gpu_memory:.1f} GB")
            
            # Sử dụng các tùy chọn tiết kiệm bộ nhớ
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype="auto",
                device_map="auto",
            )
        else:
            # Trên CPU, sử dụng float32 thông thường
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float32,
                device_map={"": device},
            )
        
        # Đảm bảo tokenizer có pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Lỗi khi tải model: {e}")
        return None, None, None
# Hàm tìm kiếm DuckDuckGo
def search_duckduckgo(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='vn-vn', max_results=max_results))
            return results
    except Exception as e:
        st.error(f"Lỗi khi tìm kiếm: {e}")
        return []

# Hàm tạo bối cảnh từ lịch sử
def generate_conversation_context(history):
    if not history:
        return ""
    
    context = "Lịch sử trò chuyện trước đây:\n"
    for i, (user_msg, bot_msg) in enumerate(history[-3:]):
        context += f"Người dùng: {user_msg}\n"
        context += f"Chuyên gia: {bot_msg}\n"
    
    return context

def generate_search_context(user_input, tokenizer, model, device):
    prompt = f"Từ câu hỏi của người dùng:{user_input}, hãy tạo một truy vấn tìm kiếm thông tin liên quan để tra cứu trên internet. Truy vấn nên ngắn gọn và tập trung vào các từ khóa chính."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=3,
            early_stopping=True,
            temperature=1.0,
            top_k = 64,
            top_p = 0.95,
            min_p = 0.0,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    search_query = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return search_query

# Hàm tạo prompt hiệu quả
def create_efficient_prompt(user_input, conversation_context):
    return f"""Bạn là chuyên gia kỹ thuật về sửa chữa, bảo trì và kiểm tra thiết bị y tế.

{conversation_context}

Câu hỏi: {user_input}

Trả lời ngắn gọn, tập trung vào vấn đề:"""

# Hàm tạo phản hồi với model
def generate_response(user_input, tokenizer, model, history, device):
    # Tìm kiếm thông tin
    # search_query = generate_search_context(user_input, tokenizer, model, device)
    # search_results = search_duckduckgo(search_query)
    
    # Tạo bối cảnh từ lịch sử
    conversation_context = generate_conversation_context(history)
    
    # # Thêm kết quả tìm kiếm vào prompt
    # search_context = ""
    # if search_results:
    #     search_context = "THÔNG TIN TÌM KIẾM ĐƯỢC:\n"
    #     for i, result in enumerate(search_results[:2]):  # Giới hạn số kết quả
    #         search_context += f"{i+1}. {result['title']}: {result['body'][:150]}...\n"
    # else:
    #     search_context = "Không tìm thấy thông tin mới từ internet.\n"
    
    # Tạo prompt hiệu quả
    prompt = create_efficient_prompt(user_input, conversation_context)
    
    # Tokenize và generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # Giảm độ dài phản hồi
            num_beams=3,  # Giảm số beams để tăng tốc
            early_stopping=True,
            temperature=1.0,
            top_k = 64,
            top_p = 0.95,
            min_p = 0.0,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode chỉ phần phản hồi được tạo (bỏ qua prompt)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Đảm bảo phản hồi không rỗng
    if not response.strip():
        response = "Tôi hiểu câu hỏi của bạn. Dựa trên kiến thức chuyên môn, "
        response += "tôi khuyên bạn nên kiểm tra các thông số điện cơ bản và liên hệ với kỹ thuật viên có chuyên môn."
    
    # # Thêm nguồn tham khảo nếu có kết quả tìm kiếm
    # if search_results:
    #     response += "\n\n---\n*Thông tin tham khảo từ các nguồn trực tuyến*"
    
    return response

# Hàm hiển thị response từng phần (streaming)
def stream_response(response):
    placeholder = st.empty()
    full_response = ""
    for chunk in response.split():
        full_response += chunk + " "
        placeholder.markdown(full_response + "▌")
        time.sleep(0.03)  # Giả lập tốc độ gõ
    placeholder.markdown(full_response)
    return full_response

# Giao diện chính
st.title("🏥 Chatbot Tư vấn Kỹ thuật Thiết bị Y tế")
st.write("Chatbot hỗ trợ tư vấn sửa chữa, bảo trì và kiểm tra thiết bị y tế")

# Hiển thị cảnh báo về hiệu suất
if torch.cuda.is_available():
    st.success("Đã phát hiện GPU, sử dụng chế độ tăng tốc")
else:
    st.warning("Không phát hiện GPU, chatbot sẽ chạy trên CPU (có thể chậm hơn)")

# Tải models
with st.spinner("Đang tải model, vui lòng chờ..."):
    medical_tokenizer, medical_model, device = load_models()
models_loaded = medical_model is not None

if not models_loaded:
    st.error("Không thể tải model. Vui lòng kiểm tra kết nối và thử lại.")
    st.stop()

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
    with st.spinner("Đang phân tích và tìm kiếm thông tin..."):
        history_messages = [(user, bot) for user, bot in st.session_state.history]
        response = generate_response(user_input, medical_tokenizer, medical_model, history_messages, device)
    
    # Hiển thị phản hồi với hiệu ứng streaming
    with st.chat_message("assistant"):
        full_response = stream_response(response)
    
    # Lưu vào lịch sử
    st.session_state.history.append((user_input, full_response))
    
    # Tự động lưu lịch sử vào file
    save_chat_history()

# Nút xóa lịch sử hội thoại
if st.button("Xóa lịch sử hội thoại"):
    st.session_state.history = []
    # Xóa file lịch sử
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    st.rerun()

# Hiển thị thông tin kỹ thuật
with st.expander("Thông tin kỹ thuật (cho chuyên gia)"):
    if user_input and models_loaded:
        st.write("**Câu hỏi:**", user_input)
    
    # Hiển thị tổng số tin nhắn trong lịch sử
    st.write(f"**Tổng số tin nhắn trong lịch sử:** {len(st.session_state.history)}")
    
    # Hiển thị thông tin device
    st.write(f"**Thiết bị đang sử dụng:** {device}")
    
    # Nút xuất lịch sử
    if st.button("Xuất lịch sử trò chuyện"):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            st.download_button(
                label="Tải xuống lịch sử",
                data=f,
                file_name="history_backup.json",
                mime="application/json"
            )