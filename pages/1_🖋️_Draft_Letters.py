import streamlit as st
import openai
from datetime import date

openai.api_key = "" # insert api key

st.set_page_config(layout="wide", page_icon="🏛", page_title="ParliamentGPT")
st.title("🖋️ Letter writer")
st.write(
    "This app helps governments and non-profits draft letters"
)

def generate_article(faqs, letter, tone, word_count):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a UK Member of Parliament writing a letter to your constituent."},
            {"role": "user", "content":"Take into consideration the following FAQs:" + faqs},
            {"role": "user", "content":"Write a letter in response to a constituent letter that reads" + letter},
            {"role": "user", "content": "This article should be written in the following style:" + ', '.join(tone)},
            {"role": "user", "content": "The article length should be" + str(word_count)},
        ]
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content
    print(result)
    return result

def clear_text():
    st.session_state["output"] = ""

def print_output():
    # st.session_state.output = "Writing article..."
    article = generate_article(faqs, letter, tone, word_count)
    st.session_state.output = ""
    st.session_state.output = article

col1, col2 = st.columns([1, 2])

with col1:
    faqs = st.text_area(label="Paste FAQs:", height=100)
    letter = st.text_area(label="Paste letter:", height=100)
    tone = st.multiselect('Tone', ['Formal', 'Sympathetic', 'Informal'], ['Formal'])
    word_count = st.slider("Word count", min_value = 250, max_value = 400, value = 300)

with col2:
    if "output_placeholder" not in st.session_state:
        st.session_state.output_placeholder = ""
    output_placeholder = st.empty()
    output_placeholder.text(st.session_state.output_placeholder)

    if "output" not in st.session_state:
        st.session_state.output = ""
    output = st.text_area(label="Output:", height=390, key="output")
        
    col1, col2, col3, col4 = st.columns(4)
    with col3:
        clear_button = st.button("Clear", on_click=clear_text)
    with col4:
        submit_button = st.button("Generate", on_click=print_output)