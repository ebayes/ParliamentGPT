import streamlit as st

st.set_page_config(layout="centered", page_icon="🏛", page_title="ParliamentGPT")
st.write("# How does ParliamentGPT work?")
st.markdown(
        """
        ### What AI models do you use?

        - 🧠 ParliamentGPT is powered by OpenAI's ChatGPT model.
        
        ### What happens to my data?

        - 🔐 Privacy is baked into everything we do and we comply with the highest data privacy standards, including GDPR. ParliamentGPT cannot access, or store, any information you upload, including FAQs, correspondence you are responding to, pdfs, and images. OpenAI's ChatGPT model also does not collect any user data. You can read more about OpenAI's privacy policy [here](www.).
        - 📝 The only data we store temporarily are any letter templates shared with [parliamentgpt@gmail.com](parliamentgpt@gmail.com) as part of AutoScribe. Their temporary storage is required to enable functionality of AutoScribe, and they are stored using Google-grade data security measures that are compliant with GDPR. These files are deleted at the end of every day, and are not used for any purpose other than to enable functionality of AutoScribe. 

    """
    )

