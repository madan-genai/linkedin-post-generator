import streamlit as st
from few_shots import FewShotPosts
from post_generator import generate_post
def main():
    st.title("LinkedIn Post Generator")
    col1, col2, col3 = st.columns(3)
    fs = FewShotPosts()
    with col1:
        selected_tag = st.selectbox("Title",options=fs.get_tags())
    with col2:
        selected_length = st.selectbox("Length",options=["Short","Medium","Long"])
    with col3:
        selected_language = st.selectbox("Language",options=["Urdu","English"])
    
     # Generate Button
    if st.button("Generate"):
        post = generate_post(selected_length, selected_language, selected_tag)
        st.write(post)

    
# Run the app
if __name__ == "__main__":
    main()