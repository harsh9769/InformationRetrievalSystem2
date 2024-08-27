import streamlit as st

from src.helper import get_text,get_chunks,get_vector_store,get_conversational_chain

def user_input(user_question):
    response=st.session_state.conversation({'question':user_question})
    st.session_state.chatHistory=response['chat_history']
    for i ,message in enumerate(st.session_state.chatHistory):
        if i%2==0:
            st.write("User: ",message.content)
        else:
            st.write("Reply: ",message.content)


def main():

    st.set_page_config("InformationRetrieval")
    st.header("Information-Retreival-System")

    user_question=st.text_input("Ask the question related to pdf")

    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory=None

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")

        Url=st.text_input("Enter your Urls")

        if st.button("Submit & Process"):
            with st.spinner("Processing...."):

                text=get_text(urls=Url)
                chunks=get_chunks(text)
                vector_store=get_vector_store(chunks)
                st.session_state.conversation=get_conversational_chain(vector_store)

                st.success("Done")


if __name__ == '__main__':
    main()