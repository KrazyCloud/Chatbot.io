import streamlit as st
from files.cleaning_text import clean_transcript
from files.extract import extract_video_id
from files.generator import *
from files.transcript import transcript_generator
import torch

# Function to get a limited number of words from a sentence
def get_limited_words(sentence, limit=30):
    words = sentence.split()
    limited_words = ' '.join(words[:limit])
    return limited_words

def main():
    # titles and head
    st.title("_:red[Youtube]_ Transcript Chatbot")
    st.markdown("Welcome to the chat! Type your messages below.")
    st.header("",divider="gray")

    video_url = st.text_input("Enter YouTube Video URL")
    if video_url:
        video_id = extract_video_id(video_url)
        if video_id:
            transcript = transcript_generator(video_id)
            cleaned_transcript = clean_transcript(transcript)

            user_input = st.text_input("Ask something")

            if user_input:
                # Tokenize the question and transcript
                inputs = tokenizer.encode_plus(user_input, cleaned_transcript, add_special_tokens=True, return_tensors='pt', max_length=512, truncation=True)

                # Get model predictions
                with torch.no_grad():
                    outputs = model(**inputs)

                start_scores = outputs.start_logits
                end_scores = outputs.end_logits

                # Get the most probable start and end positions
                start_index = torch.argmax(start_scores)
                end_index = torch.argmax(end_scores)

                # Get the answer tokens from the input IDs
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].tolist()[0])

                # Formulate the answer by combining the tokens
                answer = tokenizer.convert_tokens_to_string(tokens[start_index:end_index+1])
                limited_response = get_limited_words(answer, 50)
                st.text_area("Chatbot's Response", limited_response)
            else:
                st.write("Please ask something")
        else:
            st.write("Invalid YouTube Video URL")


if __name__ == "__main__":
    main()