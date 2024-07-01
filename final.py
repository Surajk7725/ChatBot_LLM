import streamlit as st
import time
import boto3
import openai
import google.generativeai as genai
from IPython.display import Markdown
from streamlit_chat import message
import json

def get_model_response(customer_input, aws_access_key_id, aws_secret_access_key, boto_session):
    if not aws_access_key_id and not aws_secret_access_key:
        st.info("🔑 Access Key Id or Secret Access Key are not provided yet!")
        return None

    client = boto_session.client(
        service_name='bedrock-runtime',
        region_name="us-east-1"
    )

    prompt = f"\n\nHuman:{customer_input}\n\nAssistant:"

    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 1000,
            "temperature": 0.7,
            "topP": 0.9,
            "stopSequences": []
        }
    })

    response = client.invoke_model(
        body=body,
        modelId="amazon.titan-text-premier-v1:0",
        accept='application/json',
        contentType='application/json'
    )

    msg = json.loads(response['body'].read().decode('utf-8'))
    response_text = msg['results'][0]['outputText']

    return response_text

# Function to load OpenAI model and get response
def get_chatgpt_response(api_key, question):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        stream=True,
        temperature=0.7,   # Set temperature internally
        top_p=0.9,         # Set top_p internally
        max_tokens=1000     # Set max_tokens internally
    )

    full_response = ""
    for response in response:
        token_text = response.choices[0].delta.get("content", "")
        full_response += token_text

    return full_response

# Function to load Gemini model and get response
def get_gemini_response(api_key, question):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text

# Function to load Azure OpenAI GPT-3.5 Turbo model and get response
def get_azure_gpt_response(api_base, api_version, api_key, question):
    openai.api_type = "azure"
    openai.api_base = api_base
    openai.api_version = api_version
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        temperature=0.7,
        max_tokens=2000,
        top_p=0.95,
        messages=[
            {"role": "system", "content": "How can I help you?"},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message.content

## Initialize our Streamlit app
st.set_page_config(
    page_title="LLM Chatbot Models",
    page_icon="🤖",
    layout="centered"
)
st.title("💬 Research on LLM Models")
st.caption("🚀 A dynamic conversational experience powered by various LLMs.")

# Sidebar for model selection and API key input
st.sidebar.header("🔧 Configuration")
model_choice = st.sidebar.radio("🎛 Choose the model:", ["Gemini Pro", "GPT-4.0", "Azure OpenAI GPT-3.5 Turbo","AWS"])
api_key = None

if model_choice == "Gemini Pro":
    api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")
elif model_choice == "GPT-4.0":
    api_key = st.sidebar.text_input("🔑 Chat-GPT API Key", type="password")
elif model_choice == "AWS":
    aws_access_key_id = st.sidebar.text_input("🔑 AWS Access Key Id", placeholder="access key", type="password")
    api_key = st.sidebar.text_input("🗝 AWS Secret Access Key", placeholder="secret", type="password")   
else:
    api_base = st.sidebar.text_input("🌐 Azure API Base URL", placeholder="https://<name>.openai.azure.com/")
    api_version = st.sidebar.text_input("📛 API Version", "2023-03-15-preview")
    api_key = st.sidebar.text_input("🔑 API Key", type="password")


# Initialize chat session in Streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat history
for message in st.session_state.chat_history:
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(message["content"])

# Input field for user's message at the bottom of the page
input_text = st.chat_input("Ask your question here:")

if input_text:
    if api_key:
        # Add user's message to chat and display it
        st.session_state.chat_history.append({"role": "user", "content": input_text})
        st.chat_message("user").markdown(input_text)

        # Send user's message to the selected AI model and get a response
        start_time = time.time()
        if model_choice == "Gemini Pro":
            response = get_gemini_response(api_key, input_text)
        elif model_choice == "GPT-4.0":
            response = get_chatgpt_response(api_key, input_text)
        elif model_choice == "Azure OpenAI GPT-3.5 Turbo":
            response = get_azure_gpt_response(api_base, api_version, api_key, input_text)
        elif model_choice=="AWS":
            boto_session = boto3.session.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=api_key)

            response=get_model_response(input_text,aws_access_key_id,api_key,boto_session)

        end_time = time.time()

        assistant_response = response
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Calculate evaluation metrics
        latency = end_time - start_time
        input_tokens = len(input_text.split())
        output_tokens = len(assistant_response.split())
        throughput = output_tokens / latency

        # Display the AI's response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Display metrics in the sidebar
        st.sidebar.subheader("📊 Evaluation Metrics")
        st.sidebar.write(f"- **Throughput:** {throughput:.6f} tokens/second")
        st.sidebar.write(f"- **Latency:** {latency:.6f} seconds")
        st.sidebar.write(f"- **Input Tokens:** {input_tokens}")
        st.sidebar.write(f"- **Output Tokens:** {output_tokens}")
    else:
        st.sidebar.error("⚠️ Please enter your API key to proceed.")
