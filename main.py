from huggingface_hub import InferenceClient
import json

# Use a real conversational model that supports chat
llm_client = InferenceClient(model="microsoft/Phi-3-mini-4k-instruct", timeout=120)

def suggest_activity(user_input):
    prompt = f"User is feeling sad and asked for an activity suggestion: '{user_input}'. What can I suggest to improve their mood?"

    response = llm_client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )

    activity_suggestion = response['choices'][0]['message']['content']
    return activity_suggestion

def chat():
    print("Chatbot: Hello! Ask me for activity suggestions.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = suggest_activity(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()
