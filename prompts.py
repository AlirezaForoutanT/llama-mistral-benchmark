# prompts.py

CHAT_CONVERSATIONS = [
    # Conversation 1
    [
        {"role": "user", "content": "Hello, how are you today?"},
        {"role": "assistant", "content": ""},  # model will fill
        {"role": "user", "content": "Could you tell me a joke about computers?"},
        {"role": "assistant", "content": ""},  # model will fill
    ],
    # Conversation 2
    [
        {"role": "user", "content": "I’m planning a road trip. Any tips for choosing a destination?"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Thanks! Could you also recommend some budget-friendly accommodations?"},
        {"role": "assistant", "content": ""},
    ],
    # Conversation 3
    [
        {"role": "user", "content": "What’s the best way to stay productive while working remotely?"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "How about dealing with distractions at home?"},
        {"role": "assistant", "content": ""},
    ],
]

QA_PROMPTS = [
    {
        "question": "Who wrote the novel 'Pride and Prejudice'?",
        "reference_answer": "Jane Austen",
    },
    {
        "question": "What is the chemical symbol for gold?",
        "reference_answer": "Au",
    },
    {
        "question": "In what year did the first human land on the Moon?",
        "reference_answer": "1969",
    },
    {
        "question": "What is the capital city of Canada?",
        "reference_answer": "Ottawa",
    },
    {
        "question": "Explain what a black hole is in simple terms.",
        "reference_answer": "A region in space where gravity is so strong that nothing, not even light, can escape from it.",
    },
]
