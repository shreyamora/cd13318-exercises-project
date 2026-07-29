from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt
    system_prompt = (
        "You are a NASA Mission Intelligence assistant. Answer the user's questions "
        "about NASA space missions using the provided context from mission documents. "
        "Base your answers on the context whenever possible and cite the relevant "
        "mission or source. If the context does not contain enough information to "
        "answer, say so clearly instead of making up details."
    )

    # Set context in messages
    messages: List[Dict] = [
        {"role": "system", "content": system_prompt},
    ]

    if context:
        messages.append({
            "role": "system",
            "content": f"Context from NASA mission documents:\n{context}",
        })

    # Add chat history
    if conversation_history:
        for turn in conversation_history:
            role = turn.get("role")
            content = turn.get("content")
            if role and content:
                messages.append({"role": role, "content": content})

    # Add the current user message
    messages.append({"role": "user", "content": user_message})

    # Create OpenAI Client
    client = OpenAI(api_key=openai_key)

    # Send request to OpenAI
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        # Return response
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"