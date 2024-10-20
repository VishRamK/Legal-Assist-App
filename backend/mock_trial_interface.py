import sounddevice as sd
import numpy as np
import wave
from config import Config
from langchain import OpenAI, ConversationChain
from langchain.agents import AgentExecutor, Tool, initialize_agent
from tavily import TavilyClient
from anthropic import AnthropicClient
from api.judge import Judge
import os
import time
import pyttsx3

# Set up API clients for Tavily and Anthropic
tavily_client = TavilyClient(api_key=Config.TAVILY_API_KEY)
anthropic_client = AnthropicClient(api_key=Config.ANTHROPIC_API_KEY)

# Initialize LangChain's conversational agent using OpenAI
llm = OpenAI(temperature=0.7)  # You can also use Anthropic's Claude here

def text_to_speech(text: str):
    """Convert the given text to speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


# Define Legal Advisor using Tavily
def legal_advisor_tool(question: str) -> str:
    """Use Tavily's API to get legal advice."""
    response = tavily_client.get_legal_advice(query=question)
    return response.get("advice", "No legal advice available.")

# Define Prosecutor Agent using Anthropic
def prosecutor_tool(user_response: str) -> str:
    """Use Anthropic's API to ask challenging follow-up questions."""
    prompt = (
        f"You are a prosecutor. The user said: '{user_response}'. "
        "Identify any weaknesses in their response and double down with a follow-up question."
    )
    prosecutor_response = anthropic_client.completion(prompt=prompt, model="claude-v1")
    return prosecutor_response.get("completion")

# Define Judge Agent (using LangChain's built-in LLM)
def judge_tool(judge: Judge, responses: list) -> str:
    """Judge evaluates the user’s responses."""
    prompt = """
    You are a judge evaluating the user's performance during a trial. Analyze their responses based on:
    1) Tone (confidence, hesitation)
    2) Legal soundness (how well the responses align with legal principles)
    3) Consistency (how coherent their answers are)
    Provide feedback on how the user can improve.
    Here are the user’s responses: {responses}
    """
    formatted_prompt = prompt.format(responses="\n".join(responses))
    chain = ConversationChain(llm=llm)
    return chain.run(formatted_prompt)


def record_audio(duration: int, filename: str) -> None:
    """Record audio from the microphone for a given duration."""
    print("Recording...")
    fs = 44100  # Sample rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write_wave_file(filename, recording, fs)
    print("Recording finished.")


def write_wave_file(filename: str, data: np.ndarray, fs: int) -> None:
    """Write recorded audio to a WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(fs)
        wf.writeframes(data.tobytes())


# Define the tools (agents) for the workflow
tools = [
    Tool(name="Legal Advisor", func=legal_advisor_tool, description="Provides legal advice."),
    Tool(name="Prosecutor", func=prosecutor_tool, description="Challenges the user with questions."),
    Tool(name="Judge", func=judge_tool, description="Evaluates the user's responses."),
]

# Initialize the agent with the tools
agent_executor = initialize_agent(tools, llm, agent_type="zero-shot-react-description")

# Function to run the multi-agent trial workflow
def run_trial_workflow(questions: list[str]):
    user_responses = []
    nxt = 0
    current_question = questions[nxt]

    judge = Judge()
    while True:
        # Step 1: Ask the current question via audio
        text_to_speech(current_question)

        # Step 2: Legal Advisor gives advice
        legal_advice = agent_executor.run("Legal Advisor", current_question)
        text_to_speech(legal_advice)

        # Step 3: Record audio response from the user
        audio_filename = "user_response.wav"
        record_audio(duration=10, filename=audio_filename)  # Record for 10 seconds
        
        # Step 4: Analyze tone of the audio
        judge.new_response(audio_filename)
        judge.evaluate_response()
        
        # Step 5: Convert audio to text
        user_response = judge.current_response
        print(f"User response: {user_response}")
        
        # Step 6: Prosecutor evaluates response and asks a follow-up question
        follow_up_question = agent_executor.run("Prosecutor", user_response) if judge.current_response_weight < 60 else None
        if not follow_up_question:
            if nxt >= len(questions):
                break
            current_question = questions[nxt]
        else:
            current_question = follow_up_question
    
    # Step 7: Judge evaluates all responses and gives feedback
    judge_feedback = agent_executor.run("Judge", judge, user_responses)
    text_to_speech(judge_feedback)

# Start Trial
questions = ["Where were you on the night of the incident?"] # Example
run_trial_workflow(questions)
