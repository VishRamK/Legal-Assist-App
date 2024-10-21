import sounddevice as sd
import numpy as np
import wave
from langchain_openai import OpenAI
from tavily import TavilyClient
from api.judge import Judge
import os
from dotenv import load_dotenv
import pyaudio
import time
import pyttsx3
import keyboard
# from api.document import process_file_for_case_brief
import queue
import threading
import soundfile as sf


# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
THRESHOLD = 0.01  # The threshold for detecting sound, can be tuned based on sensitivity

# Initialize the TTS engine
engine = pyttsx3.init()

# Create a queue for messages
tts_queue = queue.Queue()

def tts_worker():
    """Thread worker function for text-to-speech."""
    while True:
        text = tts_queue.get()
        if text is None:  # Exit signal
            break
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()

# Start the TTS worker thread
tts_thread = threading.Thread(target=tts_worker)
tts_thread.start()

load_dotenv()
# Set up API clients for Tavily and Anthropic
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Initialize LangChain's conversational agent using OpenAI
llm = OpenAI(temperature=0.7, api_key=os.getenv("OPEN_AI_KEY"))  # You can also use Anthropic's Claude here

def write_wave_file(filename: str, data: np.ndarray, fs: int) -> None:
    """Write recorded audio to a WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(fs)
        wf.writeframes(data.tobytes())

def record_audio(filename: str) -> None:
    """Record audio from the microphone until the user stops it."""
    fs = 44100  # Sample rate
    recording = []
    print("Press SPACE to start recording...")

    while True:
        # Wait for the spacebar to be pressed to start recording
        if keyboard.is_pressed('space'):
            print("Recording started... Press ENTER to stop.")
            # Start recording
            with sd.InputStream(samplerate=fs, channels=1, dtype='int16') as stream:
                while True:
                    data = stream.read(1024)  # Read audio data
                    recording.append(data[0])  # Store audio frames
                    
                    # Check for the enter key to stop recording
                    if keyboard.is_pressed('enter'):
                        print("Recording stopped.")
                        break
                
            # Convert list of frames to a numpy array
            audio_data = np.concatenate(recording, axis=0)

            # Write to WAV file
            write_wave_file(filename, audio_data, fs)
            break  # Exit the main while loop after recording

        time.sleep(0.1)


def text_to_speech(text: str):
    """Convert the given text to speech."""
    if not text:
        print("No text provided for speech.")
        return
    
    # Put text into the queue for the TTS worker
    tts_queue.put(text)


def question_generator(background: str, user: str):
    prompt = (
        f"The following is the background of a civil court case: \n {background}. \n"
        f"Knowing this, give me a list of 3 different questions you would ask the {user} as the opposing lawyer."
        "The questions must all be single sentences (no periods in between) ending with a question mark"
    )
    questions_to_parse = llm.invoke(prompt)
    n = len(questions_to_parse)
    i = 0
    j = 0
    questions = []
    while i < n and j < n:
        if questions_to_parse[j] != "?":
            j += 1
            if questions_to_parse[j] == ".":
                i = j + 1
                if i >= n:
                    break
                while not questions_to_parse[i].isalpha():
                    i += 1
                    if i >= n:
                        break
        else:
            j += 1
            questions.append(questions_to_parse[i:j])
    return questions


# Define Legal Advisor using Tavily
def legal_advisor_tool(question: str, background: str, evidence: str) -> str:
    """Use Tavily's API to get legal advice based on background and evidence."""
    try:
        response = tavily_client.get_search_context(query=question)
    except:
        response = "the next course of action"
    prompt = (
        f"The case background is: {background}. \n"
        f"The evidence includes: {evidence}. \n"
        f"Based on this if {response} is relevant, summarize it into two or three best options to proceed with "
        f"answering this question that was asked by the prosecutor: {question}."
    )
    output = llm.invoke(prompt)
    return f"As your AI legal assistant, here is my advice:\n{output}"


# Define Prosecutor Agent using Anthropic
def prosecutor_tool(user_response: str) -> str:
    prompt = (
        f"You are a prosecutor. The user said: '{user_response}'. "
        "If there are any weaknesses in the response, double down with a follow-up question."
    )
    return llm.invoke(prompt)

# Define Judge Agent (using LangChain's built-in LLM)
def judge_tool(judge: Judge, evidence) -> str:
    """Judge evaluates the user’s responses."""
    prompt = None
    response = judge.current_response
    w = judge.current_response_weight
    results = "\n"
    if judge.responses:
        
        results = "And their previous questions and responses:\n" + "".join(f"Question: {q}; Answer: {a}\n" for q, a in judge.responses) + "\n"
    if w < 60:
        prompt = (
            f"You are a judge evaluating the user's performance during a trial. Analyze their response based on:"
            "1) Tone (confidence, hesitation)"
            "2) Legal soundness (how well the responses align with legal principles)"
            f"3) Consistency compared with: \n {evidence}\n{results}"
            "Provide feedback on how the user can improve."
            f"Here is the user’s response to the question {response[0]}: {response[1]}"
        )
    elif w > 80:
        prompt = (
            f"You are a judge evaluating the user's performance during a trial. Analyze the strengths of their:"
            f"Here is the user’s response: {response}"
        )
    else:
        prompt = (
            f"You are a judge evaluating the user's performance during a trial. Analyze their response based on:"
            "1) Tone (confidence, hesitation)"
            "2) Legal soundness (how well the responses align with legal principles)"
            f"3) Consistency compared with: \n {evidence} \n And their previous questions and responses:\n{results}"
            "Provide feedback on the strengths of the response as well as how the user can improve."
            f"Here is the user’s response: {response}"
        )

    feedback = llm.invoke(prompt)
    return f"As your judge, here is my feedback:\n{feedback}"


# # Define the tools (agents) for the workflow
# tools = [
#     Tool(name="Legal Advisor", func=legal_advisor_tool, description="Provides legal advice."),
#     Tool(name="Prosecutor", func=prosecutor_tool, description="Challenges the user with questions."),
#     Tool(name="Judge", func=judge_tool, description="Evaluates the user's responses."),
# ]

# # Initialize the agent with the tools
# agent_executor = initialize_agent(tools, llm, agent_type="zero-shot-react-description")

# Function to run the multi-agent trial workflow
def run_trial_workflow(evidence: str, background: str, user: str, message_queues: list[queue.Queue]):   
    questions = question_generator(background, user)    
    nxt = 0
    current_question = questions[nxt]

    judge = Judge()
    while True:
        # Step 1: Ask the current question via audio
        message_queues[1].put(f"Current question: {current_question}\n\n")
        text_to_speech(current_question)

        # Step 2: Legal Advisor gives advice
        legal_advice = legal_advisor_tool(question=current_question, evidence=evidence, background=background)
        message_queues[0].put(f"As your AI legal assistant, here is my advice:\n{legal_advice}\n\n")
        text_to_speech(legal_advice)

        # Step 3: Record audio response from the user
        audio_filename = "user_response.wav"
        record_audio(audio_filename)  # Record
        
        # Step 4: Analyze tone of the audio
        judge.new_response(current_question, audio_filename)
        judge.evaluate_response(evidence)
        
        # Step 5: Convert audio to text
        user_response = judge.current_response[1]
        message_queues[1].put(f"{user}'s response: {user_response}\n\n")
        
        # Step 6: The Judge gives feedback
        judge_feedback = judge_tool(judge, evidence)
        message_queues[2].put(f"As your judge, here is my feedback:\n{judge_feedback}\n\n")
        text_to_speech(judge_feedback)
        
        # Step 7: Prosecutor evaluates response and asks a follow-up question
        message_queues[2].put(f"After careful evaluation, I have given your response a score of {judge.current_response_weight} out of 100.\n\n")
        follow_up_question = prosecutor_tool(user_response=user_response) if judge.current_response_weight < 60 else None
        if not follow_up_question:
            if nxt >= len(questions):
                break
            current_question = questions[nxt]
        else:
            current_question = follow_up_question
    
    # Step 8: Judge evaluates all responses and gives feedback
    win_prob = judge.calculate_win_probability()
    message_queues[2].put(f"Your overall performance in this trial gives you a {win_prob}% chance of winning this case.")

# Start Trial