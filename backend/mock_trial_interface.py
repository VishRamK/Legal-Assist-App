import sounddevice as sd
import numpy as np
import wave
from config import Config
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor, Tool, initialize_agent
from tavily import TavilyClient
from anthropic import Client
from api.judge import Judge
import os
from dotenv import load_dotenv
import speech_recognition as sr
import pyaudio
import time
import openai
import pyttsx3
import keyboard

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
THRESHOLD = 0.01  # The threshold for detecting sound, can be tuned based on sensitivity

load_dotenv()
# Set up API clients for Tavily and Anthropic
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
anthropic_client = Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

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

def record_audio_on_space(output_file="output.wav"):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']}")

    frames = []
    recording = False
    print("Press SPACE to start recording...")

    while True:
        if keyboard.is_pressed('space'):  # Check if the spacebar is pressed
            if not recording:
                print("Recording started...Press Enter to stop")
                recording = True

                # Start recording
                while recording:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)

                    # Optional: Print or process audio data as needed
                    # Example: amplitude = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
                    # print("Amplitude:", amplitude)

                    # Check for space to stop recording
                    if keyboard.is_pressed('enter'):
                        print("Recording stopped.")
                        recording = False
                        break
            break

    # Save the recorded audio to file
    save_audio(frames, p, output_file)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print(f"Audio saved to {output_file}")


def save_audio(frames, p, output_file):
    """Save the recorded frames to a .wav file."""
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio saved to {output_file}")


def text_to_speech(text: str):
    """Convert the given text to speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


# Define Legal Advisor using Tavily
def legal_advisor_tool(question: str) -> str:
    """Use Tavily's API to get legal advice."""
    try:
        response = tavily_client.get_search_context(query=question)
    except:
        response = "Answer as you please"
    prompt = (
        f"Summarize {response} to two or three best options to proceed with"
        f"answering this question that was asked by the prosecutor: {question}."
    )
    return llm.invoke(prompt)

# Define Prosecutor Agent using Anthropic
def prosecutor_tool(user_response: str) -> str:
    """Use Anthropic's API to ask challenging follow-up questions."""
    prompt = (
        f"You are a prosecutor. The user said: '{user_response}'. "
        "Identify any weaknesses in their response and double down with a follow-up question."
    )
    return llm.invoke(prompt)

# Define Judge Agent (using LangChain's built-in LLM)
def judge_tool(responses: list) -> str:
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
    return llm.invoke(formatted_prompt)


# def record_audio(duration: int, filename: str) -> None:
#     """Record audio from the microphone for a given duration."""
#     print("Recording...")
#     fs = 44100  # Sample rate
#     recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
#     sd.wait()  # Wait until recording is finished
#     write_wave_file(filename, recording, fs)
#     print("Recording finished.")


# def write_wave_file(filename: str, data: np.ndarray, fs: int) -> None:
#     """Write recorded audio to a WAV file."""
#     with wave.open(filename, 'wb') as wf:
#         wf.setnchannels(1)  # Mono
#         wf.setsampwidth(2)  # 2 bytes for int16
#         wf.setframerate(fs)
#         wf.writeframes(data.tobytes())


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
        legal_advice = legal_advisor_tool(question=current_question)
        text_to_speech(legal_advice)

        # Step 3: Record audio response from the user
        audio_filename = "user_response.wav"
        record_audio(audio_filename)  # Record
        
        # Step 4: Analyze tone of the audio
        judge.new_response(audio_filename)
        judge.evaluate_response("")
        
        # Step 5: Convert audio to text
        user_response = judge.current_response
        print(f"User response: {user_response}")
        
        # Step 6: Prosecutor evaluates response and asks a follow-up question
        follow_up_question = prosecutor_tool(user_response=user_response) if judge.current_response_weight < 60 else None
        if not follow_up_question:
            if nxt >= len(questions):
                break
            current_question = questions[nxt]
        else:
            current_question = follow_up_question
    
    # Step 7: Judge evaluates all responses and gives feedback
    judge_feedback = judge_tool(responses=user_responses)
    print(judge.calculate_win_probability())
    text_to_speech(judge_feedback)

# Start Trial
questions = ["Where were you on the night of the incident?"] # Example
run_trial_workflow(questions)
