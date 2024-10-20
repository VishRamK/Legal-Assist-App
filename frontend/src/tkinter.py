import tkinter as tk
from tkinter import scrolledtext
import importlib.util
import os

# Path to your processing file (process_file.py)
process_file_path = "path/to/your/process_file.py"

class ChatbotApp:
    def __init__(self, root, bot_name, row, col):
        self.bot_name = bot_name

        # Frame for the chatbot
        self.frame = tk.Frame(root, padx=10, pady=10)
        self.frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

        # Label for the chatbot
        self.label = tk.Label(self.frame, text=bot_name, font=("Arial", 14))
        self.label.pack()

        # ScrolledText for the chat display
        self.chat_display = scrolledtext.ScrolledText(self.frame, wrap=tk.WORD, height=10, width=30)
        self.chat_display.pack(expand=True, fill='both')

        # Entry field for the user input
        self.input_field = tk.Entry(self.frame, width=30)
        self.input_field.pack(fill='x', pady=5)

        # Button to send the message
        self.send_button = tk.Button(self.frame, text="Send", command=self.send_message)
        self.send_button.pack()

    def send_message(self):
        user_message = self.input_field.get()
        if user_message:
            # Display the user's message in the chat display
            self.chat_display.insert(tk.END, f"User: {user_message}\n")

            # Call the handler to process the input and display the responses
            handle_message(user_message)

            # Clear the input field
            self.input_field.delete(0, tk.END)

    def display_bot_response(self, response):
        # Display the bot's response in the chat display
        self.chat_display.insert(tk.END, f"{self.bot_name}: {response}\n")

def fetch_processing_output(x1):
    # Load the process_file.py dynamically and call each function separately
    spec = importlib.util.spec_from_file_location("process_file", process_file_path)
    process_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(process_file)

    # Call the three functions and get y1, y2, y3
    y1 = process_file.process_for_chatbot1(x1)
    y2 = process_file.process_for_chatbot2(x1)
    y3 = process_file.process_for_chatbot3(x1)
    return y1, y2, y3

def handle_message(user_message):
    # Send x1 (user_message) to the processing file and get y1, y2, y3
    y1, y2, y3 = fetch_processing_output(user_message)

    # Display the responses in the respective chatbots
    chatbot1.display_bot_response(y1)
    chatbot2.display_bot_response(y2)
    chatbot3.display_bot_response(y3)

def create_chatbots():
    root = tk.Tk()
    root.title("Multi-Chatbot Window")

    # Make the window full screen
    root.state('zoomed')

    # Configure rows and columns of the grid to expand
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)

    # Create three chatbots
    global chatbot1, chatbot2, chatbot3
    chatbot1 = ChatbotApp(root, "Chatbot 1", 0, 0)
    chatbot2 = ChatbotApp(root, "Chatbot 2", 0, 1)
    chatbot3 = ChatbotApp(root, "Chatbot 3", 0, 2)

    root.mainloop()

if __name__ == "__main__":
    create_chatbots()
