import tkinter as tk
from tkinter import scrolledtext
import importlib.util
import threading
import queue
from mock_trial_interface import run_trial_workflow
from api.document import process_file_for_case_brief

class ChatbotApp:
    def __init__(self, root: tk.Tk, bot_name: str, row: int, col: int, msgQ: queue.Queue):
        self.bot_name = bot_name
        self.frame = tk.Frame(root, padx=10, pady=10)
        self.frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

        self.label = tk.Label(self.frame, text=bot_name, font=("Arial", 14))
        self.label.pack()

        self.chat_display = scrolledtext.ScrolledText(self.frame, wrap=tk.WORD, height=10, width=30)
        self.chat_display.pack(expand=True, fill='both')

        self.input_field = tk.Entry(self.frame, width=30)
        self.input_field.pack(fill='x', pady=5)

        self.send_button = tk.Button(self.frame, text="Send", command=self.start_workflow)
        self.send_button.pack()

        self.message_queue = msgQ  # Create a queue for messages
        self.check_queue()  # Start checking the queue

    def send_message(self, message):
        if message:
            self.chat_display.insert(tk.END, f"User: {message}\n")
            self.chat_display.yview(tk.END)  # Scroll to the end

    def display_bot_response(self, response):
        self.chat_display.insert(tk.END, f"{self.bot_name}: {response}\n")
        self.chat_display.yview(tk.END)  # Scroll to the end

    def run_trial(self, message):
        # Here you need to get your background and evidence
        background, evidence = process_file_for_case_brief("backend/api/Documents/Case1_brief.pdf", "pdf")
        
        # Start the workflow and pass the list of message queues
        run_trial_workflow(background, evidence, "defendant", [chatbot1.message_queue, chatbot2.message_queue, chatbot3.message_queue])

    def start_workflow(self):
        message = self.input_field.get()  # Get the message from the input field
        self.send_message(message)  # Send the user's message to the chat display
        self.input_field.delete(0, tk.END)

        # Start the trial workflow in a separate thread
        threading.Thread(target=self.run_trial, args=(message,)).start()


    def check_queue(self):
        try:
            while True:
                # Get the message from the queue
                message = self.message_queue.get_nowait()
                self.display_bot_response(message)  # Display bot's response
        except queue.Empty:
            pass

        # Continue checking the queue every 100ms
        self.frame.after(100, self.check_queue)


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

    chatbot1_queue = queue.Queue()
    chatbot2_queue = queue.Queue()
    chatbot3_queue = queue.Queue()

    chatbot1 = ChatbotApp(root, "Chatbot 1", 0, 0, chatbot1_queue)
    chatbot2 = ChatbotApp(root, "Chatbot 2", 0, 1, chatbot2_queue)
    chatbot3 = ChatbotApp(root, "Chatbot 3", 0, 2, chatbot3_queue)

    root.mainloop()

if __name__ == "__main__":
    create_chatbots()
