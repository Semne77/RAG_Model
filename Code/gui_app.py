import tkinter as tk
from tkinter import scrolledtext
from logic import ask_question  # Your existing function

class RAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎾 Tennis Q&A - RAG App")
        self.root.geometry("650x480")
        self.root.configure(bg="#377DB8")

        # Frame for content
        frame = tk.Frame(root, bg="#f8f9fa")
        frame.pack(pady=20)

        # Title
        title = tk.Label(frame, text="Ask Anything About The BIG 3 of Tennis 🎾", font=("Arial", 16, "bold"), bg="#f8f9fa", fg="#333")
        title.pack(pady=10)

        # Entry
        self.question_entry = tk.Entry(frame, width=50, font=("Arial", 12), bd=2, relief="groove")
        self.question_entry.pack(pady=10)

        # Submit Button
        self.submit_button = tk.Button(frame, text="🔍 Get Answer", command=self.get_answer, bg="#dfff4f", fg="Black",
                                       activebackground="#45a049", font=("Arial", 12), padx=10, pady=5, bd=0)
        self.submit_button.pack(pady=5)

        # Output Text
        self.output_text = scrolledtext.ScrolledText(root, width=70, height=12, font=("Arial", 11), bg="#f1f1f1",
                                                     relief="flat", wrap=tk.WORD, fg="Black")
        self.output_text.pack(pady=10)
        self.output_text.config(state=tk.DISABLED)

        # Exit Button
        self.exit_button = tk.Button(root, text="Exit", command=root.quit, bg="#dfff4f", fg="Black",
                                     activebackground="#c0392b", font=("Arial", 12), padx=10, pady=5, bd=0)
        self.exit_button.pack(pady=10)

    def get_answer(self):
        question = self.question_entry.get().strip()
        if question == "":
            self.append_output("⚠️ Please enter a question.")
            return

        self.append_output(f"\n🔍 Question: {question}")
        self.append_output("⏳ Searching...")

        # Call your existing function
        answer = ask_question(question)
        self.append_output(f"🧠 Answer: {answer}")

    def append_output(self, text):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop()
