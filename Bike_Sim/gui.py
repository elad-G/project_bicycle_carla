import tkinter as tk
from tkinter import messagebox

def submit_info():
    user_id = entry_id.get()
    if user_id:
        messagebox.showinfo("Welcome!", f"Hello {user_id}!\nEnjoy your CARLA simulation!")
    else:
        messagebox.showwarning("Input Required", "Please enter your ID.")

# Create main window
root = tk.Tk()
root.title("CARLA Simulator Info")
root.geometry("400x200")

# Add widgets
label_id = tk.Label(root, text="Enter your ID:", font=("Arial", 12))
label_id.pack(pady=10)

entry_id = tk.Entry(root, font=("Arial", 12))
entry_id.pack(pady=5)

info_label = tk.Label(root, text="Welcome to the CARLA simulator.\nDrive safely and enjoy!", font=("Arial", 10), wraplength=350, justify="center")
info_label.pack(pady=10)

submit_button = tk.Button(root, text="Submit", command=submit_info, font=("Arial", 12), bg="lightblue")
submit_button.pack(pady=10)

# Run the app
root.mainloop()
