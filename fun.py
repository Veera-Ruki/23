import sqlite3
global num
num = {}

def create_table():
    with sqlite3.connect("login.db") as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                email TEXT NOT NULL,
                role TEXT NOT NULL
            )
        ''')
def login_interface(username, password):
    
    if not username or not password:
        return "Username and password are required for login."
        
    else:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()

            if user and user[2] == password:
                num = 1
                return "Login successful"
                
            else:
                return "Invalid username or Password"
        except sqlite3.OperationalError as e:
            return "An error occurred while trying to log in: {e}"
            response["success"] = False
    return response   
        

def signup_interface(new_username, new_password, new_email):
    
    if not new_username or not new_password or not new_email:
        return "All fields are required for signup."
        
    else:
        role = "user"
        try:
            conn = sqlite3.connect("login.db")
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)",
                           (new_username, new_password, new_email, role))
            conn.commit()
            login = login_interface(username,password)
            return "Signup successful!", login
            
        except sqlite3.IntegrityError:
            return "Username already exists. Please choose a different one."

         


        
def share(out, inp):
    whatsapp_link = f"https://wa.me/?text={out}%20{inp}"
    facebook_link = f"https://www.facebook.com/sharer/sharer.php?u={inp}&quote={out}"
    instagram_link = f"https://www.instagram.com/?url={inp}&title={out}"
    twitter_link = f"https://twitter.com/intent/tweet?text={out}%20{inp}"
    button_label = "Share"
    show_icons = False
    def toggle_icons():
        nonlocal show_icons, button_label
        show_icons = not show_icons
        

    button = gr.Button(button_label)
    button.click(toggle_icons)

    if show_icons:
        display_style = "flex"
    else:
        display_style = "none"

    gr.HTML(f'<div id="iconsContainer" style="display: {display_style};">'
        f'<a href="{whatsapp_link}" target="_blank"><img src="icons/wa.png" alt="WhatsApp"></a>'
        f'<a href="{facebook_link}" target="_blank"><img src="icons/fb.png" alt="Facebook"></a>'
        f'<a href="{instagram_link}" target="_blank"><img src="icons/in.png" alt="Instagram"></a>'
        f'<a href="{twitter_link}" target="_blank"><img src="icons/tw.png" alt="Twitter"></a>'
        '</div>')

    


