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

         

def check(num):
    
    if num:
        return gr.Group(visible=True) 
    else:
        return gr.Group(visible=False)



