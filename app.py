import os, json, sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
from ml_model import INPUT_COLUMNS, DISPLAY_FIELDS, train_models, clean_dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "development-secret-change-before-deployment")

DATA = os.path.join(ROOT, "uploads", "heart_dataset.csv")
MODEL = os.path.join(ROOT, "models", "heart_model.joblib")
REPORT = os.path.join(ROOT, "models", "training_report.json")
DB = os.path.join(ROOT, "heart_disease.db")

os.makedirs(os.path.dirname(DATA), exist_ok=True)
os.makedirs(os.path.dirname(MODEL), exist_ok=True)

# --- Flask-Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "error"

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    con = db()
    row = con.execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()
    con.close()
    if row:
        return User(id=row[0], username=row[1])
    return None

# --- Database Setup ---
def db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS prediction_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        created_at TEXT,
        prediction TEXT,
        probability REAL)""")
    
    # Seed default admin account if it doesn't exist yet
    admin_exists = con.execute("SELECT 1 FROM users WHERE username = ?", ("admin",)).fetchone()
    if not admin_exists:
        con.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            ("admin", generate_password_hash("heart123"))
        )
    con.commit()
    return con

# --- Routes ---
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/signin", methods=["GET", "POST"])
@app.route("/signin", methods=["GET", "POST"])
def signin():
    if current_user.is_authenticated:
        return redirect(url_for("abstract"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Username and password are required.", "error")
            return redirect(url_for("signin"))
        
        con = db()
        try:
            cur = con.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, generate_password_hash(password))
            )
            con.commit()
            
            # Log the newly created user in automatically
            user = User(id=cur.lastrowid, username=username)
            login_user(user)
            
            flash("Account created! Welcome.", "success")
            return redirect(url_for("abstract"))
        except sqlite3.IntegrityError:
            flash("Username already exists. Choose another.", "error")
        finally:
            con.close()
    return render_template("signin.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("abstract"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        con = db()
        row = con.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
        con.close()

        if row and check_password_hash(row[2], password):
            user = User(id=row[0], username=row[1])
            login_user(user)
            next_page = request.args.get("next")
            return redirect(next_page or url_for("abstract"))
        
        flash("Incorrect username or password.", "error")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been signed out.", "success")
    return redirect(url_for("home"))

@app.route("/abstract")
@login_required
def abstract():
    return render_template("abstract.html")

@app.route("/preview")
@login_required
def preview():
    if not os.path.exists(DATA):
        flash("Upload a CSV dataset on the Training page first.", "error")
        return redirect(url_for("training"))
    df = pd.read_csv(DATA).fillna("")
    return render_template("preview.html", columns=list(df.columns), rows=df.head(12).values.tolist(), total=len(df))

@app.route("/training", methods=["GET", "POST"])
@login_required
def training():
    report = None
    if os.path.exists(REPORT):
        with open(REPORT, encoding="utf-8") as f:
            report = json.load(f)
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or not file.filename.lower().endswith(".csv"):
            flash("Choose a CSV dataset file.", "error")
            return redirect(url_for("training"))
        file.save(DATA)
        try:
            df = pd.read_csv(DATA)
            clean_dataset(df)
            flash(f"Dataset uploaded successfully: {len(df)} rows found. Open Preview or train the model.", "success")
            return redirect(url_for("preview"))
        except Exception as e:
            os.remove(DATA)
            flash("Dataset could not be accepted: " + str(e), "error")
            return redirect(url_for("training"))
    return render_template("training.html", report=report, model_ready=os.path.exists(MODEL))

@app.route("/train-model", methods=["POST"])
@login_required
def train_model():
    if not os.path.exists(DATA):
        flash("Upload the dataset before training.", "error")
        return redirect(url_for("training"))
    try:
        report, model = train_models(DATA)
        joblib.dump(model, MODEL)
        with open(REPORT, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        flash("Training and testing completed. The prediction model is ready.", "success")
        return redirect(url_for("predict"))
    except Exception as e:
        flash("Training failed: " + str(e), "error")
    return redirect(url_for("training"))

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if not os.path.exists(MODEL):
        flash("Please upload your dataset and train the model first.", "error")
        return redirect(url_for("training"))
    if request.method == "POST":
        try:
            values = {}
            for col in INPUT_COLUMNS:
                values[col] = float(request.form[col])
            model = joblib.load(MODEL)
            X = pd.DataFrame([[values[c] for c in INPUT_COLUMNS]], columns=INPUT_COLUMNS)
            pred = int(model.predict(X)[0])
            prob = float(model.predict_proba(X)[0][1])
            outcome = "Possible heart disease" if pred == 1 else "No heart disease predicted"
            con = db()
            con.execute(
                "INSERT INTO prediction_history(username, created_at, prediction, probability) VALUES(?, ?, ?, ?)",
                (current_user.username, datetime.now().isoformat(timespec="seconds"), outcome, prob)
            )
            con.commit()
            con.close()
            return render_template("result.html", outcome=outcome, probability=round(prob * 100, 1), positive=bool(pred))
        except Exception as e:
            flash("Please check all patient values. " + str(e), "error")
            return redirect(url_for("predict"))
    return render_template("predict.html", fields=DISPLAY_FIELDS)

@app.route("/history")
@login_required
def history():
    con = db()
    rows = con.execute(
        "SELECT created_at, prediction, probability FROM prediction_history WHERE username = ? ORDER BY id DESC LIMIT 30",
        (current_user.username,)
    ).fetchall()
    con.close()
    return render_template("history.html", rows=rows)

if __name__ == "__main__":
    db().close()
    app.run(debug=True,port=5000)
