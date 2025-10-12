from flask import Flask, render_template, request
from optimizer import optimize_prompt
from evaluator import evaluate_prompt
import sqlite3

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/optimize", methods=["POST"])
def optimize():
    user_prompt = request.form["prompt"]
    optimized = optimize_prompt(user_prompt)
    score = evaluate_prompt(optimized)

    # Save to SQLite
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS prompts (raw TEXT, optimized TEXT, clarity REAL)")
    c.execute("INSERT INTO prompts VALUES (?, ?, ?)", (user_prompt, optimized, score["clarity_score"]))
    conn.commit()
    conn.close()

    return render_template("index.html", optimized_prompt=optimized)

if __name__ == "__main__":
    app.run(debug=True)
