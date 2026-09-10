from flask import Flask, jsonify

app = Flask(__name__)


@app.get("/")
def index():
    return jsonify({
        "project": "hiver-support-agent",
        "status": "ready",
        "note": "This repo is a Python support-agent project. Vercel needs an entrypoint file, so this minimal app exposes the project status.",
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
