# HOW TO RUN (start FastAPI server first):
# .venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
#
# Then in a second terminal:
# .venv\Scripts\python -m locust -f locustfile.py --headless \
#   -u 50 -r 10 --run-time 60s \
#   --host http://localhost:8000 \
#   --html load_test_report.html
#
# Effective throughput = RPS on /batch_predict × 10 (batch size)
# Target: effective msgs/sec >= 847

from locust import HttpUser, task, between, events

class DarkTraceUser(HttpUser):
    # Near-zero wait time to maximize request rate
    wait_time = between(0, 0.01)

    @task(1)
    def predict_single(self):
        self.client.post("/predict", json={
            "message": "got that fire plug dm me asap, cash only 🔥"
        })

    @task(3)
    def predict_batch(self):
        # 10 varied messages (some high risk, some low risk)
        messages = [
            "got that fire plug dm me asap, cash only 🔥",
            "what time is the meeting tomorrow?",
            "need a bump for tonight ❄️ HMU",
            "anyone going to the gym later?",
            "fresh batch of molly just landed 💊",
            "can you send me the presentation slides?",
            "looking for some good weed in the area 🍁",
            "happy birthday bro have a good one",
            "who's got the best shrooms rn? 🍄",
            "let's grab lunch around 1pm"
        ]
        self.client.post("/batch_predict", json={
            "messages": messages
        })
