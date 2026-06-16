import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import time
import json
import random
import threading
import urllib.request

SERVER_URL  = "http://localhost:8000"
TOTAL_TIME  = 30
NUM_THREADS = 8
BATCH_SIZE  = 10

SAMPLES = [
    ("got that fire plug dm me asap cash only",       "HIGH"),
    ("hey what time is the meeting tomorrow?",         "SAFE"),
    ("need a bump for tonight dm for details",         "HIGH"),
    ("can you send the project presentation slides?",  "SAFE"),
    ("fresh batch of molly just landed best quality",  "HIGH"),
    ("going to gym later anyone want to join?",        "SAFE"),
    ("looking for weed in the area cash only",         "HIGH"),
    ("happy birthday bro have an amazing day!",        "SAFE"),
    ("who has the best shrooms quick delivery",        "HIGH"),
    ("let us grab lunch around 1pm at the usual",      "SAFE"),
    ("selling 10 bars tonight plug verified",          "HIGH"),
    ("did you finish the report for the client?",      "SAFE"),
    ("pure white girl scale tested no cut",            "HIGH"),
    ("reminder team standup at 9am sharp tomorrow",    "SAFE"),
]

lock            = threading.Lock()
total_requests  = 0
total_messages  = 0
total_errors    = 0
high_risk_found = 0
sample_log      = []
start_time      = None

def post_json(endpoint, data):
    body = json.dumps(data).encode()
    req  = urllib.request.Request(
        f"{SERVER_URL}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())

def worker(tid):
    global total_requests, total_messages, total_errors, high_risk_found
    while True:
        if time.time() - start_time >= TOTAL_TIME:
            break
        batch    = random.choices(SAMPLES, k=BATCH_SIZE)
        messages = [m for m, _ in batch]
        t0 = time.time()
        try:
            result  = post_json("/batch_predict", {"messages": messages})
            lat_ms  = (time.time() - t0) * 1000
            preds   = result.get("results", [])   # API returns "results"
            flagged = sum(1 for p in preds if p.get("risk_level") == "High")
            with lock:
                total_requests  += 1
                total_messages  += len(messages)
                high_risk_found += flagged
                if preds:
                    sample_log.append({
                        "tid": tid, "msg": messages[0][:48],
                        "risk": preds[0].get("risk_level","?"),
                        "score": preds[0].get("prediction", 0),  # "prediction" = probability
                        "lat": lat_ms,
                    })
                    if len(sample_log) > 6:
                        sample_log.pop(0)
        except Exception:
            with lock:
                total_errors += 1

def stats_loop():
    last_reqs, last_msgs, last_t = 0, 0, time.time()
    while True:
        time.sleep(3)
        elapsed = time.time() - start_time
        if elapsed >= TOTAL_TIME + 1:
            break
        with lock:
            reqs, msgs, errs = total_requests, total_messages, total_errors
            flagged = high_risk_found
            snaps   = list(sample_log)
        dt   = max(time.time() - last_t, 0.001)
        rps  = (reqs - last_reqs) / dt
        mps  = (msgs - last_msgs) / dt
        last_reqs, last_msgs, last_t = reqs, msgs, time.time()
        pct = min(elapsed / TOTAL_TIME, 1.0)
        bar = ("#" * int(pct * 26)).ljust(26, ".")
        print(f"\n  [Time: {elapsed:>4.0f}s] [{bar}]")
        print(f"  +--------------------------------------------------+")
        print(f"  | Requests fired      : {reqs:>7,}  ({rps:>6.1f} req/sec)   |")
        print(f"  | Messages processed  : {msgs:>7,}  ({mps:>6.0f} msg/sec)   |")
        print(f"  | High Risk flagged   : {flagged:>7,}  ({flagged/max(msgs,1)*100:>4.1f}% of traffic) |")
        print(f"  | Errors              : {errs:>7,}                     |")
        print(f"  +--------------------------------------------------+")
        if snaps:
            print(f"  -- Sample requests (most recent {len(snaps)}) --")
            for s in snaps:
                tag = "[HIGH RISK]" if s["risk"] == "High" else "[ SAFE    ]"
                print(f"    T-{s['tid']} | {tag} score={s['score']:.2f} | {s['lat']:.0f}ms | \"{s['msg']}\"")

def main():
    global start_time
    print("=" * 58)
    print("  DARK TRACE AI -- LIVE LOAD TEST OBSERVER")
    print(f"  {NUM_THREADS} threads x {BATCH_SIZE} msgs/batch x {TOTAL_TIME}s")
    print("=" * 58)

    try:
        urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
        print("  Server: OK (port 8000) -- 4 Uvicorn workers running")
    except Exception:
        print("  ERROR: Server not running! Start it first.")
        return

    print()
    print("  What you will see:")
    print("    - 8 threads firing simultaneous batch requests")
    print("    - Each batch = 10 msgs (drug talk + innocent chat)")
    print("    - AI returns risk_level (High/Low) + risk_score per msg")
    print("    - Live stats every 3 seconds")
    print()
    print("  Launching in 2 seconds...")
    time.sleep(2)
    print()

    start_time = time.time()
    threads = [threading.Thread(target=worker, args=(i+1,), daemon=True) for i in range(NUM_THREADS)]
    for t in threads: t.start()
    sp = threading.Thread(target=stats_loop, daemon=True)
    sp.start()
    for t in threads: t.join(timeout=TOTAL_TIME + 5)
    sp.join(timeout=5)

    elapsed = time.time() - start_time
    print("\n" + "=" * 58)
    print("  FINAL RESULTS")
    print("=" * 58)
    print(f"  Duration           : {elapsed:.1f}s")
    print(f"  Total Requests     : {total_requests:,}  batches fired")
    print(f"  Total Messages     : {total_messages:,}  individual msgs classified")
    print(f"  Effective Speed    : {total_messages/elapsed:,.0f}  messages / second")
    print(f"  High Risk Flagged  : {high_risk_found:,}  ({high_risk_found/max(total_messages,1)*100:.1f}% of traffic)")
    print(f"  Errors             : {total_errors}")
    print("=" * 58)

if __name__ == "__main__":
    main()
