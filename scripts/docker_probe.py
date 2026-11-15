"""Exercise the running container over HTTP, including feedback across restarts."""

import argparse
import json
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8000"
STATE = Path("/app/data/events/acceptance-state.json")


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    with urlopen(
        Request(BASE + path, data=data, headers={"Content-Type": "application/json"}),
        timeout=30,
    ) as response:
        return json.load(response)


def cards(answer):
    return answer["route"] + answer["more"] + answer["quick"]


def ids(answer):
    return {card["item"]["item_id"] for card in cards(answer)}


def feedback(user, item):
    request(
        "/api/events",
        {"user_id": user, "item_id": item, "type": "known", "ts": "2026-11-16T00:00:00Z"},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["smoke", "seed", "restart"])
    parser.add_argument("--encoder", choices=["dense", "tfidf"], required=True)
    args = parser.parse_args()
    for attempt in range(90):
        try:
            health = request("/api/health")
            break
        except (URLError, TimeoutError):
            if attempt == 89:
                raise
            time.sleep(1)
    assert (health["status"], health["recommender"], health["encoder"]) == (
        "ok",
        "ranking",
        args.encoder,
    ), health
    assert health["catalog_items"] == 1497, health
    for path, marker in [("/", b"/static/app.js"), ("/static/app.js", b"fetch")]:
        with urlopen(BASE + path, timeout=10) as response:
            assert marker in response.read(), path
    cold = request("/api/recommendations", {"profile": {}, "limit": 6})
    assert len(cold["more"]) == 6 and cold["quick"]
    profile = {"tasks": ["marketing"], "industries": ["services"]}
    body = {"profile": profile, "user_id": "docker-acceptance", "limit": 6}
    answer = request("/api/recommendations", body)
    assert len(answer["more"]) == 6 and answer["route"]
    assert all(card["reasons"] and card["next_step"]["url"] for card in cards(answer))
    assert request("/api/recommendations", body | {"mode": "baseline"})["mode"] == "baseline"
    personas = request("/api/personas")
    assert len(personas) == 8
    for persona in personas:
        result = request(
            "/api/recommendations",
            {"profile": persona["profile"], "user_id": persona["user_id"], "limit": 6},
        )
        assert result["more"], persona["id"]
        excluded = {
            e["item_id"]
            for e in persona["history"]
            if e["type"] in {"complete", "known", "dislike"}
        }
        assert not ids(result) & excluded, persona["id"]
    if args.phase == "seed":
        item = answer["more"][0]["item"]["item_id"]
        feedback(body["user_id"], item)
        assert item not in ids(request("/api/recommendations", body))
        persona = personas[0]
        demo_body = {"profile": persona["profile"], "user_id": persona["user_id"], "limit": 6}
        demo_item = request("/api/recommendations", demo_body)["more"][0]["item"]["item_id"]
        feedback(persona["user_id"], demo_item)
        assert demo_item not in ids(request("/api/recommendations", demo_body))
        STATE.write_text(json.dumps({"item": item, "demo_item": demo_item, "demo_body": demo_body}))
    elif args.phase == "restart":
        state = json.loads(STATE.read_text())
        assert state["item"] not in ids(answer), "Ordinary feedback was lost"
        assert state["demo_item"] in ids(request("/api/recommendations", state["demo_body"])), (
            "Demo feedback survived restart"
        )
        events = Path("/app/data/events/events.jsonl").read_text().splitlines()
        assert all(json.loads(line)["user_id"] != state["demo_body"]["user_id"] for line in events)
    print(
        json.dumps(
            {"phase": args.phase, "health": health, "personas": len(personas), "result": "passed"}
        )
    )


if __name__ == "__main__":
    main()
