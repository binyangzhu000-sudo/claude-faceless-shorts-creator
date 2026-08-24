#!/usr/bin/env python3
"""
gen_clip.py — AI video clips via fal.ai or the optional Atlas Cloud provider.

Model-agnostic BY DESIGN: the model id is just a string, so swapping models is a --model
flag, never a code change. Extra model-specific inputs pass through with --set key=value.
Saves the mp4 + a sidecar .json (provider, model, payload, request id) for reproducibility.

Usage:
  python tools/gen_clip.py --prompt "..." --out shorts/ch-1-rate-limiting/assets/hook.mp4
  python tools/gen_clip.py --model fal-ai/kling-video/v2.5-turbo/pro/text-to-video \\
      --prompt "..." --set duration=5 --out clip.mp4
  python tools/gen_clip.py --model <image-to-video-model> --prompt "..." \\
      --set image_url=https://... --out clip.mp4          # reference/first-frame models
  python tools/gen_clip.py --provider atlas --prompt "..." \\
      --set image=https://... --out clip.mp4

  --provider  fal (default) or atlas
  --model     provider model id (provider-specific default; check the live model catalog)
  --prompt    text prompt (sent as "prompt")
  --aspect    aspect_ratio (default 9:16)
  --set k=v   any extra payload field, repeatable (numbers/bools auto-parsed; JSON accepted)
  --timeout   seconds to wait (default 900)
  --dry-run   print the payload, no API call

Needs FAL_KEY or ATLASCLOUD_API_KEY in .env. Check the selected provider's current price
before generating; state the cost when proposing a clip in a video plan.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = "fal-ai/veo3.1/fast"
QUEUE = "https://queue.fal.run"
ATLAS_DEFAULT_MODEL = "bytedance/seedance-v1.5-pro/image-to-video"
ATLAS_API = "https://api.atlascloud.ai/api/v1"
ATLAS_GET_ATTEMPTS = 3


class AtlasApiError(RuntimeError):
    pass


def load_env():
    env = {}
    p = os.path.join(ROOT, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return {**env, **os.environ}


def get_arg(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def parse_val(v):
    try:
        return json.loads(v)
    except (ValueError, json.JSONDecodeError):
        return v


def req_json(url, key, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Authorization": f"Key {key}",
                                        "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        sys.exit(f"fal API error {e.code} at {url}:\n{e.read().decode()[:800]}")


def atlas_payload(payload):
    """Translate the established fal aliases to the default Atlas model schema."""
    payload = dict(payload)
    for old, new in (("image_url", "image"), ("end_image_url", "last_image")):
        value = payload.pop(old, None)
        if value is not None:
            payload.setdefault(new, value)
    duration = payload.get("duration")
    if isinstance(duration, str) and duration.isdigit():
        payload["duration"] = int(duration)
    return payload


def atlas_json(url, key, body=None, sleep=time.sleep):
    """Submit once; only result GETs receive bounded transient retries."""
    method = "POST" if body is not None else "GET"
    attempts = 1 if method == "POST" else ATLAS_GET_ATTEMPTS
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "User-Agent": "claude-faceless-shorts-creator/atlas"},
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:800]
            retryable = method == "GET" and (error.code == 429 or error.code >= 500)
            if retryable and attempt + 1 < attempts:
                sleep(2 ** attempt)
                continue
            raise AtlasApiError(f"Atlas API error {error.code} at {url}:\n{detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            if method == "GET" and attempt + 1 < attempts:
                sleep(2 ** attempt)
                continue
            raise AtlasApiError(f"Atlas API request failed at {url}: {error}") from error
    raise AtlasApiError(f"Atlas API request failed at {url}")


def run_atlas(model, payload, key, timeout):
    submission = atlas_json(f"{ATLAS_API}/model/generateVideo", key,
                            body={"model": model, **payload})
    submission_data = submission.get("data", submission)
    prediction_id = submission_data.get("id")
    if not prediction_id:
        raise AtlasApiError("Atlas submission returned no prediction id")

    result_url = f"{ATLAS_API}/model/result/{prediction_id}"
    print(f"queued: {prediction_id}")
    started = time.time()
    last = ""
    while True:
        result = atlas_json(result_url, key)
        result_data = result.get("data", result)
        status = str(result_data.get("status", "?")).lower()
        if status != last:
            print(f"  {status}  (+{int(time.time()-started)}s)")
            last = status
        if status in ("completed", "succeeded"):
            outputs = result_data.get("outputs") or []
            url = next((item for item in outputs if isinstance(item, str)), None)
            url = url or find_video_url(result_data)
            if not url:
                raise AtlasApiError("Atlas completed without a video URL")
            return prediction_id, url
        if status in ("failed", "error", "cancelled", "canceled"):
            raise AtlasApiError(f"Atlas generation {status}: {json.dumps(result_data)[:800]}")
        if time.time() - started > timeout:
            raise AtlasApiError(
                f"timeout after {int(timeout)}s (prediction {prediction_id} may still finish; "
                f"re-poll {result_url})"
            )
        time.sleep(5)


def find_video_url(obj):
    """Walk any response schema for the first video-looking url."""
    if isinstance(obj, dict):
        u = obj.get("url")
        if isinstance(u, str) and (".mp4" in u or "video" in obj.get("content_type", "")):
            return u
        for v in obj.values():
            found = find_video_url(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_video_url(v)
            if found:
                return found
    return None


def main():
    args = sys.argv[1:]
    provider = get_arg(args, "--provider", "fal").lower()
    if provider not in ("fal", "atlas"):
        sys.exit("--provider must be fal or atlas")
    default_model = ATLAS_DEFAULT_MODEL if provider == "atlas" else DEFAULT_MODEL
    model = get_arg(args, "--model", default_model)
    prompt = get_arg(args, "--prompt")
    out = get_arg(args, "--out")
    aspect = get_arg(args, "--aspect", "9:16")
    timeout = float(get_arg(args, "--timeout", "900"))
    dry = "--dry-run" in args

    if not prompt or not out:
        sys.exit("need --prompt and --out (see file header)")

    payload = {"prompt": prompt, "aspect_ratio": aspect}
    for i, a in enumerate(args):
        if a == "--set":
            k, _, v = args[i + 1].partition("=")
            payload[k] = parse_val(v)
    if provider == "atlas":
        payload = atlas_payload(payload)

    print(f"provider = {provider}")
    print(f"model = {model}")
    print("payload =", json.dumps(payload, indent=2)[:600])
    if dry:
        print("[dry-run] no API call.")
        return

    key_name = "ATLASCLOUD_API_KEY" if provider == "atlas" else "FAL_KEY"
    key = load_env().get(key_name, "").strip()
    if not key:
        sys.exit(f"{key_name} not set in .env")

    if provider == "atlas":
        try:
            request_id, url = run_atlas(model, payload, key, timeout)
        except AtlasApiError as error:
            sys.exit(str(error))
    else:
        sub = req_json(f"{QUEUE}/{model}", key, body=payload)
        status_url = sub.get("status_url") or f"{QUEUE}/{model}/requests/{sub['request_id']}/status"
        response_url = sub.get("response_url") or f"{QUEUE}/{model}/requests/{sub['request_id']}"
        request_id = sub.get("request_id")
        print(f"queued: {request_id}")

        t0 = time.time()
        last = ""
        while True:
            st = req_json(f"{status_url}?logs=1", key)
            s = st.get("status", "?")
            if s != last:
                print(f"  {s}  (+{int(time.time()-t0)}s)")
                last = s
            if s == "COMPLETED":
                break
            if s in ("FAILED", "ERROR", "CANCELLED"):
                sys.exit(f"generation {s}: {json.dumps(st)[:800]}")
            if time.time() - t0 > timeout:
                sys.exit(f"timeout after {int(timeout)}s (request {request_id} may still finish; "
                         f"re-poll {response_url})")
            time.sleep(5)

        result = req_json(response_url, key)
        url = find_video_url(result)
        if not url:
            sys.exit("no video url in response:\n" + json.dumps(result)[:800])

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    urllib.request.urlretrieve(url, out)
    print(f"video -> {os.path.relpath(out, ROOT)}  ({os.path.getsize(out)//1024}KB)")

    sidecar = os.path.splitext(out)[0] + ".json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump({"provider": provider, "model": model, "payload": payload,
                   "request_id": request_id,
                   "source_url": url, "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"meta  -> {os.path.relpath(sidecar, ROOT)}")


if __name__ == "__main__":
    main()
