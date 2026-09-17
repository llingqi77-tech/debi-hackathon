#!/usr/bin/env python3
"""
《她的花向上开》24镜 MiniMax-H3 批量视频生成脚本

基于 API 文档：
  创建任务: POST https://api.openai-next.com/hailuo/v2/video_generation
  查询任务: GET  https://api.openai-next.com/hailuo/v2/query/video_generation/{task_id}

功能：
  - 解析 24镜图生视频提示词.md，提取每镜首帧/尾帧/台词/声音/运镜描述
  - 自动拼接全局固定提示词
  - 图片自动压缩（PNG→JPEG），避免请求体过大导致断连
  - 所有 HTTP 请求通过 curl 发送，绕过 Cloudflare 反爬
  - 自动轮询任务状态，成功后下载视频
  - 断点续跑：已成功的镜头自动跳过
  - 支持 --query-task-id 单独查询已有任务

用法：
  export MINIMAX_API_KEY="sk-你的完整密钥"

  # 预览第1镜请求（不实际调用）
  python3 generate_videos.py --dry-run --start 1 --end 1

  # 生成第1镜
  python3 generate_videos.py --start 1 --end 1

  # 生成全部24镜
  python3 generate_videos.py

  # 查询一个已创建的任务状态（比如之前提交的任务）
  python3 generate_videos.py --query-task-id 442785878196456
"""

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ═══════════════════════════════════════════════════════
#  配置（对应 API 文档参数）
# ═══════════════════════════════════════════════════════
API_BASE = "https://api.openai-next.com/hailuo/v2"
CREATE_URL = f"{API_BASE}/video_generation"
QUERY_URL = f"{API_BASE}/query/video_generation"

MODEL = "MiniMax-H3"       # 文档: MiniMax-H3 或 MiniMax-H3-Max
RESOLUTION = "2K"          # 文档: H3 支持 768P / 2K
DURATION = 5                # 文档: H3 支持 4-15 秒
RATIO = "16:9"              # 首尾帧模式下 API 自动适配，传入不影响

# 轮询配置
POLL_INTERVAL = 10          # 轮询间隔（秒），文档推荐 10 秒
POLL_TIMEOUT = 900          # 单镜最大等待（秒），2K 视频可能较慢

# 重试配置
MAX_RETRY = 3               # 单镜失败最大重试次数
REQUEST_TIMEOUT = 180       # curl 最大超时（秒），大图上传需较长

# 图片压缩（PNG→JPEG，减小请求体避免 Broken pipe）
IMAGE_MAX_DIM = 1672        # 最大边长，0=不缩放
IMAGE_JPEG_QUALITY = 85     # JPEG 质量 1-95

# 请求头（模拟浏览器 + 原始 curl 的 Cookie）
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/128.0.0.0 Safari/537.36",
    "Cookie": "SITE_TOTAL_ID=062daceeb0a4e716d647fa9494d7d1ef",
}

# 路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = SCRIPT_DIR / "24镜图生视频提示词.md"
OUTPUT_DIR = SCRIPT_DIR / "output"
STATE_FILE = SCRIPT_DIR / "generation_state.json"


# ═══════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════

def load_api_key():
    key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if not key:
        print("错误：未设置环境变量 MINIMAX_API_KEY")
        print('请先执行：export MINIMAX_API_KEY="sk-你的完整密钥"')
        sys.exit(1)
    return key


def _curl_post(url, payload, api_key, timeout=REQUEST_TIMEOUT):
    """用 curl 发 POST（绕过 Cloudflare 对 Python urllib 的拦截）"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        payload_path = f.name
    try:
        cmd = [
            "curl", "--location", "--silent", "--show-error",
            "--max-time", str(timeout),
            url,
            "-H", f"Authorization: Bearer {api_key}",
            "-H", "Content-Type: application/json",
            "-H", f"User-Agent: {REQUEST_HEADERS['User-Agent']}",
            "-H", f"Cookie: {REQUEST_HEADERS['Cookie']}",
            "--data", f"@{payload_path}",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 15)
        if r.returncode != 0:
            raise RuntimeError(f"curl 退出码 {r.returncode}: {r.stderr.strip()}")
        body = r.stdout.strip()
        if not body:
            raise RuntimeError("curl 返回空响应")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            preview = body[:300].replace("\n", " ")
            raise RuntimeError(f"响应不是有效 JSON（前300字符）: {preview}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"curl 请求超时（{timeout}s）")
    finally:
        os.unlink(payload_path)


def _curl_get(url, api_key, timeout=30):
    """用 curl 发 GET"""
    cmd = [
        "curl", "--location", "--silent", "--show-error",
        "--max-time", str(timeout),
        url,
        "-H", f"Authorization: Bearer {api_key}",
        "-H", f"User-Agent: {REQUEST_HEADERS['User-Agent']}",
        "-H", f"Cookie: {REQUEST_HEADERS['Cookie']}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    if r.returncode != 0:
        raise RuntimeError(f"curl 退出码 {r.returncode}: {r.stderr.strip()}")
    body = r.stdout.strip()
    if not body:
        raise RuntimeError("curl 返回空响应")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        preview = body[:300].replace("\n", " ")
        raise RuntimeError(f"响应不是有效 JSON（前300字符）: {preview}")


def _curl_download(url, output_path, timeout=180):
    """用 curl 下载文件"""
    cmd = [
        "curl", "--location", "--silent", "--show-error",
        "--max-time", str(timeout),
        "-H", f"User-Agent: {REQUEST_HEADERS['User-Agent']}",
        "-o", str(output_path),
        url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 15)
    if r.returncode != 0:
        raise RuntimeError(f"下载失败 curl 退出码 {r.returncode}: {r.stderr.strip()}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("下载文件为空")


# ═══════════════════════════════════════════════════════
#  提示词解析
# ═══════════════════════════════════════════════════════

def parse_prompts(md_path):
    """解析提示词 markdown，返回 (global_prompt, shots_list)"""
    text = md_path.read_text(encoding="utf-8")

    # 全局固定提示词
    m = re.search(r"## 全局固定提示词.*?```text\s*\n(.*?)```", text, re.DOTALL)
    global_prompt = m.group(1).strip() if m else ""

    # 各镜头
    shot_pattern = re.compile(
        r"### 镜头(\d+)｜[^\n]*\n(.*?)(?=\n### 镜头|\n## |\Z)", re.DOTALL
    )
    shots = []
    for m in shot_pattern.finditer(text):
        num = int(m.group(1))
        body = m.group(2)

        first = re.search(r"首帧[：:]\s*`([^`]+)`", body)
        last = re.search(r"尾帧[：:]\s*`([^`]+)`", body)
        dlg = re.search(r"台词[^：:\n]*[：:]\s*(.+?)(?:\n- |\n```|\Z)", body, re.DOTALL)
        snd = re.search(r"声音[：:]\s*(.+?)(?:\n- |\n```|\Z)", body, re.DOTALL)
        prm = re.search(r"```text\s*\n(.*?)```", body, re.DOTALL)

        dialogue = re.sub(r"\s+", " ", dlg.group(1).strip()) if dlg else "无"
        sound = re.sub(r"\s+", " ", snd.group(1).strip()) if snd else "无"

        shots.append({
            "num": num,
            "first_frame": first.group(1) if first else "",
            "last_frame": last.group(1) if last else "",
            "dialogue": dialogue,
            "sound": sound,
            "main_prompt": prm.group(1).strip() if prm else "",
        })

    shots.sort(key=lambda s: s["num"])
    return global_prompt, shots


def build_full_prompt(shot, global_prompt):
    """拼接完整提示词：运镜 + 台词参考 + 声音参考 + 全局固定词"""
    parts = [shot["main_prompt"]]
    if shot["dialogue"] and shot["dialogue"] != "无":
        parts.append(f"台词参考（仅做自然轻微启唇，不生成语音）：{shot['dialogue']}")
    if shot["sound"] and shot["sound"] != "无":
        parts.append(f"环境声音参考：{shot['sound']}")
    if global_prompt:
        parts.append(global_prompt)
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════
#  图片处理
# ═══════════════════════════════════════════════════════

def encode_image(image_path):
    """压缩图片为 JPEG 后转 base64 data URI"""
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    try:
        from PIL import Image
        img = Image.open(image_path)
        # 处理 alpha 通道
        if img.mode in ("RGBA", "P", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            mask = img.split()[-1] if img.mode in ("RGBA", "LA") else None
            bg.paste(img, mask=mask)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        # 缩放
        if IMAGE_MAX_DIM > 0:
            w, h = img.size
            if max(w, h) > IMAGE_MAX_DIM:
                ratio = IMAGE_MAX_DIM / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        # 编码 JPEG
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=IMAGE_JPEG_QUALITY, optimize=True)
        raw = buf.getvalue()
        orig_kb = image_path.stat().st_size / 1024
        new_kb = len(raw) / 1024
        print(f"  图片压缩: {image_path.name} {orig_kb:.0f}KB → {new_kb:.0f}KB JPEG")
    except ImportError:
        print(f"  警告: 未安装 Pillow，使用原始图片（pip3 install Pillow）")
        raw = image_path.read_bytes()

    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ═══════════════════════════════════════════════════════
#  API 调用
# ═══════════════════════════════════════════════════════

def create_task(api_key, prompt, first_b64, last_b64, dry_run=False):
    """创建视频生成任务，返回 task_id
    对应文档: POST /hailuo/v2/video_generation
    """
    payload = {
        "model": MODEL,
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": first_b64}, "role": "first_frame"},
            {"type": "image_url", "image_url": {"url": last_b64}, "role": "last_frame"},
        ],
        "resolution": RESOLUTION,
        "duration": DURATION,
        "ratio": RATIO,
    }

    if dry_run:
        print(f"  [DRY-RUN] POST {CREATE_URL}")
        print(f"  [DRY-RUN] model={MODEL}, resolution={RESOLUTION}, "
              f"duration={DURATION}s, ratio={RATIO}")
        print(f"  [DRY-RUN] prompt 长度: {len(prompt)} 字符")
        print(f"  [DRY-RUN] 首帧 base64: {len(first_b64)} 字符")
        print(f"  [DRY-RUN] 尾帧 base64: {len(last_b64)} 字符")
        return "dry-run-task-id"

    print(f"  提交请求: POST {CREATE_URL}")
    result = _curl_post(CREATE_URL, payload, api_key)

    # 检查 API 错误
    if result.get("type") == "error" or "error" in result:
        err = result.get("error", result)
        raise RuntimeError(f"API 返回错误: {json.dumps(err, ensure_ascii=False)}")

    task_id = result.get("task_id")
    if not task_id:
        raise RuntimeError(f"响应中无 task_id: {json.dumps(result, ensure_ascii=False)}")
    print(f"  task_id: {task_id}")
    return task_id


def query_task(api_key, task_id):
    """查询任务状态，返回任务详情 dict
    对应: GET /hailuo/v2/query/video_generation/{task_id}
    """
    url = f"{QUERY_URL}/{task_id}"
    result = _curl_get(url, api_key)
    # 响应可能是 {"task": {...}} 或直接 {...}
    return result.get("task", result)


def poll_task(api_key, task_id):
    """轮询直到任务成功或失败，返回任务详情"""
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > POLL_TIMEOUT:
            raise TimeoutError(f"任务 {task_id} 轮询超时（{POLL_TIMEOUT}s）")

        task = query_task(api_key, task_id)
        status = task.get("status", "").lower()

        if status in ("succeeded", "success", "completed", "done"):
            return task
        if status in ("failed", "cancelled", "canceled", "error"):
            raise RuntimeError(f"任务 {task_id} 失败: "
                               f"{json.dumps(task, ensure_ascii=False)}")
        # queued / queueing / preparing / processing / running / pending
        print(f"  状态: {task.get('status', status)}，已等待 {int(elapsed)}s...")
        time.sleep(POLL_INTERVAL)


def extract_video_url(task):
    """从任务详情中提取视频 URL"""
    content = task.get("content", {})
    if isinstance(content, dict):
        url = content.get("url", "")
        if url:
            return url
    # 尝试其他可能的字段
    for key in ("video_url", "url", "output_url", "file_url"):
        if task.get(key):
            return task[key]
    raise RuntimeError(f"任务成功但未找到视频URL: {json.dumps(task, ensure_ascii=False)}")


def download_video(video_url, output_path):
    """下载视频到本地"""
    print(f"  下载视频: {output_path.name}")
    _curl_download(video_url, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  下载完成: {size_mb:.1f} MB")


# ═══════════════════════════════════════════════════════
#  状态管理
# ═══════════════════════════════════════════════════════

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ═══════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════

def process_shot(api_key, shot, global_prompt, state, dry_run=False, force=False):
    """处理单个镜头，返回 True/False"""
    num = shot["num"]
    key = f"shot_{num:02d}"
    output_path = OUTPUT_DIR / f"镜头{num:02d}.mp4"

    # 跳过已完成
    if not force and state.get(key, {}).get("status") == "succeeded" and output_path.exists():
        print(f"[镜头{num:02d}] 已生成，跳过（--force 可重新生成）")
        return True

    print(f"{'='*60}")
    print(f"[镜头{num:02d}] 开始处理")
    print(f"  首帧: {shot['first_frame']}")
    print(f"  尾帧: {shot['last_frame']}")
    print(f"  台词: {shot['dialogue'][:50]}")
    print(f"  声音: {shot['sound'][:50]}")

    try:
        first_b64 = encode_image(SCRIPT_DIR / shot["first_frame"])
        last_b64 = encode_image(SCRIPT_DIR / shot["last_frame"])
        full_prompt = build_full_prompt(shot, global_prompt)
    except FileNotFoundError as e:
        print(f"  错误: {e}")
        state[key] = {"status": "failed", "error": str(e)}
        save_state(state)
        return False

    # 重试循环
    for attempt in range(1, MAX_RETRY + 1):
        try:
            print(f"  创建任务（第 {attempt}/{MAX_RETRY} 次）...")
            task_id = create_task(api_key, full_prompt, first_b64, last_b64, dry_run)

            if dry_run:
                state[key] = {"status": "dry-run", "task_id": task_id}
                save_state(state)
                return True

            # 轮询
            task = poll_task(api_key, task_id)
            video_url = extract_video_url(task)

            # 下载
            download_video(video_url, output_path)

            state[key] = {
                "status": "succeeded",
                "task_id": task_id,
                "video_url": video_url,
                "output_file": str(output_path),
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_state(state)
            print(f"[镜头{num:02d}] 完成 ✓")
            return True

        except (RuntimeError, TimeoutError, OSError) as e:
            print(f"  错误: {e}")
            if attempt < MAX_RETRY:
                wait = 5 * attempt
                print(f"  {wait}s 后重试...")
                time.sleep(wait)
            else:
                state[key] = {
                    "status": "failed",
                    "error": str(e),
                    "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_state(state)
                print(f"[镜头{num:02d}] 失败 ✗（已重试 {MAX_RETRY} 次）")
                return False

    return False


def cmd_query_task_id(api_key, task_id):
    """单独查询一个任务状态"""
    print(f"查询任务: {task_id}")
    try:
        task = query_task(api_key, task_id)
        print(json.dumps(task, ensure_ascii=False, indent=2))
        status = task.get("status", "").lower()
        if status in ("succeeded", "success", "completed"):
            url = extract_video_url(task)
            print(f"\n视频URL: {url}")
    except RuntimeError as e:
        print(f"查询失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="MiniMax-H3 批量视频生成")
    parser.add_argument("--start", type=int, default=1, help="起始镜头（默认1）")
    parser.add_argument("--end", type=int, default=24, help="结束镜头（默认24）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不调用API")
    parser.add_argument("--force", action="store_true", help="强制重新生成已完成镜头")
    parser.add_argument("--query-task-id", type=str, default=None,
                        help="单独查询指定 task_id 的状态（不执行生成）")
    args = parser.parse_args()

    api_key = load_api_key()

    # 单独查询模式
    if args.query_task_id:
        cmd_query_task_id(api_key, args.query_task_id)
        return

    # 解析提示词
    print(f"解析提示词: {PROMPTS_FILE.name}")
    global_prompt, shots = parse_prompts(PROMPTS_FILE)
    print(f"共 {len(shots)} 镜，全局提示词 {len(global_prompt)} 字符")

    shots = [s for s in shots if args.start <= s["num"] <= args.end]
    if not shots:
        print(f"范围 {args.start}-{args.end} 内无镜头")
        sys.exit(1)
    print(f"本次处理: 镜头{shots[0]['num']:02d} - {shots[-1]['num']:02d}（{len(shots)}镜）\n")

    OUTPUT_DIR.mkdir(exist_ok=True)
    state = load_state()

    success = 0
    failed = 0
    for shot in shots:
        if process_shot(api_key, shot, global_prompt, state, args.dry_run, args.force):
            success += 1
        else:
            failed += 1
        print()

    print(f"{'='*60}")
    print(f"完成！成功: {success}，失败: {failed}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"状态文件: {STATE_FILE}")
    if failed > 0:
        print(f"\n失败镜头可重跑（自动跳过已成功）：")
        print(f"  python3 {Path(__file__).name} --start {args.start} --end {args.end}")


if __name__ == "__main__":
    main()
