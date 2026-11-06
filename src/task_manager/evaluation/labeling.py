"""A self-contained page for labelling the sample by keyboard.

It holds real, if anonymized, correspondence, so it makes no network requests: the content security
policy forbids them. Message text is inserted only through textContent, which keeps <ME> visible and
stops anything in a message from running as code.
"""

import hashlib
import json
from pathlib import Path

from task_manager.evaluation.sampling import Sample

LABELING_FILE = "labeling.html"
LABELS_FILE = "labels.json"
ALL_LABELS = (
    (1, "Задача мне"),
    (2, "Возможно мне"),
    (3, "Задача, но не мне"),
    (4, "Не задача"),
    (5, "Часть задачи"),
)
ALL_LABELS_HINT = "«Часть задачи»: само сообщение не задача, но обсуждение ведёт к задаче."
OWN_LABELS = ((1, "Задача мне: я взял работу на себя"), (4, "Не задача"))
OWN_LABELS_HINT = "Размечаются ваши сообщения: задача, только если вы пообещали сделать работу."

_PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>Разметка сообщений</title>
<style>
  :root { color-scheme: light dark; --bg: #f6f6f4; --card: #fff; --text: #1c1c1a; --muted: #6b6b66;
          --line: #d8d8d2; --accent: #2457c5; --chosen: #dbe6ff; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #171716; --card: #222220; --text: #ecece8; --muted: #a3a39c; --line: #3a3a36;
            --accent: #8db0ff; --chosen: #2a3a63; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 16px/1.5 system-ui, sans-serif; }
  main { max-width: 860px; margin: 0 auto; padding: 16px; }
  header { display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
           justify-content: space-between; margin-bottom: 12px; }
  h1 { font-size: 18px; margin: 0; }
  .muted { color: var(--muted); font-size: 14px; }
  .line { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
          padding: 8px 12px; margin: 6px 0; }
  .line.target { border: 2px solid var(--accent); padding: 12px; }
  .meta { color: var(--muted); font-size: 13px; }
  .text { white-space: pre-wrap; overflow-wrap: anywhere; }
  h2 { font-size: 14px; color: var(--muted); margin: 14px 0 4px; font-weight: 600; }
  .buttons { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 8px;
             margin: 14px 0; }
  button { font: inherit; color: inherit; background: var(--card); border: 1px solid var(--line);
           border-radius: 8px; padding: 10px; cursor: pointer; text-align: left; }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  button.chosen { background: var(--chosen); border-color: var(--accent); }
  kbd { font-weight: 700; margin-right: 6px; }
  .nav { display: flex; flex-wrap: wrap; gap: 8px; }
  .nav button { padding: 6px 12px; }
  progress { width: 100%; height: 8px; }
</style>
</head>
<body>
<main>
  <header>
    <h1>Разметка сообщений</h1>
    <span class="muted" id="progress-text"></span>
  </header>
  <progress id="progress"></progress>
  <div id="context"></div>
  <div class="buttons" id="buttons"></div>
  <div class="nav">
    <button id="back">← Назад</button>
    <button id="forward">Вперёд →</button>
    <button id="next-empty">К неразмеченному</button>
    <button id="save">Скачать labels.json</button>
  </div>
  <p class="muted">Цифры ставят метку и переходят дальше. Стрелка вправо оставляет метку как есть.
  <span id="hint"></span> Прогресс хранится в этом браузере, но надёжная копия только в скачанном
  файле: скачивайте его время от времени и в конце.</p>
</main>
<script type="application/json" id="sample">__DATA__</script>
<script>
"use strict";
const data = JSON.parse(document.getElementById("sample").textContent);
const LABELS = data.choices;
document.getElementById("hint").textContent = data.hint;
// A new page with corrected labels must not be overridden by what an older one left behind.
const STORAGE_KEY = "task-manager-labels-" + data.seed + "-" + data.revision;
const POSITION_KEY = STORAGE_KEY + "-position";
const labels = load();
let position = startPosition();

function load() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || { ...data.labels };
  } catch {
    return { ...data.labels };
  }
}
function store() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
    localStorage.setItem(POSITION_KEY, String(position));
  } catch {}
}
function firstEmpty() {
  return Math.max(0, data.examples.findIndex(e => !(e.id in labels)));
}
function startPosition() {
  try {
    const saved = Number(localStorage.getItem(POSITION_KEY));
    if (saved > 0 && saved < data.examples.length) return saved;
  } catch {}
  return firstEmpty();
}
function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function renderLine(line, isTarget) {
  const box = make("div", isTarget ? "line target" : "line");
  const reply = line.reply_to === null ? "" : " · ответ на " + line.reply_to;
  box.append(make("div", "meta",
    "[" + line.msg_id + "] " + line.sent_at.replace("T", " ").slice(0, 16) + " · " +
    (line.sender || "неизвестно") + reply));
  box.append(make("div", "text", line.text || "(без текста)"));
  if (line.files.length) box.append(make("div", "meta", "Файлы: " + line.files.join(", ")));
  return box;
}
function renderBlock(root, title, lines) {
  if (!lines.length) return;
  root.append(make("h2", "", title));
  lines.forEach(line => root.append(renderLine(line, false)));
}
function render() {
  const example = data.examples[position];
  const root = document.getElementById("context");
  root.replaceChildren();
  renderBlock(root, "Цепочка ответов", example.reply_chain);
  renderBlock(root, "Предыдущие сообщения", example.previous);
  root.append(make("h2", "", "Размечаемое сообщение · " + example.chat_kind +
    (example.mentions_me ? " · есть упоминание" : "")));
  root.append(renderLine(example.target, true));

  const buttons = document.getElementById("buttons");
  buttons.replaceChildren();
  LABELS.forEach(([value, name]) => {
    const button = make("button", labels[example.id] === value ? "chosen" : "");
    button.append(make("kbd", "", String(value)), document.createTextNode(name));
    button.addEventListener("click", () => choose(value));
    buttons.append(button);
  });

  const done = Object.keys(labels).length;
  const bar = document.getElementById("progress");
  bar.max = data.examples.length;
  bar.value = done;
  document.getElementById("progress-text").textContent =
    "Сообщение " + (position + 1) + " из " + data.examples.length + " · размечено " + done;
  window.scrollTo(0, 0);
}
function move(step) {
  position = Math.min(data.examples.length - 1, Math.max(0, position + step));
  store();
  render();
}
function choose(value) {
  labels[data.examples[position].id] = value;
  if (position < data.examples.length - 1) position += 1;
  store();
  render();
}
function save() {
  const blob = new Blob([JSON.stringify(labels, null, 1)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "labels.json";
  link.click();
  URL.revokeObjectURL(link.href);
}
document.getElementById("back").addEventListener("click", () => move(-1));
document.getElementById("forward").addEventListener("click", () => move(1));
document.getElementById("next-empty").addEventListener("click", () => {
  position = firstEmpty();
  render();
});
document.getElementById("save").addEventListener("click", save);
document.addEventListener("keydown", event => {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if (LABELS.some(([value]) => String(value) === event.key)) choose(Number(event.key));
  else if (event.key === "ArrowLeft") move(-1);
  else if (event.key === "ArrowRight") move(1);
});
render();
</script>
</body>
</html>
"""


def _embed(value: object) -> str:
    """JSON that can't end its script element, whatever a message contains."""
    text = json.dumps(value, ensure_ascii=False)
    for char, escape in (
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("&", "\\u0026"),
        (" ", "\\u2028"),
        (" ", "\\u2029"),
    ):
        text = text.replace(char, escape)
    return text


def render_labeling_page(
    sample: Sample,
    labels: dict[int, int] | None = None,
    choices: tuple[tuple[int, str], ...] = ALL_LABELS,
    hint: str = ALL_LABELS_HINT,
) -> str:
    """Signs, groups and model answers are left out: the owner judges only the message."""
    ids = {example.message_id for example in sample.examples}
    current = {str(key): value for key, value in sorted((labels or {}).items()) if key in ids}
    revision = hashlib.sha256(json.dumps(current).encode()).hexdigest()[:8]
    payload = {
        "seed": sample.seed,
        "revision": revision,
        "labels": current,
        "choices": choices,
        "hint": hint,
        "examples": [
            {
                "id": str(example.message_id),
                "chat_kind": example.thread.chat_kind.value,
                "mentions_me": example.thread.mentions_me,
                "target": example.thread.target.model_dump(),
                "reply_chain": [line.model_dump() for line in example.thread.reply_chain],
                "previous": [line.model_dump() for line in example.thread.previous],
            }
            for example in sample.examples
        ],
    }
    return _PAGE.replace("__DATA__", _embed(payload))


def write_labeling_page(
    sample: Sample,
    directory: Path,
    labels: dict[int, int] | None = None,
    choices: tuple[tuple[int, str], ...] = ALL_LABELS,
    hint: str = ALL_LABELS_HINT,
) -> Path:
    path = directory / LABELING_FILE
    path.write_text(render_labeling_page(sample, labels, choices, hint), encoding="utf-8")
    return path


def read_labels(path: Path) -> dict[int, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    labels = {int(key): int(value) for key, value in raw.items()}
    invalid = sorted(set(labels.values()) - {1, 2, 3, 4, 5})
    if invalid:
        raise ValueError(f"unknown labels: {invalid}")
    return labels
