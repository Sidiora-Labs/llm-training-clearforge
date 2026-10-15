"use strict";
const $ = (id) => document.getElementById(id);
let history = [],
  controller = null,
  modelInfo = [];
const welcome = $("welcome").cloneNode(true);
const headers = () => ({
  "Content-Type": "application/json",
  ...($("key").value ? { Authorization: `Bearer ${$("key").value}` } : {}),
});
const error = (text) => {
  $("error").textContent = text;
  $("error").hidden = !text;
};
function busy(active) {
  $("send").disabled = active || !$("model").value;
  for (const id of ["model", "system", "reset", "connect"])
    $(id).disabled = active;
  $("stop").hidden = !active;
  $("regenerate").disabled = active || !history.some((m) => m.role === "user");
  $("status").textContent = active
    ? "Generating locally…"
    : $("model").value
      ? "Ready · Local model"
      : "No model connected";
}
function bubble(role, content) {
  $("welcome")?.remove();
  const el = document.createElement("article");
  el.className = `message ${role}`;
  const name = document.createElement("span");
  name.className = "name";
  name.textContent = role === "user" ? "You" : "Clearforge";
  const body = document.createElement("div");
  body.textContent = content;
  el.append(name, body);
  $("messages").append(el);
  return body;
}
function render() {
  $("messages").replaceChildren();
  if (!history.length) $("messages").append(welcome.cloneNode(true));
  for (const m of history) bubble(m.role, m.content);
}
function scroll() {
  $("messages").scrollTop = $("messages").scrollHeight;
}
function context() {
  const m = modelInfo.find((m) => m.id === $("model").value);
  $("context").textContent = m
    ? `${m.context_length.toLocaleString()} token context · Includes history and response`
    : "";
}
async function connect() {
  error("");
  try {
    const response = await fetch("/v1/models", { headers: headers() });
    const result = await response.json();
    if (!response.ok) throw Error(result.error?.message || "Could not connect");
    modelInfo = result.data;
    $("model").replaceChildren();
    for (const m of modelInfo) {
      const option = document.createElement("option");
      option.value = m.id;
      option.textContent = m.id;
      $("model").append(option);
    }
    context();
    busy(false);
  } catch (e) {
    error(e.message);
    $("status").textContent = "Connection needed";
  }
}
async function generate() {
  error("");
  controller = new AbortController();
  busy(true);
  const body = bubble("assistant", "");
  let answer = "",
    completed = false;
  try {
    const messages = [
      ...($("system").value.trim()
        ? [{ role: "system", content: $("system").value }]
        : []),
      ...history,
    ];
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: headers(),
      signal: controller.signal,
      body: JSON.stringify({
        model: $("model").value,
        messages,
        max_tokens: Number($("max").value),
        temperature: Number($("temperature").value),
        top_p: Number($("top-p").value),
        repetition_penalty: Number($("penalty").value),
        stream: true,
        stream_options: { include_usage: true },
      }),
    });
    if (!response.ok) {
      const data = await response.json();
      throw Error(data.error?.message || "Generation failed");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const payload = frame
          .split("\n")
          .filter((l) => l.startsWith("data:"))
          .map((l) => l.slice(5).trimStart())
          .join("\n");
        if (!payload) continue;
        if (payload === "[DONE]") {
          completed = true;
          continue;
        }
        const data = JSON.parse(payload);
        if (data.error) throw Error(data.error.message);
        answer += data.choices?.[0]?.delta?.content || "";
        body.textContent = answer;
        scroll();
        if (data.usage)
          $("status").textContent =
            `${data.usage.completion_tokens} tokens generated`;
      }
    }
    if (!completed)
      throw Error(
        "The connection ended before generation completed. Retry when the server is ready.",
      );
    history.push({ role: "assistant", content: answer });
  } catch (e) {
    if (e.name === "AbortError") {
      if (answer) history.push({ role: "assistant", content: answer });
      else body.parentElement.remove();
    } else {
      error(e.message);
      body.parentElement.remove();
    }
  } finally {
    controller = null;
    busy(false);
    scroll();
    $("prompt").focus();
  }
}
$("composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (controller) return;
  const prompt = $("prompt").value.trim();
  if (!prompt) return;
  // A failed/cancelled request can leave a final user turn: replace it on retry.
  if (history.at(-1)?.role === "user") history.pop();
  history.push({ role: "user", content: prompt });
  $("prompt").value = "";
  render();
  await generate();
});
$("prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("composer").requestSubmit();
  }
});
$("stop").addEventListener("click", () => controller?.abort());
$("regenerate").addEventListener("click", () => {
  if (controller || !history.length) return;
  if (history.at(-1).role === "assistant") history.pop();
  render();
  generate();
});
$("reset").addEventListener("click", () => {
  history = [];
  render();
  error("");
  busy(false);
  $("prompt").focus();
});
$("connect").addEventListener("click", connect);
$("model").addEventListener("change", () => {
  history = [];
  render();
  context();
  busy(false);
});
$("messages").addEventListener("click", (event) => {
  const button = event.target.closest("[data-prompt]");
  if (button) {
    $("prompt").value = button.dataset.prompt;
    $("prompt").focus();
  }
});
$("settings-toggle").addEventListener("click", () => {
  const expanded =
    $("settings-toggle").getAttribute("aria-expanded") !== "true";
  $("settings-toggle").setAttribute("aria-expanded", String(expanded));
  document.querySelector("aside").dataset.expanded = String(expanded);
});
connect();
