"use client";

import { useRef, useState } from "react";

type Mode = "doubt" | "practice" | "pyq";

export default function ChatPage() {
  const [mode, setMode] = useState<Mode>("doubt");
  const [exam, setExam] = useState("GATE_DA");
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [sources, setSources] = useState<any[]>([]);
  const controllerRef = useRef<AbortController | null>(null);

  async function send() {
    setOutput("");
    setSources([]);
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    const res = await fetch("http://localhost:8000/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input, mode, exam }),
      signal: controller.signal,
    });

    if (!res.ok || !res.body) {
      setOutput(`HTTP ${res.status}`);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buf = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      // SSE parsing: events are separated by \n\n
      const parts = buf.split("\n\n");
      buf = parts.pop() || "";

      for (const part of parts) {
        const lines = part.split("\n");
        let event = "message";
        let data = "";
        for (const ln of lines) {
          if (ln.startsWith("event:")) event = ln.slice(6).trim();
          if (ln.startsWith("data:")) data += ln.slice(5).trim();
        }
        if (!data) continue;
        const obj = JSON.parse(data);

        if (event === "token") {
          setOutput((prev) => prev + (obj.delta || ""));
        } else if (event === "final") {
          setOutput(obj.answer || "");
          setSources(obj.used_chunks || []);
        }
      }
    }
  }

  return (
    <main style={{ padding: 24, maxWidth: 1000, margin: "0 auto" }}>
      <h1>Chat</h1>

      <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
        <label>
          Exam:
          <select value={exam} onChange={(e) => setExam(e.target.value)} style={{ marginLeft: 8 }}>
            <option value="GATE_DA">GATE_DA</option>
            <option value="UPSC_PRELIMS">UPSC_PRELIMS</option>
            <option value="UPSC_MAINS">UPSC_MAINS</option>
          </select>
        </label>

        <label>
          Mode:
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode)} style={{ marginLeft: 8 }}>
            <option value="doubt">Doubt</option>
            <option value="practice">Practice</option>
            <option value="pyq">PYQ</option>
          </select>
        </label>
      </div>

      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        rows={4}
        style={{ width: "100%", padding: 12 }}
        placeholder="Ask a question..."
      />

      <button onClick={send} style={{ marginTop: 8, padding: "10px 14px" }}>
        Send
      </button>

      <section style={{ marginTop: 24 }}>
        <h2>Answer</h2>
        <pre style={{ whiteSpace: "pre-wrap", background: "#f6f6f6", padding: 12, borderRadius: 8 }}>
          {output}
        </pre>
      </section>

      <section style={{ marginTop: 24 }}>
        <h2>Sources (retrieved)</h2>
        <div style={{ display: "grid", gap: 12 }}>
          {sources.map((s, i) => (
            <div key={i} style={{ padding: 12, border: "1px solid #ddd", borderRadius: 8 }}>
              <div style={{ fontWeight: 600 }}>
                [chunk:{s.chunk_id}] {s.source_title}
              </div>
              <div style={{ opacity: 0.8, fontSize: 12 }}>
                exam={s.exam} subject={s.subject} topic={s.topic} score={s.score?.toFixed?.(3)}
              </div>
              <pre style={{ whiteSpace: "pre-wrap" }}>{s.content}</pre>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
