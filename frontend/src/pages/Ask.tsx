import { useState } from "react";
import { api, type Citation } from "../api/client";

export default function Ask() {
  const [q, setQ] = useState("");
  const [answer, setAnswer] = useState("");
  const [cites, setCites] = useState<Citation[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      const r = await api.ask(q);
      setAnswer(r.answer);
      setCites(r.citations);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h2>Ask the Office</h2>
      <p className="muted">Answers come only from documents in the Library, with sources.</p>
      <label>
        Question
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Who can drive the church van?" />
      </label>
      <button className="primary" disabled={busy || !q.trim()} onClick={submit}>
        {busy ? "Looking..." : "Ask"}
      </button>
      {error && <p className="error">{error}</p>}
      {answer && (
        <div className="two-col">
          <section className="card"><h3>Answer</h3><pre className="wrap">{answer}</pre></section>
          <section className="card">
            <h3>Sources</h3>
            {cites.map((c, i) => (
              <div key={c.chunk_id} className="cite">
                <strong>[{i + 1}] {c.title}{c.page ? `, p. ${c.page}` : ""}</strong>
                <p className="small">{c.text}</p>
              </div>
            ))}
          </section>
        </div>
      )}
    </div>
  );
}
