import { useEffect, useState } from "react";
import { api, type Health } from "../api/client";

export default function Settings() {
  const [h, setH] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.health().then(setH).catch((e) => setError(String(e))); }, []);

  if (error) return <p className="error">{error}</p>;
  if (!h) return <p>Loading...</p>;
  const ok = (b: boolean) => (b ? "✓" : "✗");
  return (
    <div>
      <h2>Settings</h2>
      <table>
        <tbody>
          <tr><th>Pack</th><td>{h.pack_title}</td></tr>
          <tr><th>Ollama</th><td>{ok(h.ollama.available)} {h.ollama.url}</td></tr>
          <tr><th>Chat model</th><td>{ok(h.chat_model_installed)} {h.chat_model}</td></tr>
          <tr><th>Embedding model</th><td>{ok(h.embed_model_installed)} {h.embed_model}</td></tr>
          <tr><th>Installed models</th><td>{h.ollama.models.join(", ") || "none"}</td></tr>
          <tr><th>Data folder</th><td>{h.data_dir}</td></tr>
        </tbody>
      </table>
      <p className="muted small">Back up by copying the data folder. Models live in Ollama.</p>
    </div>
  );
}
