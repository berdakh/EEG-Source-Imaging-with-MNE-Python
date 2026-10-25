import { useEffect, useState } from "react";
import { api, type Document } from "../api/client";

export default function Library() {
  const [docs, setDocs] = useState<Document[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = () => api.documents().then(setDocs).catch((e) => setError(String(e)));
  useEffect(() => { refresh(); }, []);

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      await api.upload(file);
      await refresh();
    } catch (err) {
      setError(String(err));
    }
  }

  return (
    <div>
      <h2>Library</h2>
      <p className="muted">Bulletins, newsletters, bylaws, policies. PDF, Word, Markdown, text.</p>
      <input type="file" accept=".pdf,.docx,.md,.txt" onChange={onFile} />
      {error && <p className="error">{error}</p>}
      <table>
        <thead><tr><th>Title</th><th>Type</th><th>Chunks</th><th>Added</th><th></th></tr></thead>
        <tbody>
          {docs.map((d) => (
            <tr key={d.id}>
              <td>{d.title}</td><td>{d.kind}</td><td>{d.chunks}</td><td>{d.created_at}</td>
              <td><button onClick={() => api.deleteDocument(d.id).then(refresh)}>Remove</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
