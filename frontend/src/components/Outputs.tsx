import type { RunResult } from "../api/client";

export default function Outputs({ result }: { result: RunResult | null }) {
  if (!result) return null;
  return (
    <div className="outputs">
      {Object.entries(result.outputs).map(([name, text]) => (
        <section key={name} className="card">
          <header>
            <h3>{name.replace(/_/g, " ")}</h3>
            <button onClick={() => navigator.clipboard.writeText(String(text ?? ""))}>Copy</button>
          </header>
          <textarea defaultValue={String(text ?? "")} rows={12} />
        </section>
      ))}
      {result.files.length > 0 && (
        <p className="muted small">Saved: {result.files.join(", ")}</p>
      )}
    </div>
  );
}
