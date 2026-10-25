import { useState } from "react";
import { api, type RunResult } from "../api/client";
import Outputs from "../components/Outputs";

const TYPES = ["Donor thank-you", "Visitor follow-up", "Benevolence response", "Volunteer appreciation", "Event invitation"];

export default function Letters() {
  const [type, setType] = useState(TYPES[0]);
  const [fields, setFields] = useState("recipient: \ndate: \ndetails: ");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);

  function parseFields(text: string): Record<string, string> {
    const out: Record<string, string> = {};
    for (const line of text.split("\n")) {
      const i = line.indexOf(":");
      if (i > 0) out[line.slice(0, i).trim()] = line.slice(i + 1).trim();
    }
    return out;
  }

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.runWorkflow("letter", { letter_type: type, fields: parseFields(fields) }));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h2>Letter Drafter</h2>
      <label>
        Letter type
        <select value={type} onChange={(e) => setType(e.target.value)}>
          {TYPES.map((t) => <option key={t}>{t}</option>)}
        </select>
      </label>
      <label>
        Details (one per line, name: value)
        <textarea rows={6} value={fields} onChange={(e) => setFields(e.target.value)} />
      </label>
      <button className="primary" disabled={busy} onClick={run}>{busy ? "Writing..." : "Draft letter"}</button>
      {error && <p className="error">{error}</p>}
      <Outputs result={result} />
    </div>
  );
}
