import { useState } from "react";
import { api, type RunResult } from "../api/client";
import Outputs from "./Outputs";

type Props = {
  workflowId: string;
  title: string;
  description: string;
  inputName: string;
  inputLabel: string;
  placeholder?: string;
};

/** One text input in, a set of editable outputs out. Covers most church workflows. */
export default function WorkflowRunner(p: Props) {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.runWorkflow(p.workflowId, { [p.inputName]: value }));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h2>{p.title}</h2>
      <p className="muted">{p.description}</p>
      <label>
        {p.inputLabel}
        <textarea
          rows={10}
          value={value}
          placeholder={p.placeholder}
          onChange={(e) => setValue(e.target.value)}
        />
      </label>
      <button className="primary" disabled={busy || !value.trim()} onClick={run}>
        {busy ? "Working..." : "Generate"}
      </button>
      {error && <p className="error">{error}</p>}
      <Outputs result={result} />
    </div>
  );
}
