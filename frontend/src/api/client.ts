export type Health = {
  ok: boolean;
  pack: string;
  pack_title: string;
  ollama: { available: boolean; url: string; models: string[] };
  chat_model: string;
  embed_model: string;
  chat_model_installed: boolean;
  embed_model_installed: boolean;
  data_dir: string;
};

export type Citation = {
  chunk_id: number;
  document_id: number;
  title: string;
  page: number | null;
  text: string;
  score: number;
};

export type RunResult = {
  outputs: Record<string, string>;
  citations: Citation[];
  files: string[];
};

export type Document = {
  id: number;
  title: string;
  kind: string;
  created_at: string;
  chunks: number;
};

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetch("/api/health").then((r) => json<Health>(r)),
  runWorkflow: (id: string, inputs: Record<string, unknown>) =>
    fetch(`/api/workflows/${id}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inputs }),
    }).then((r) => json<RunResult>(r)),
  ask: (question: string) =>
    fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }).then((r) => json<{ answer: string; citations: Citation[] }>(r)),
  documents: () => fetch("/api/library").then((r) => json<Document[]>(r)),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return fetch("/api/library", { method: "POST", body: form }).then((r) =>
      json<{ document_id: number; job_id: string; pages: number }>(r),
    );
  },
  deleteDocument: (id: number) =>
    fetch(`/api/library/${id}`, { method: "DELETE" }).then((r) => json<{ ok: boolean }>(r)),
  audit: () => fetch("/api/audit").then((r) => json<unknown[]>(r)),
};
