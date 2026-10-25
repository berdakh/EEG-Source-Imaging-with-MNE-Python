# Church Pilot: MVP Scope and Repo Layout

Status: draft, written before discovery. Every feature below is a hypothesis.
After the interview, keep the two features the administrator actually asked for
and cut the rest.

## 1. Goal

A free, fully offline assistant for one church office that removes the
most repetitive weekly writing and listening work. It must run on one office
machine or a small box we bring, send nothing to the internet after install,
and be usable by a non-technical administrator after a 30-minute walkthrough.

The pilot exists to produce four things: a working install, a testimonial,
real usage data, and a reusable core that later verticals (law, CPA, HOA)
plug into as "packs".

## 2. Principles

- Local only. The only network call is downloading models at install time.
- Human approves everything. The app drafts; it never sends, posts, or files.
- Citations or nothing. Any answer from documents shows the source and page.
- Workflows over chat. Buttons that do one job well. Chat is a secondary tab.
- Vertical content is configuration. Prompts, templates and workflows live in a
  pack folder, not in application code.

## 3. MVP features (ranked, cut from the bottom)

### F1. Sermon Studio
Upload or drag in the Sunday recording (mp3, m4a, mp4, wav).
- Transcribe locally (faster-whisper). Show progress; a 40-minute sermon on a
  CPU should finish in well under the hour.
- One click each: summary, key points, small-group discussion guide with
  questions, blog post draft, three social posts, bulletin blurb for next week.
- Scripture references in the transcript are detected and linked to a local
  public-domain Bible text (WEB or KJV) so quotes are verified, not invented.
- Every output is editable and exportable as .docx and .md.
- Transcript is stored and searchable ("when did I preach on forgiveness").

### F2. Announcement Hub
Paste the raw announcements for the week, in any messy form.
- Produces the bulletin blurb, the newsletter section, the email version, the
  social post, and the slide text, all at once, in the church's voice.
- Voice is learned from their past bulletins and newsletters loaded into the
  library (few-shot, not fine-tuning).
- Length and tone presets per channel. Edit any output inline.
- Export to .docx; copy buttons for each channel.

### F3. Ask the Office
Question answering over the church's own documents with citations.
- Library ingests PDF, DOCX, TXT, MD, and past bulletins and newsletters.
- Hybrid retrieval (BM25 plus local embeddings). Answers cite file and page,
  and clicking a citation shows the passage.
- Typical questions: building use policy, bylaws, who to call for the van,
  what did we announce about the retreat last year.
- "I don't know, it's not in the documents" is an acceptable and expected answer.

### F4. Letter Drafter
Templated letters that get retyped every month.
- Templates: donor thank-you, visitor follow-up, benevolence response,
  volunteer appreciation, event invitation, generic church letterhead letter.
- Fill fields by hand or from a CSV (name, amount, date). No member-database
  integration in the MVP.
- Produces .docx on the church letterhead.

### F5. Foundations (not visible, non-negotiable)
- Activity log: who did what, when, which documents were retrieved.
- Everything in one data folder that can be backed up by copying it.
- Single office account with a PIN or password. No multi-user roles yet.
- Settings page: pick model, show disk and memory use, "test offline" button
  that confirms no outbound connections.

## 4. Explicitly out of scope for the pilot

- Integrations with Planning Center, Breeze, ChurchTrac, Mailchimp, QuickBooks.
- Sending email, posting to social media, or writing to any external system.
- Giving records and financial reports.
- Multi-user permissions, remote access, mobile app.
- Fine-tuning. Voice comes from examples in the prompt.
- Speaker diarization, live captioning, translation (phase 2 candidates).

## 5. Success criteria (four-week pilot)

- Administrator uses it unprompted in at least 3 of 4 weeks.
- Self-reported time saved of 2 or more hours per week.
- Sermon processed and outputs produced within one hour of upload each week.
- Zero outbound connections during use, verified with a firewall log.
- One written quote from staff we can use with permission.
- A list of the top three things they wish it did, feeding the next vertical.

## 6. Stack decisions

| Layer | Choice | Why |
|---|---|---|
| Inference | Ollama | One install, handles GPU/CPU, model pulls, embeddings, vision; swap models by name |
| Default model | Qwen3 8B or Gemma 3 12B (Q4) on 16GB+; 4B variants on weak machines | Good writing quality for the size; pack config picks per hardware |
| Embeddings | nomic-embed-text or bge-m3 via Ollama | Small, multilingual, local |
| Transcription | faster-whisper (CTranslate2), small or medium | Runs on CPU in reasonable time; medium if GPU present |
| Backend | Python 3.12, FastAPI, SQLite + sqlite-vec, rank-bm25 | Your strongest language; one file database; hybrid search without a vector DB server |
| Frontend | React + Vite + TypeScript, served by the backend | Standard; works in a browser today and inside a Tauri shell later |
| Desktop shell | Phase 1: launcher script and tray icon, open in browser. Phase 2: Tauri with Python sidecar | Ship the pilot in weeks, package properly after it proves out |
| Documents out | python-docx with church letterhead template | Office already lives in Word |
| Packaging | uv for Python deps, PyInstaller for the backend binary later | Reproducible installs |
| Tests | pytest, Playwright for the UI smoke test | Students can extend these safely |

Deployment target: one Windows or Mac machine in the office acting as a small
server; any office PC on the LAN can open it in a browser. If their hardware is
too weak (likely), bring a Mac mini with 16GB or 24GB as the appliance.

## 7. Core abstraction: packs and workflows

A pack is a folder of configuration that defines a vertical. The church pack
is the first. A workflow is a YAML file:

```yaml
id: sermon_discussion_guide
title: Small-group discussion guide
inputs:
  - name: transcript
    type: document         # from library or a fresh transcription
steps:
  - type: generate
    prompt: prompts/discussion_guide.md
    model: default
outputs:
  - type: docx
    template: templates/discussion_guide.docx
  - type: markdown
```

Step types for the MVP: transcribe, retrieve, generate, extract (JSON schema),
render (docx). Everything F1 to F4 is expressible with these five. Future
verticals mostly add prompts and templates, which is work students can do
against synthetic data without seeing client files.

## 8. Repo layout

Working name only; rename when the LLC has a brand.

```
church-pilot/
├── README.md
├── LICENSE
├── pyproject.toml                 # uv-managed Python project
├── justfile                       # dev, test, build, install commands
├── .env.example
├── .github/workflows/ci.yml       # lint, pytest, frontend build
│
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app, mounts /api and the built UI
│   │   ├── config.py              # data dir, model names, hardware profile
│   │   ├── db/                    # SQLite schema, migrations, sqlite-vec setup
│   │   ├── llm/                   # Ollama client, streaming, prompt rendering (Jinja)
│   │   ├── ingest/                # loaders (pdf, docx, txt, md, audio), chunking
│   │   ├── rag/                   # embeddings, BM25 + vector hybrid search, citations
│   │   ├── transcribe/            # faster-whisper wrapper, segments, scripture detection
│   │   ├── workflows/             # engine: load pack YAML, run steps, store runs
│   │   ├── render/                # docx and markdown output
│   │   ├── jobs/                  # background queue for long tasks (transcription)
│   │   ├── audit/                 # activity log
│   │   └── api/                   # routers: library, workflows, ask, jobs, audit, settings
│   └── tests/
│       ├── unit/
│       └── integration/           # runs against a tiny model in CI
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── pages/                 # Home, SermonStudio, Announcements, Ask, Letters, Library, Settings
│       ├── components/            # editor, citation viewer, job progress, file drop
│       ├── api/                   # typed client for /api
│       └── styles/
│
├── packs/
│   └── church/
│       ├── pack.yaml              # manifest: name, nav items, default model per hardware tier
│       ├── prompts/               # summary.md, discussion_guide.md, announcement_*.md, ...
│       ├── workflows/             # sermon_studio.yaml, announcements.yaml, letters.yaml, ask.yaml
│       ├── templates/             # letterhead.docx, discussion_guide.docx, letter_*.docx
│       ├── data/                  # public-domain Bible text (WEB) for reference checks
│       └── samples/               # synthetic bulletins, a short fake sermon audio, fake letters
│
├── desktop/
│   ├── launcher/                  # start.bat, start.command, tray icon (phase 1)
│   └── tauri/                     # phase 2 shell with Python sidecar
│
├── scripts/
│   ├── install.ps1 / install.sh   # install Ollama, pull models, create venv, create data dir
│   ├── pull_models.py             # pulls the model set for the detected hardware tier
│   ├── make_synthetic_data.py     # generates the packs/church/samples content
│   └── offline_check.py           # proves no outbound connections during a run
│
└── docs/
    ├── mvp-scope.md               # this document
    ├── architecture.md
    ├── pilot-runbook.md           # install day checklist, 30-minute training script
    └── discovery/                 # interview notes and samples (never client PII in git)
```

Runtime data lives outside the repo in one folder, for example
`~/ChurchAI/` on Mac or `%LOCALAPPDATA%\ChurchAI\` on Windows:
`app.db` (SQLite), `files/` (originals), `outputs/`, `logs/`. Models live
wherever Ollama keeps them. Backup is "copy that folder".

## 9. Four-week build plan (evenings and weekends)

Week 1: skeleton and plumbing
- Repo, uv, FastAPI, React scaffold, justfile, CI.
- Ollama client with streaming; settings page that shows model and hardware.
- Library: upload, ingest PDF/DOCX/MD, chunk, embed, store. Hybrid search works
  from a test script.
- Workflow engine runs a YAML with a single generate step.

Week 2: Sermon Studio
- faster-whisper job queue with progress.
- Scripture detection against the local Bible text.
- Workflows: summary, key points, discussion guide, blog, social, blurb.
- Docx rendering with the letterhead template.

Week 3: Announcement Hub and Ask the Office
- Multi-output announcement workflow with voice examples pulled from the library.
- Ask page with citations and passage viewer.
- Activity log and PIN login.

Week 4: install and train
- Install scripts for Windows and Mac; offline check script.
- Synthetic sample pack so the demo works with no church data.
- Install day at the church, 30-minute training, leave the runbook.
- Weekly 15-minute check-in for four weeks.

## 10. Risks and mitigations

- Office hardware too weak: decide on install day from the specs collected in
  discovery; bring a Mac mini appliance if needed.
- Transcription quality on poor audio: test with their actual recording from a
  past Sunday before build week 2.
- Invented scripture quotes or wrong references: verify against the local Bible
  text and mark unverified references visibly.
- Administrator does not adopt it: pick the first feature from their own words
  in discovery, not from this document; make it reachable in two clicks.
- Scope creep from an enthusiastic pastor: everything outside F1 to F4 goes on
  the "next vertical" list, not in the pilot.

## 11. Open decisions

- Repo location: new repository (recommended) versus the current EEG repo branch.
- Appliance budget: are you willing to buy a Mac mini for the pilot if needed?
- Model tier: do you want to test Qwen3 and Gemma 3 side by side on real sermon
  text before fixing the default?
- Name: a working name for the product and the LLC, so the UI and data folder
  are not renamed later.
