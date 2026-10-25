from app.ingest import Page
from app.rag import add_document, embed_document
from app.workflows import RunContext, run_workflow


async def test_sermon_studio_renders_all_outputs(pack, stub_llm, tmp_path):
    wf = pack.workflows["sermon_studio"]
    ctx = RunContext(pack=pack, llm=stub_llm, outputs_dir=tmp_path / "out")
    result = await run_workflow(wf, {"transcript": "Today we look at Matthew 6:25."}, ctx)
    assert set(result.outputs) == {
        "summary",
        "key_points",
        "discussion_guide",
        "blog_post",
        "social_posts",
        "bulletin_blurb",
    }
    assert len(stub_llm.calls) == 6
    assert all("church office" in c["prompt"] for c in stub_llm.calls), "voice include missing"
    names = {f.name for f in result.files}
    assert "discussion_guide.docx" in names and "summary.md" in names


async def test_ask_retrieves_and_cites(pack, stub_llm, conn):
    policy = add_document(
        conn,
        "Building Use Policy",
        "/x/policy.md",
        "md",
        [Page("The fellowship hall seats 120 people. Keys are issued by the office.")],
    )
    bulletin = add_document(
        conn,
        "Bulletin",
        "/x/b.md",
        "md",
        [Page("Fall festival on the back lawn with chili and games.")],
    )
    for doc_id in (policy, bulletin):
        await embed_document(conn, doc_id, stub_llm.embed, "stub")
    ctx = RunContext(pack=pack, llm=stub_llm, conn=conn)
    result = await run_workflow(
        pack.workflows["ask"], {"question": "How many does the hall seat?"}, ctx
    )
    assert result.citations, "expected at least one citation"
    assert result.citations[0].title == "Building Use Policy"
    assert "[1]" in stub_llm.calls[-1]["prompt"]


async def test_missing_required_input_is_an_error(pack, stub_llm):
    import pytest

    with pytest.raises(ValueError):
        await run_workflow(pack.workflows["ask"], {}, RunContext(pack=pack, llm=stub_llm))
