from app.workflows import load_pack


def test_church_pack_loads(settings):
    pack = load_pack(settings.packs_dir, "church")
    assert pack.manifest.name == "church"
    assert {"sermon_studio", "announcements", "ask", "letter"} <= set(pack.workflows)


def test_every_prompt_referenced_exists(pack):
    for wf in pack.workflows.values():
        for step in wf.steps:
            if step.prompt:
                assert (pack.prompts_dir / step.prompt).exists(), f"{wf.id}: {step.prompt}"


def test_nav_points_at_real_workflows(pack):
    for item in pack.manifest.nav:
        for wf_id in item.workflows:
            assert wf_id in pack.workflows, f"nav {item.id} -> missing {wf_id}"
