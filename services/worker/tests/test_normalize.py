from exam_tutor_worker.pipelines.normalize import normalize_markdown

def test_normalize_markdown():
    md = "a\n\n\n\n  b  \n"
    out = normalize_markdown(md)
    assert "\n\n\n" not in out
