from tools.normalize_benchmark_corpus import normalize_whitespace, parse_html_document


def test_normalize_whitespace_collapses_runs():
    assert normalize_whitespace("a   b\n\n\nc") == "a b\n\nc"


def test_parse_html_document_strips_hidden_and_scripts(tmp_path):
    target = tmp_path / "sample.html"
    target.write_text(
        """
        <html>
          <body>
            <div style="display:none">hidden text</div>
            <script>console.log('ignore')</script>
            <div>Visible paragraph one with enough words to survive filtering.</div>
            <div>Visible paragraph two with enough words to survive filtering.</div>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    text = parse_html_document(target)

    assert "hidden text" not in text
    assert "console.log" not in text
    assert "Visible paragraph one" in text
