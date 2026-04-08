from tools.normalize_benchmark_corpus import normalize_whitespace, parse_document, parse_html_document


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


def test_parse_document_falls_back_to_html_for_mislabeled_pdf(tmp_path):
    target = tmp_path / "sample.pdf"
    target.write_text(
        """
        <!DOCTYPE html>
        <html>
          <body>
            <div>Visible annual report paragraph with enough words to survive filtering.</div>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    text = parse_document(target)

    assert "Visible annual report paragraph" in text
