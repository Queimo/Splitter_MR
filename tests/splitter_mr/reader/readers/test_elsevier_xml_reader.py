from pathlib import Path

from src.splitter_mr.reader.readers.elsevier_xml_reader import ElsevierXmlReader


SAMPLE_XML = """<?xml version='1.0' encoding='UTF-8'?>
<full-text-retrieval-response xmlns='http://www.elsevier.com/xml/svapi/article/dtd' xmlns:dc='http://purl.org/dc/elements/1.1/' xmlns:ce='http://www.elsevier.com/xml/common/dtd'>
  <coredata>
    <dc:title>Electrode materials for future batteries</dc:title>
  </coredata>
  <originalText>
    <xocs:doc xmlns:xocs='http://www.elsevier.com/xml/xocs/dtd'>
      <xocs:serial-item>
        <article>
          <ce:abstract>
            <ce:para>We evaluate several chemistries for next-generation storage.</ce:para>
          </ce:abstract>
          <ce:authkeywords>
            <ce:author-keyword>battery</ce:author-keyword>
            <ce:author-keyword>electrochemistry</ce:author-keyword>
          </ce:authkeywords>
          <ce:sections>
            <ce:section>
              <ce:section-title>Introduction</ce:section-title>
              <ce:para>Energy transition demands robust storage systems.</ce:para>
              <ce:table>
                <ce:label>Table 1</ce:label>
                <ce:caption>
                  <ce:simple-para>Electrochemical performances of typical MAB technologies.</ce:simple-para>
                </ce:caption>
                <tgroup cols='2'>
                  <thead>
                    <row>
                      <entry>Batteries</entry>
                      <entry>Theoretical Voltage (V)</entry>
                    </row>
                  </thead>
                  <tbody>
                    <row>
                      <entry>Zn-air</entry>
                      <entry>1.65</entry>
                    </row>
                    <row>
                      <entry>Mg-air</entry>
                      <entry>1.81</entry>
                    </row>
                  </tbody>
                </tgroup>
              </ce:table>
            </ce:section>
          </ce:sections>
        </article>
      </xocs:serial-item>
    </xocs:doc>
  </originalText>
</full-text-retrieval-response>
"""


def test_elsevier_xml_reader_generates_clean_markdown(tmp_path: Path):
    xml_file = tmp_path / "paper.xml"
    xml_file.write_text(SAMPLE_XML, encoding="utf-8")

    reader = ElsevierXmlReader()
    result = reader.read(xml_file)

    assert result.conversion_method == "md"
    assert result.reader_method == "elsevier_xml"
    assert "# Electrode materials for future batteries" in result.text
    assert "## Abstract" in result.text
    assert "### Introduction" in result.text
    assert "**Table 1 — Electrochemical performances of typical MAB technologies.**" in result.text
    assert "| Batteries | Theoretical Voltage (V) |" in result.text
    assert "| Zn-air | 1.65 |" in result.text
    assert "ce:section-title" not in result.text
    assert "<ce:" not in result.text


def test_elsevier_xml_reader_delegates_non_xml(tmp_path: Path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("plain text", encoding="utf-8")

    reader = ElsevierXmlReader()
    result = reader.read(txt_file)

    assert result.conversion_method == "txt"
    assert result.text == "plain text"
