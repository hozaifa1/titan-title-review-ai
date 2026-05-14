// BAML declaration for the Hour 3-6 extraction step.
// Generate a Python client with `baml-cli generate` once model credentials are
// configured; titan.ingest.extract.ExtractTitleDocument keeps the same public
// function signature for the offline checkpoint path.

client<llm> GeminiFlash {
  provider google-ai
  options {
    model "gemini-2.0-flash"
  }
}

class Provenance {
  doc_id string
  page int
  char_span int[]?
  snippet string?
}

class PartyParty {
  name string
  role string
  is_entity bool
  capacity string?
}

class TitleDocument {
  doc_id string
  doc_type string
  file_path string
  page_count int
  parsed_at string
  parties PartyParty[]
  extraction_warnings string[]
}

function ExtractTitleDocument(markdown: string, doc_type: string) -> TitleDocument {
  client GeminiFlash
  prompt #"
    Extract a TitleDocument from this title-source markdown.

    Rules:
    - Use null/empty arrays for absent fields.
    - Do not guess missing recording references, legal descriptions, dates, or amounts.
    - Preserve provenance with page number, character span, and a short snippet when available.
    - If the source is ambiguous, add a concise extraction_warnings entry.

    doc_type: {{ doc_type }}

    markdown:
    {{ markdown }}
  "#
}
