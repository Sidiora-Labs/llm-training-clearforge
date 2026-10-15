# Data preparation and packing

Preparation accepts explicitly supplied UTF-8 JSONL sources with declared licenses
and languages. Only declarations `en`, `eng`, and `english` are included; this is
source filtering, not language detection or license verification. The provenance
file retains each input locator, source hash, license, content hash, and duplicate
status. Memory use is bounded by the largest JSONL record plus SQLite's working
buffers; the corpus and deduplication index are stored on disk.

Splits are deterministic hashes of the seed and canonical complete record content.
Exact duplicate records across all included sources receive one split assignment.
This does not guarantee separation of near duplicates, overlapping excerpts, or
shared messages in otherwise different conversations. Such overlap requires
additional corpus-specific curation before preparation.

Tokenizer training uses only the verified training split. Its published identity
can be reused to pack a different corpus, including supervised chat for an existing
base model. Manifest versions, modes, counts and required provenance hash controls
are validated. Inputs are checked before use and again before atomic publication;
persistent mutation causes failure without publishing a partial output. This is
an integrity check for local files, not authentication of their author or a
transactional snapshot against a writer that changes and restores bytes.
Tokenizer files with enabled padding or truncation are rejected. Content encoding
prevents reserved control-token injection. `decode(ids)` retains structural tokens;
`decode(ids, skip_special_tokens=True)` omits them for display.

Pretraining concatenates documents with explicit BOS and EOS boundaries into a
bounded sequence buffer. Every text token and EOS target is retained exactly once;
BOS targets are ignored, including boundaries within a row. Consecutive windows
share one context token, and only the final partial corpus window is padded.
Causal attention can span document boundaries within a window; EOS is a delimiter,
not a separate attention mask. Packing holds at most one encoded document plus a
sequence buffer, and writes arrays through bounded chunks into memory-mapped files.

Supervised chat remains isolated by conversation. Supported roles are an optional
initial system message followed by alternating user and assistant messages. Tool
roles are unsupported. Only assistant content and its EOS are supervised; user,
system, role, padding and leading context tokens are ignored. An empty assistant
message supervises EOS. Conversations without supervised targets and windows
without shifted supervision are recorded in exclusion files.

Long conversations use overlapping windows and preserve every assistant target,
but each window sees only its own tokens, including the one-token overlap. Earlier
prompt content may therefore be absent for later targets. Choose a context length
appropriate for the corpus; this implementation does not reconstruct prompts at
window boundaries. Labels align with input tokens in all arrays; the model applies
the next-token shift exactly once.
