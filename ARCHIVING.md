# Annual course archive for Zenodo

Status: preparation policy only. No archive or DOI is created by this change.
The academic-year cutoff and archive scope must be confirmed before preparing
that year's final snapshot. Publication and removal of old materials require
explicit user approval.

## Prepare a complete, reproducible snapshot

1. Identify the academic year, final Git commit, published course URL and
   preparation date. Preserve a read-only snapshot of that exact version.
   Keep originals and prior published versions.
2. Inventory public teaching sources and rendered HTML, slide PDFs/PowerPoint,
   figures, scripts, notebooks, lab data, exercise sheets and public solutions.
   Include the required runtime/dependency information and build instructions.
   Do not include `.git`, secrets, caches or active databases.
3. Inventory external dependencies: GitHub Release downloads, narrated audio,
   video, shared modules, and feedback/Exit Ticket services. With appropriate
   rights, include necessary assets, not just a list of links. Record source
   URLs, release versions and hashes. Explain any unavoidable exclusion and
   its effect on offline use. Preserve form definitions and synthetic examples
   when permitted, not live response databases or credentials. Do not represent
   an archive with missing essential media as complete.
4. Review the proposed public archive for personal data, grades, student IDs,
   names, submissions, feedback responses, email addresses and image metadata.
   Keep confidential exams, unpublished keys and Thai instructor notes private
   unless the user explicitly approves a separate reviewed inclusion. Redact
   only in the proposed archive, preserving private originals.
5. Review third-party text, figures, fonts, software, music, video and generated
   narration terms. Record authors, sources, licenses and permission evidence
   in `CREDITS` and `RIGHTS`. Do not assume one license covers all material.
   Resolve missing rights or record justified exclusions before approval.

## Reviewable deliverables

- Versioned archive with a README, offline entry point and build/run guide.
- Complete file inventory, external-dependency manifest and explicit exclusions.
- `SHA256SUMS` for included files; independently verify hashes after packaging.
- A separate SHA-256 checksum for the final archive payload.
- Proposed Zenodo metadata: title, academic year/version, creators and confirmed
  affiliations/ORCIDs, description, keywords, resource type, language, dates,
  license/access conditions, funding if applicable, and related identifiers
  linking the exact repository commit and course website.
- Credits/rights review and a privacy review receipt without personal data.
- Verification report covering extraction, offline navigation, necessary media,
  runnable labs, and public links. Report any unresolved dependency honestly.

## Approval and publication

Present the exact archive, checksums, metadata, exclusions and unresolved rights
or privacy issues for user review. Do not upload/publish to Zenodo without
approval. After authorized publication, verify the actual record, metadata,
downloads and file hashes. Add the verified **version-specific DOI** to the
course website's archive index, labeled with the academic year and version.
A concept DOI may be linked separately but does not identify the exact snapshot.
Retain the current site, old compatibility pages and media. Archiving does not
authorize deletion, redirection, NFS publication or storage cleanup.
