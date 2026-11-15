# Course 2105623 publishing instructions

These project rules supplement the account-wide lab standards.

## Inspect before editing

Inspect the worktree, uncommitted changes, branch, remotes, current remote HEAD,
and `.github/workflows/` before changing or publishing anything. Preserve existing
work. Use a separate actor/task branch or workspace and record its machine,
owner/task and base commit. Never discard changes or change a remote before
understanding its role.

## Publish only to the dedicated course repository

- Repository: https://github.com/soorathep/course-2105623
- Pages root: https://soorathep.github.io/course-2105623/
- Course home: https://soorathep.github.io/course-2105623/teaching/2105623/
- Push approved course updates to this repository, never to `soorathep/Test`.
- The `main` push workflow is `.github/workflows/pages.yml`. Inspect its actual
  filename and contents before publishing if it changes.
- Preserve the `/course-2105623` project-site prefix and existing source paths.
  The workflow runs Jekyll, preserves downloadable Quarto source, applies
  `tools/fix_site_links.py`, checks the Pages size limit and deploys.
- Do not copy the research site's CNAME or change its domain settings.

## Compatibility and private material

Preserve content, working links, videos, audio, downloads, feedback forms,
Exit Tickets and all student-facing functions. Existing `skhgroup.net` pages
and media remain compatibility endpoints. Do not remove or redirect them
without checking impact and obtaining the user's explicit approval.
Existing Test Release assets are intentional dependencies, not authorization
to publish new course changes to Test. Preserve their verified links.
Keep form origins/backends unchanged unless explicitly authorized. Do not
submit synthetic responses into student results during routine link checks.
Keep final exams, private answer keys, student submissions/grades and Thai
speaker notes on the local machine. Public exercise solutions already
published remain part of the course website.

## Verify and report

After changes, run the appropriate content/function checks, test the published
site, and verify GitHub Pages deployment for the exact pushed commit. Report
the repository, commit, published URL, changes, test scope, broken links and
unresolved issues. A successful build alone does not establish working forms,
media playback or correct numerical results. Label read-only availability tests
separately from actual submission/playback tests.

## Annual Zenodo archive

At the end of each academic year, prepare the complete reviewable snapshot
described in ARCHIVING.md. Include necessary files, metadata, credits and
SHA-256 checksums. Review personal data and third-party rights. Do not publish
the archive or remove old materials without the user's approval. Add the
version-specific DOI only after publication and verification, never a guessed
or concept DOI in place of the archived version. No end-of-year date or
recurring automation has been assumed.
