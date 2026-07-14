# data/ — Arctic Shift subreddit dumps (the pipeline's data source)

Save the `.jsonl` files you download from the Arctic Shift download tool
(https://arctic-shift.photon-reddit.com/download-tool) in this folder — both the
posts files and the comments files, for every subreddit in the study.

- **File names do not matter.** The collector reads EVERY `*.jsonl` file in this
  folder and tells posts from comments by their content (posts have a `title`,
  comments have a `body` + `link_id`), so you can name or split the files however
  you like.
- One JSON record per line (exactly what the download tool produces).
- Files are matched into threads automatically: each comment is attached to its
  post via `link_id`, across files.
- The `.jsonl` files are ignored by git (they are data, not code); this README is
  the only tracked file in the folder.

Then run, from the repository root:

    python3 run_pipeline.py            # screens every post in this folder (no network)

The collector screens all posts against the study queries and the corpus-tier
rule from `config/keywords.json`, rebuilds the matching threads in
`cache/threads/`, and the rest of the pipeline continues from there.
