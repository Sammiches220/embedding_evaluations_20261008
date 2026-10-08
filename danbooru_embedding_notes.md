# Embedding Notes

## What to embed (Danbooru Profile)


1. Per-post metadata:

Domains are estimated and should be verified

```bash
Index([
'tag_string_general',  # ✅
'tag_string_character',  # ✅
'tag_string_copyright',  # ✅
'tag_string_artist',  # ✅
'tag_string_meta',   # ✅


'media_asset',   # ✅❔  Rationale: I don't know what this does??
'has_large',  # ❌❔  Rationale: Implementation junk
'bit_flags',  # ❌❔  Rationale: Implementation state flags
'tag_string',  # ❌❔  Rationale: Duplicated with tag_* columns

'id',  # ❌  Rationale: Identity/key
'created_at',  # ❌  Rationale: Danbooru curation timeline is irrelevant for content. "Era" dimension may be useful, but too variable relative to content
'score',  # ❌  Rationale: Should be used for filtering / recommendation later, not fundamental representation of the content
'md5',  # ❌  Rationale: Identity/key
'rating',  # ❌  Rationale: Should be used for filtering / recommendation later, not fundamental representation of the content
'image_width',  # ❌  Rationale: Not relavent to semantic meaning. Should be used for filtering / recommendation later
'image_height',  # ❌  Rationale: Not relavent to semantic meaning. Should be used for filtering / recommendation later
'tag_count_meta',  # ❌  Rationale: Not relavent to semantic meaning. Should be used for filtering / recommendation later
'is_pending',  # ❌  Rationale: Moderation lifecycle
'is_flagged',  # ❌  Rationale: Moderation lifecycle
'is_deleted',  # ❌  Rationale: Moderation lifecycle.  Note that deleted files are never truly deleted
'tag_count',  # ❌  Rationale: Should be used for filtering / recommendation later, not fundamental representation of the content
'fav_count',  # ❌  Rationale: Should be used for filtering / recommendation later, not fundamental representation of the content
'file_ext',   # ❌  Rationale: Should be used for filtering / recommendation later, not fundamental representation of the content
'file_size',  # ❌  Rationale: Not relavent to semantic meaning. Should be used for filtering / recommendation later
'tag_count_general',  # ❌  Rationale: Should be used for filtering / recommendation later, not fundamental representation of the content
'tag_count_artist', # ❌  Rationale: Should be used for filtering / recommendation later, not fundamental representation of the content
'tag_count_character',  # ❌  Rationale: Should be used for filtering / recommendation later, not fundamental representation of the content
'tag_count_copyright',  # ❌  Rationale: Should be used for filtering / recommendation later, not fundamental representation of the content
'up_score', # ❌  Rationale: Should be used for filtering / recommendation later, not fundamental representation of the content
'down_score',  # ❌  Rationale: Should be used for filtering / recommendation later, not fundamental representation of the content
'updated_at',  # ❌  Rationale: Danbooru curation timeline is irrelevant for content. "Era" dimension may be useful, but too variable relative to content
'uploader_id',  # ❌  Rationale: not meaningful for content
'source',  # ❌  Rationale: although meaningful, inconsistent.
'last_comment_bumped_at',  # ❌  Rationale: Irrelevant to content
'last_noted_at',  # ❌  Rationale: Irrelevant to content
'parent_id',  # ❌  Rationale: Irrelevant to content
'has_children', # ❌  Rationale: Existence of children / variations has inconsistent reasoning
'approver_id',  # ❌  Rationale: Irrelevant to content
'is_banned',  # ❌  Creator ban status, although meaningful, is inconsistent with individual content
'pixiv_id',   # ❌  Rationale: Irrelevant to content
'last_commented_at',  # ❌  Rationale: Irrelevant to content
'has_active_children',  # ❌  Rationale: Irrelevant to content
'has_visible_children',   # ❌  Rationale: Irrelevant to content
'file_url',  # ❌  Rationale: Irrelevant to content
'large_file_url',  # ❌  Rationale: Irrelevant to content
'preview_file_url'  # ❌  Rationale: Irrelevant to content
],
      dtype='str')
```


2. Automated generated tags

