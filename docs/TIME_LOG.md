# Time log

All local times use America/Los_Angeles (PDT, UTC−07:00).

Entries before October 7, 2026 were reconstructed on that date, not recorded live. Sources: Git commit
timestamps, Codex session transcripts (request times) and file modification times of recorded reports.
Commit times show when work was saved, not when it started. File times in `docs/` before September 11 were
reset by a copy on that date and are not used. Elapsed figures are wall-clock time between readings,
including user review, discussion, builds and rendering; they are not compute time. Compute times are
quoted separately from the recorded reports.

## LeWM runtime and the first drone model (June 2–15)

| Milestone | Local time | Source |
| --- | --- | --- |
| First commit | 2026-06-02 23:33:53 | Git |
| Rust/Candle CUDA runtime scaffolded | 2026-06-02 23:46:39 | Git (`175a233`) |
| Native PushT trainer with exact resume | 2026-06-03 00:42:59 | Git (`85d7b69`) |
| Python-vs-Rust benchmark chart | 2026-06-03 09:03:25 | Git (`650bc2f`) |
| Modular drone world-model pipeline | 2026-06-12 16:45:38 | Git (`5c02b9b`) |
| Fused CUDA drone planning and Bevy viewer | 2026-06-12 19:43:11 | Git (`a736108`) |
| Trained drone LeWM controls the simulator | 2026-06-14 17:48:42 | Git (`d0a8ba4`) |
| Last June commit: plan-trace diagnostics | 2026-06-15 12:31:02 | Git (`e4dea74`) |

Work was committed on seven days: June 2–3, 10 and 12–15. No session transcripts exist for this period,
so no start times or elapsed totals are given.

## SkyJEPA from the paper (September 3)

| Milestone | Local time | Source |
| --- | --- | --- |
| Native SkyJEPA UAV pipeline committed | 2026-09-03 17:16:50 | Git (`96fb8e6`) |
| Trim-aware geometric action prior (hybrid design) | 2026-09-03 19:08:49 | Git (`6bc8a36`) |
| Trained controller showcased with video | 2026-09-03 20:14:45 | Git (`c0242d5`) |
| Last commit of the day | 2026-09-03 22:10:06 | Git (`613083a`) |

First to last commit: **4 hours 53 minutes 16 seconds**. The start of work is not recorded.

## SkyJEPA audit, protocol and three-seed results (September 4–5)

| Milestone | Local time | Source |
| --- | --- | --- |
| Remediation plan and acceptance gates | 2026-09-04 22:57:06 | Git (`c1b8fb9`) |
| Preregistered three-seed protocol frozen | 2026-09-05 00:02:36 | Git (`b8ae05c`) |
| Complete three-seed results recorded | 2026-09-05 01:43:46 | Git (`985556d`) |
| Corrected recording and GIF committed | 2026-09-05 12:16:46 | Git (`898df0b`) |
| Guide consolidated, history separated | 2026-09-05 13:26:29 | Git (`01d9e58`) |

- Plan to complete results: **2 hours 46 minutes 40 seconds**.
- Recorded compute: three models at about 34 minutes each, **102.47 minutes** of training in total on a
  shared RTX 4090 ([SkyJEPA guide](skyjepa.md)).

## Questions about the learned controller (September 11)

- Codex session in this repository, **12:35:08–13:26:34**. Questions about what the network learned and how
  per-step rewards accumulate. No code changes.

## JEPA-Anything, the JEPA Gym and the five-seed study (September 18)

One Codex session in this repository. Request times are from its transcript; completion times are file
modification times of the recorded reports, now copied to `docs/media/` and `benchmarks/`.

| Milestone | Local time | Source |
| --- | --- | --- |
| Session start ("get latest") | 2026-09-18 18:11:07 | Transcript |
| JEPA-Anything investigation requested | 18:20:48 | Transcript |
| Comparison and video requested, not replacement | 18:28:35 | Transcript |
| SkyJEPA vs JEPA-Anything video finished | 18:52:34 | File time |
| Assessment and comparison reports written | 18:54 | File time |
| Move to MuJoCo requested | 19:37:14 | Transcript |
| Engineering-gym look requested | 19:40:35 | Transcript |
| "JEPA Gym" name chosen | 20:06:26 | Transcript |
| Heading control requested | 20:09:43 | Transcript |
| Drone geometry review requested (skids, rotors) | 20:29:11 | Transcript |
| First gym study video (three seeds, plus frame) | 20:35:32 | File time |
| X-frame heading demonstration | 20:44:47 | File time |
| X frame accepted | 20:57:19 | Transcript |
| Five-seed protocol fixed | 20:59:33 | File time |
| Single-frame simplification requested | 21:09:39 | Transcript |
| Single-frame parity check (zero state difference) | 21:15:37 | File time |
| Five-seed study complete | 21:55:11 | File time |
| Five-seed comparison video finished | 21:56:24 | File time |
| Verification complete | 21:59:09 | File time |

- Session start to the first comparison video: **41 minutes 27 seconds**.
- MuJoCo request to the first gym study video: **58 minutes 18 seconds**.
- Session start to the verified five-seed study: **3 hours 48 minutes 2 seconds**.
- Recorded compute for the five-seed study: **50 minutes 32 seconds** end to end (data, two concurrent
  training jobs and serial evaluation on the RTX 4090; `complete.json`). OPF representation training for
  seed 7 took 33.5 seconds.
- Follow-up questions in the same session on **2026-09-19, 21:02:49–21:08:06**: whether OPF flew without the
  physics probe, and how much the geometric action prior contributes. Answers are reflected in
  [the results](jepa-gym-results.md) and the journal.

## Move to rozgo and the engineering journal (October 7)

Recorded live in this session.

| Milestone | Local time | Notes |
| --- | --- | --- |
| Session start | 2026-10-07 17:30:59 | First event of the Claude Code session |
| September 18 work committed | 19:31:35 | `0128aaf` |
| Repository transferred from VertexStudio to rozgo and pushed | 19:32:03 | GitHub push time; old URLs redirect |
| Journal built | 19:41:30 | `build/journal`, 20 media files, 9.5 MB |
| Journal published to rozgo.github.io/le-wm-nv | 19:44:00 | gh-pages, from `a8d8823`; page and media checked live |
| Card added to the MuJoCo Sandbox home page | 19:46:24 | rozgo/mujoco-sandbox gh-pages, from `cb0d84e` |
| Time log closed | 19:47:22 | |

Elapsed for this session at closing: **2 hours 16 minutes 23 seconds** from 17:30:59, including review of
the earlier work, the transfer, the journal and both publishes.
