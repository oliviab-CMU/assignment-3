# scottyrl — Assignment 3 Report

**11-768 AI Agents, CMU, Fall 2026.**

*This report is written to be read by a generalist: every section states what was
done, shows the evidence (plots/tables), and explains what the evidence means in
plain terms before drawing conclusions.*

---

## Part 2: Synchronous REINFORCE and GRPO

### What we built and why

The trainer alternates two phases. First it **collects rollouts**: the agent
plays the task against the environment, one episode per task, using the current
policy. Then it **updates** the policy from those episodes and refreshes the
sampling policy so the next batch is on-policy.

The same loop implements both algorithms; the only difference is one line of
math that decides how much each episode is worth:

- **REINFORCE** (`p2a`, 1 rollout per task) uses the raw episode reward as the
  advantage: $A = r$. A success is reinforced, a failure contributes nothing.
  It is unbiased but noisy, because every task is judged on a single coin flip.
- **GRPO** (`p2b`, 8 rollouts per task) compares each rollout to the other
  rollouts of the *same task*: $A_i = r_i - \frac{1}{G}\sum_k r_k$. The group
  mean acts as a baseline, so easy and hard tasks cancel out and only
  "better than my usual attempt" is reinforced. When all $G$ rewards are
  identical there is no signal at all; we call that a *degenerate* group and skip
  it entirely. Following Dr.GRPO we do not divide by the group's standard
  deviation, which avoids up-weighting changes on tasks where the policy is
  nearly deterministic.

The token-level mask matters as much as the advantage: only tokens the policy
itself sampled are trained on, so the model learns to *act* rather than to
predict the environment's replies.

### Setup

Both runs use the provided configs unchanged, on the `serve_number_search_sync`
deployment: Qwen3.5-4B with a rank-32 LoRA adapter, 50 optimizer steps, 8 tasks
per step, and an evaluation every 10 steps on 50 held-out tasks with 4 rollouts
each. `p2a` uses the unclipped `importance_sampling` loss; `p2b` uses clipped
`ppo`. A 10-step probe run was done first and checked out (real transcripts,
importance weights ≈ 1.0, no sanity warnings).

### Results

Every panel below plots a metric against training step, as the handout
specifies. Each pass@ curve is labelled with its exact logged key, because
training and evaluation use different values of *k* and are different
quantities.

#### p2a — REINFORCE (1 rollout per task)

![p2a diagnostics](report_plots/p2a_number_search_reinforce_diagnostics.png)

#### p2b — GRPO (8 rollouts per task)

![p2b diagnostics](report_plots/p2b_number_search_grpo_diagnostics.png)

#### Side by side

![REINFORCE vs GRPO](report_plots/compare_reinforce_vs_grpo.png)

| | p2a REINFORCE (G=1) | p2b GRPO (G=8) |
|---|---|---|
| Eval success rate, step 0 → 50 | 0.34 → 0.455 (peak 0.465 at step 40) | 0.42 → 0.995 |
| Eval success rate at step 10 | 0.32 | 0.925 |
| Eval `pass@4`, step 0 → 50 | 0.78 → 0.84 | 0.84 → 1.00 |
| Train success rate, steps 40–49 mean | 0.54 | 0.99 |
| Mean `train/grad_norm` | 0.357 | 0.074 |
| `train/entropy_proxy`, first → last step | 0.195 → 0.180 | 0.155 → 0.102 |
| Steps with nothing to train on | 0 of 50 | 19 of 50 (all rewards identical) |
| Mean importance weight | 1.000 | 1.000 |
| Wall-clock for 50 steps | 21.5 min | 35.0 min |

### What the evidence means

**The pipeline is correct.** In both runs the importance weight *mean* sits at
exactly 1.000 at every step (the `mean` curve in each drift plot). That is the
fingerprint of synchronous on-policy training: every batch was sampled by the
very policy that then trained on it. The tails confirm it — the per-step
minimum and maximum stay within roughly 0.53–1.48 (`p2a`) and 0.55–2.29 (`p2b`),
the ordinary spread of a single sampling draw, and the per-step std stays
between 0.013 and 0.032. In the asynchronous Part 3 run those tails widen
measurably, which is the contrast that makes this diagnostic worth plotting. If
sampled log-probabilities were attached to the wrong tokens, the mean would
drift away from 1.0; it does not, so this is the cleanest available check that
the masking and log-prob alignment work. No episodes were abandoned in either
run, and the p2a run never had a batch with nothing to learn from.

**GRPO learns much faster and much further.** It goes from 42% to 92.5% held-out
success in the first 10 steps and reaches ~99% by step 20, while REINFORCE only
creeps from 34% to a peak of 46.5% at step 40 before slipping back to 45.5% at
step 50. The train-time curves agree: GRPO's mean train success rate in the
last ten steps is 0.99, versus 0.54 for REINFORCE, which still oscillates
between roughly 0.1 and 0.9 from batch to batch. Note that `pass@4` for
REINFORCE (0.84) is far above its `success_rate` (0.455): the policy *can*
solve the task, it just does so unreliably, which is exactly the variance a
baseline is supposed to remove.

**Is that difference expected? Yes, and for a concrete reason.** In Number
Search a task is either solved or not, so a reward is 0 or 1. Under REINFORCE
the advantage is that sparse number itself, so a failed episode contributes no
gradient at all, and a successful one is reinforced regardless of whether the
policy usually solves that task already. GRPO instead uses the other seven
rollouts on the *same* task as a control. A hard task where one rollout in eight
succeeds produces one strongly positive and seven negative advantages, whereas
an easy task where all eight succeed produces none. The estimator is therefore
far less noisy at the same number of update steps — it spends its gradient
budget where there is still something to learn. This is the textbook motivation
for GRPO, and the handout predicts it.

**Two secondary observations.** First, GRPO's gradient norm is much smaller
(0.074 vs 0.357) and its entropy proxy falls further (0.155 → 0.102). The smaller
gradient is expected because centering removes the large positive rewards that
dominate an uncentered REINFORCE gradient; the sharper entropy drop is the
policy becoming confident as it masters the task. Second, GRPO's efficiency
shows up as *degeneracy*, not as a bug: the degenerate fraction first reaches
1.0 at step 17, and from then on a typical step has all eight groups with all
eight rollouts succeeding, so they are correctly skipped. 19 of 50 steps have no
training examples at all. That is the
expected end state once a task is solved — there is simply nothing left to learn
from it — and it also means the late `p2b` eval plateau at ~99% is real rather
than a stalled run. The `[sanity]` warnings the framework printed about
degenerate groups are therefore benign here; they would be a real problem only
if they appeared *early*, before the task was solved.

**Cost.** GRPO costs about 1.6× more wall-clock per step here (35 vs 21.5 min for
50 steps) because it samples eight times as many episodes per task (4,739 action
tokens per step vs 845, roughly a 5.6× increase — less than 8× because a solved
GRPO episode needs fewer turns). It is still the better buy: it reaches in 10 steps a
success rate REINFORCE never approaches in 50.

---

## Part 3: Asynchronous GRPO

### What we built and why

The synchronous trainer wastes a lot of time. In `p2b` the learner spends 68% of
every step **idle**, waiting for rollouts to finish: with 8 tasks × 8 rollouts
per step, the sampler works for ~23 s while the trainer sits still, then the
trainer works for ~4 s while the sampler sits still.

The asynchronous trainer breaks that alternation into a pipeline. Eight
**rollout workers** continuously generate groups and push them into a bounded
queue; the **learner** pulls groups off the queue, updates, and publishes a new
policy version. Training and sampling now happen at the same time, on separate
GPUs.

The price is that a group may be generated by a policy a version or two behind
the one now being trained. Two mechanisms handle that:

- **Staleness bound.** Each result carries the policy version that produced it.
  The learner checks `current_version − group_version` *when it consumes the
  group* and discards anything staler than `max_staleness = 2`.
- **Importance weighting.** Accepted off-policy data is reweighted by
  $\rho = \pi_\text{new}(a \mid s) / \pi_\text{old}(a \mid s)$, and the `ppo`
  loss clips that ratio so a single badly-weighted token cannot dominate a
  batch.

Two correctness details matter for the design. Every group takes **one**
`(policy, version)` pair captured before its first rollout, so a publish that
happens mid-group cannot mislabel it. And evaluation is scheduled as a
**background task** on a captured pair, so a slow evaluation never stalls the
learner.

### Setup

The provided config runs unchanged on the `serve_number_search_async`
deployment (2×H100, **split** — one GPU trains, one samples, unlike Part 2's
colocated server). It matches `p2b` exactly except for the concurrency: 8 groups
of 8 rollouts per step, 50 steps, `max_staleness = 2`, queue capacity 16, and
the clipped `ppo` loss. A 6-step probe at full width was run first and checked
out: exactly 48 groups consumed for 6 steps, 8 stale drops, importance weights
at 1.000, and `learner_wait_frac` already down to 0.01.

The run finished with `{'train_steps': 50, 'groups_consumed': 400,
'groups_dropped_stale': 36}`, also written to the last record of
`runs/p3_number_search_async.jsonl` as `summary/*`.

### Results

![p3 diagnostics](report_plots/p3_number_search_async_diagnostics.png)

| | p2b GRPO sync | p3 GRPO async |
|---|---|---|
| Eval success rate, step 0 → 50 | 0.42 → 0.995 | 0.29 → 0.99 |
| Eval success rate at step 20 | 0.99 | 0.645 |
| Eval `pass@4`, step 0 → 50 | 0.84 → 1.00 | 0.70 → 1.00 |
| Train success rate, steps 40–49 mean | 0.99 | 0.99 |
| Mean `train/grad_norm` | 0.074 | 0.079 |
| `train/entropy_proxy`, first → last | 0.155 → 0.102 | 0.183 → 0.065 |
| Steps with nothing to train on | 19 of 50 | 6 of 50 |
| Importance weight — mean | 1.000 | 1.000 (0.9992–1.0007) |
| Importance weight — min | 0.549 | 0.389 |
| Importance weight — max | 2.287 | 2.802 |
| Importance weight — std | 0.016 (0.013–0.023) | 0.025 (0.015–0.044) |
| `learner_wait_frac` (median, updating steps) | 0.68 | 0.21 |
| `learner_wait_frac` (median, all steps) | 0.71 | 0.27 |
| Wall-clock for 50 steps (first to last record) | 35.0 min | 22.4 min |

**Importance weights need to be read as a distribution, not a mean.** The mean
`ρ` is 1.0000 at every one of the 44 updating steps, which looks like perfectly
on-policy data — and the mean nearly hides the truth. The tails say otherwise:
the **minimum** `ρ` per step averages 0.63 and reaches **0.389**, while the
**maximum** averages 1.53 and reaches **2.80**. In other words, in the worst
step some individual token was 2.6× *less* likely under the training policy
than under the policy that sampled it (1/0.389), and another up to 2.8× more
likely. That is exactly
what one expects from a 1–2-version lag: the policy has moved a little, and a
few token-level ratios move a lot. The mean stays at 1 because for every token
pushed up another is pushed down. The per-step **std** (median 0.025) is the
honest summary of the spread, and min/max bound it.

This is also why the clipped `ppo` loss is the right choice here rather than
plain `importance_sampling`: a maximum ratio of 2.8 would give that token 2.8×
its gradient weight with no ceiling. The spread is also visibly wider than
synchronous `p2b`'s (min 0.389 vs 0.549, max 2.80 vs 2.29, std 0.025 vs 0.016),
as it must be: `p2b` is strictly on-policy apart from ordinary sampling noise.
The drift panel shows all four moments; note that the `mean` curve is visually
indistinguishable from the dotted 1.0 reference, which is precisely the point
that the mean alone is uninformative.

**Learning.** The async run reaches the same place more slowly per update: 0.29
→ 0.99 held-out success, but it gets to 0.645 at step 20 where `p2b` was already
at 0.99. Two things explain that, and neither is a bug. First, `p3` starts lower
(0.29 vs 0.42 baseline) — the initial policy snapshotted by a different server
is simply a bit weaker. Second, 8.3% of all generated groups (36 of 436) were
discarded as stale, so the learner sees somewhat less fresh data per step. It
still finishes at 0.99 with `pass@4 = 1.00`, and 6 skipped steps versus `p2b`'s
19 shows it stays useful later into the run. Note the last training record
reports `train/policy_version = 44`, not 50: six skipped steps meant the policy
advanced 44 times over 50 iterations, which is the documented step-versus-version
lag and a good check that the accounting is right.

**Speed.** The overlap works. Median `learner_wait_frac` falls from **0.68 to
0.21** over the steps that actually update — the learner spends 21% of a step
waiting instead of 68%. (Counting all 50 steps, including ones with no update
to do, the medians are 0.71 and 0.27; `p3`'s six skipped steps are the last six
of the run, where the learner has nothing to wait for, so the updating-step
figure is the honest one for the overlap claim.) Total wall-clock for the same
50 steps drops from 35.0 to 22.4 minutes, a **1.6× speedup**. Measured per
*update*, which is the fairer unit because `p2b` had 31 updates and `p3` had 44,
it is 67.8 s → 30.6 s, a **2.2× speedup**.

The per-update number understates the architectural win and the total number
overstates it, so both are reported. The reason `p3` is not 3–4× faster overall
is visible in the timing panel: an async update is *slower* (7.9 s median over
updating steps vs 4.0 s) because the async server trains on **one** GPU while
the colocated sync server trains on **two**. Async buys overlap and pays for it
with half the training hardware — which is the trade the deployment makes by
design, and the reason `serve_number_search_async.py` exists as a separate
server.

### How the agent's behavior changed

Three training transcripts, one from each third of the run. Headers are quoted
from the first line of each `.jsonl` file.

**Early — step 5, `number_search.train.35`, `policy_version=2`, reward 0.0**
(range 93–1023, target 1010):

```
t0 guess(558) too low    t1 guess(790) too low    t2 guess(906) too low
t3 guess(964) too low    t4 guess(991) too low    t5 guess(1002) too low
t6 guess(1011) too high  t7 guess(1006) too low   t8 guess(1009) too low
t9  "I have used all 10 guesses..."  → no tool call (out of steps)
```

**Middle — step 25, `number_search.train.226`, `policy_version=23`, reward 0.0**
(range 44–630, target 84):

```
t0 337 too high   t1 190 too high   t2 118 too high   t3  75 too low
t4 141 too high ← outside the surviving interval [76,117]
t5 108 too high   t6  95 too high   t7  90 too high   t8  80 too low
t9  85 too high (out of steps)
```

**Late — step 45, `number_search.train.391`, `policy_version=40`, reward 1.0**
(range 36–927, target 154):

```
t0 481 too high  t1 258 too high  t2 146 too low   t3 202 too high
t4 174 too high  t5 160 too high  t6 153 too low   t7 156 too high
t8 154  Correct!
```

Three concrete changes, each also measurable across the whole run:

1. **From scanning to bisecting.** The early transcript only ever moves uphill
   (558, 790, 906, 964, 991, 1002) and burns eight guesses getting within 90 of
   the answer. The late one halves the interval each turn. To measure this
   across every logged transcript, each `guess(n)` is scored by its distance
   from the midpoint of the interval still consistent with the feedback so far,
   divided by that interval's width. The score is 0 for a perfect bisection, 0.5
   for a guess at an endpoint, and 0.25 on average for a blind guess. Taking the
   median over all guesses in one step's directory, it falls from **0.145 at
   step 5 to 0.001 at step 45**. The policy learned the algorithm, not just the
   format. (Recompute from `runs/p3_number_search_async_trajectories/`: replay
   each episode's feedback to get the interval, then take the median of
   `|n - mid| / width` over the `guess` calls in each `step_NNNNNN/`.)

2. **It stops repeating itself.** At step 5, 6.5% of guesses were a number the
   agent had already guessed in that episode and been told the answer to. That
   falls to 3.5% by step 15, 1.8% by step 25, 0.6% by step 30, and **0.0%** at
   steps 40 and 45. Re-asking a question you already have the answer to is pure
   waste, and the run learns to stop doing it.

3. **It recovers from its own mistakes — unevenly.** The middle transcript guesses
   141, which the feedback already ruled out (after `118 is too high` and
   `75 is too low` the surviving interval is [76, 117]). It immediately returns
   to the interval with 108 and keeps narrowing. Not all eight rollouts of that
   group do as well: one of them, after `82 is too low`, guesses 77, 80, 79, 81
   and 82 again — all below the interval [83, 92] — and never recovers, and
   two others spend their last guess just outside an interval that has already
   collapsed to a single value. The early agent handles this worse still: when
   it runs out of steps it ignores the format entirely and writes a sentence,
   which is rejected as unparseable. Note that unparsed output is rare at every
   stage (at most 1.2% of turns), so this is a recovery story, not a
   formatting-collapse story.

The `policy_version` in the headers is worth a glance. A step directory is
filled by whatever training rollouts *finish* during that iteration, and with
eight workers each captures its policy pair at the top of its own group. So a
publish landing while several groups are in flight splits one directory across
versions: step 5 holds 1 episode at version 2, 53 at version 3 and 14 at
version 4. That is the async behaviour working as intended — each group
correctly reports the version that actually generated it, and the version a
group carries is fixed at its start even when a publish happens midway through
its own eight rollouts.

