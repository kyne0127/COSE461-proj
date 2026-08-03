# Introduction

Language-guided robot manipulation has seen remarkable progress with the advent of vision-language models (VLMs) capable of interpreting natural language instructions and grounding them in visual scenes [CITE]. A central challenge in this paradigm is **ambiguity**: when a task instruction such as *"pick the cup and put it in the red box"* does not uniquely identify the target or destination in the current scene, the robot must request clarification before proceeding. Existing methods address this challenge at task initiation—before the robot begins moving—by detecting whether the initial scene is ambiguous with respect to the given instruction [AmbResVLM, KnowNo, CLARA].

However, robot manipulation unfolds over time. Between the moment the task is accepted and the moment the robot reaches a pick or place action, the scene can change: a bystander places a second cup on the table, the original cup is moved to a different location, or the target object is removed entirely. A scene that was unambiguous at task initiation may become ambiguous or invalid mid-execution. **Existing ambiguity resolution methods have no mechanism to detect or respond to these dynamic changes.** The robot continues executing toward a grounding that is no longer valid, risking grasping the wrong object, failing to find the destination, or causing unsafe behavior.

This gap motivates three concrete research questions that our work addresses.

**RQ1. Can a robot system detect grounding invalidation during task execution and adapt its plan while preserving the user's original intent?**
Existing ambiguity resolution methods detect and resolve ambiguity only at the initial scene, before task planning begins. In practice, however, the scene changes dynamically during execution. The user's intent—expressed as a fixed task instruction—remains constant, but the objects or destinations that intent refers to may no longer be valid. Re-evaluating grounding in response to environmental changes, while keeping the user's intent fixed, is the core problem that existing approaches leave unaddressed.

**RQ2. How can scene recognition under a fixed task query avoid force-matching bias when the target environment has changed?**
Even if RQ1 is recognized as a problem, using a fixed task query to perceive a changed scene introduces unintended consequences. Open-vocabulary detectors such as GDino always return the highest-confidence match for a text query, regardless of whether the target object is actually present—a behavior known as force-matching. For example, when the target object has been removed from the scene, the detector may hallucinate a match to a visually similar but irrelevant object. This causes the system to misclassify true object absence (INVALID) as object movement (AMBIGUOUS), leading to incorrect robot decisions.

**RQ3. Does episode memory—accumulated scene recognition results across prior checkpoints—improve grounding evaluation accuracy at the current checkpoint?**
Robot execution is a temporally continuous process, and the grounding state at any checkpoint depends on what occurred at previous ones. When an object has moved, for instance, it is impossible to determine from the current image alone whether the detected object is the same instance or a newly introduced one—prior position information is required. Existing methods treat each checkpoint independently, making them poorly suited for scenarios where temporal context is essential. We investigate whether injecting episode memory into the grounding evaluator meaningfully improves CONTINUE/ASK/STOP accuracy.

---

To address these questions, we propose **MemoryGrounding**, a two-stage mid-execution grounding monitor. The two stages directly correspond to the two fundamental sub-problems in RQ1:

**Stage 1 — Trigger Detection.** A lightweight finite state machine (FSM) continuously monitors the robot's execution video stream using GDino-based per-frame detection. The FSM detects semantically significant scene changes—object count shifts, movements beyond a spatial threshold, and disappearances—and fires a trigger signal when grounding re-evaluation is warranted. This stage operates efficiently at video rate and avoids the computational cost of invoking the full grounding evaluator on every frame.

**Stage 2 — Grounding Evaluation.** Upon trigger, a memory-conditioned evaluator determines the current grounding state and issues a robot decision (CONTINUE, ASK, or STOP). This stage combines two complementary modules: (i) a rule-based GDino+Molmo ensemble with explicit force-match correction (addressing RQ2), and (ii) a fine-tuned Molmo decision model conditioned on episode memory $\mathcal{M}$ accumulated across all prior checkpoints (addressing RQ3). The two modules are combined under a conservative voting policy that leverages the reliability of rule-based STOP detection and the semantic flexibility of the learned model for ASK disambiguation.

The key insight connecting the two stages is that the trigger signal itself provides the first piece of evidence for Stage 2: the *nature* of the detected change—whether an object appeared, disappeared, or moved—shapes the prior over possible grounding states and is recorded in $\mathcal{M}$ for future reference.

---

We make the following contributions:

1. **Two-stage problem decomposition.** We formally define mid-execution grounding validity monitoring as a two-stage problem—trigger detection and grounding evaluation—and demonstrate that both stages are necessary for reliable deployment on continuous robot execution streams. This decomposition is absent from prior work.

2. **FSM-based trigger system (Stage 1).** We propose a finite state machine that monitors continuous video using lightweight GDino detection, firing grounding re-evaluation only when semantically significant scene changes are detected. This enables efficient deployment without invoking heavy VLMs on every frame.

3. **Force-match-aware ensemble (Stage 2, RQ2).** We introduce check_grounding_ensemble, a GDino+Molmo ensemble that explicitly addresses open-vocabulary detector force-matching by preferring Molmo's absence signal over GDino's spurious detections when the two detectors disagree. We further propose memory-validated change correction, which retroactively corrects false "moved" records in episode memory when object absence is confirmed by the ensemble.

4. **Episode memory for grounding evaluation (Stage 2, RQ3).** We introduce EpisodeMemory, a structured accumulation of detection history, scene changes, and past decisions across checkpoints. We show that memory-conditioned grounding evaluation outperforms checkpoint-independent methods, particularly for scene changes that require temporal context such as object movement.

5. **memory_ensemble and evaluation.** We propose a conservative voting ensemble combining rule-based and fine-tuned decision making, and evaluate on vla-evaluation-v4—a dataset of 70 samples across 7 scenario types covering the full space of mid-execution scene changes. memory_ensemble achieves [X]% decision accuracy, outperforming rule-based-only ([Y]%) and fine-tuned-only ([Z]%) approaches.
