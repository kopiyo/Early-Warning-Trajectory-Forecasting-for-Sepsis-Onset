# Early-Warning-Trajectory-Forecasting-for-Sepsis-Onset
Sepsis kills faster than it is detected, and outcome is strongly time-dependent. The field has produced many predictive models, and almost none of them are used at the bedside.

Early Warning Trajectory Forecasting for Sepsis Onset
Temporal Convolutional Embeddings of ICU Vital-Sign Dynamics
The problem we are solving

Sepsis kills faster than it is detected, and outcome is strongly time-dependent. The field has produced many predictive models, and almost none of them are used at the bedside. Two reasons:

    Alert fatigue. Deployed sepsis alerts commonly fire at positive predictive values in the low tens of percent. When most alerts are wrong, clinicians rationally learn to dismiss them — and the true positives get dismissed too.
    No context. A score of 0.82 does not say whether the patient is stabilising or collapsing, which organ system is driving it, or how fast it is moving. It is a snapshot of a problem that is fundamentally about motion. A patient at 0.55 and falling and a patient at 0.55 and climbing need opposite decisions, and the scalar cannot tell them apart.

The AI solution we are offering

Stop asking the model for a number. Ask it for a coordinate.

A causal Temporal Convolutional Network (TCN) encoder reads a rolling 24-hour window of physiology and emits a 32-dimensional vector. Slide the window forward one hour at a time and the patient becomes a trajectory through that space rather than a point on a dial.

The encoder is trained on a joint objective — reconstruction (keeps physiological detail) plus supervised contrastive loss (forces sepsis-relevant structure into the geometry). The operational alert is then a scalar distance from the stable manifold, computed in the 32-D space where it is metrically meaningful.
The two research questions this notebook answers
	Question 	Evidence produced here
RQ1 	Does the TCN trajectory embedding detect impending sepsis ~6 h in advance at a lower false-alarm rate than a clinical-style baseline and gradient boosting? 	AUPRC, PPV, and false alerts per patient-day vs. baselines
RQ2 	Does the latent space organise into distinguishable directions of decompensation (phenotypes), rather than a single severity axis? 	Supervised 3-axis projection + clustering of drift vectors + animated 3D trajectories

RQ1 is the credibility floor. RQ2 is the actual novel contribution — and it is the answer to the hardest question you will get at the poster: "why not just plot the probability over time?" A scalar says how bad. A trajectory says which way.
