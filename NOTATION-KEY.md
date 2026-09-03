# Family-Wide Notation Key

Cross-reference for the d(q) research program: a foundational preprint (**Paper 0**, *Contractible Reliability Curves*, Lucz & Forstner 2026) plus six papers (A–F) that specialize it. Each paper is internally consistent; this key resolves symbols that are reused with different meanings across papers, so the dissertation can be assembled without retrofitting. Created 2026-05-26; Paper 0 enthroned as the root 2026-06-28.

## The papers

| Paper | Topic | Venue |
|---|---|---|
| **0** | **Contractible reliability curves — the d(q) SLA foundation (all papers cite this)** | preprint (Lucz & Forstner 2026) |
| A | Cross-subsidization in pooled SLA / unit-of-work menus | Decision Support Systems |
| B | Dynamic viability of fixed-price separating SLA menus | Electronic Commerce Research and Applications |
| C | Engineering subsumption: tail-latency controls as implicit d(q) | Future Generation Computer Systems |
| D | Adversarial friction menus for bot containment | Computers & Security |
| E | Economic robots.txt: priced AI-mediated content access | Information Economics and Policy |
| F | Evolutionary dynamics of bot-defender friction games | Chaos, Solitons & Fractals |

**Dissertation chapter order (ARC_SYNC_20260710):** A -> C -> B -> D -> F -> E. A = economic failure; C = engineering implementation bridge; B = dynamic viability of the published menu; D = trust boundary + empirical owner of the AGWA derivations; F = nonlinear dynamic boundary; **E = constructive endpoint (institutional synthesis).**

**Paper 0 is the root.** It defines the shared core below — `d(q)`, `p(q)`, `h(θ,q) = p(q) + θ·d(q)`, `q*(θ)`, the Welfare-Loss-Minimizing Self-Selection Theorem (separating menu beats pooling), and the incident model — and every paper A–F cites it as the foundation (`Lucz & Forstner 2026, Contractible Reliability Curves`). A–F are specializations, not independent roots; A is the cross-subsidy deep-dive of Paper 0 §4.4/§6.3, not the family root.

## Shared core objects (defined in Paper 0; inherited by A–F)

| Symbol | Meaning |
|---|---|
| q in [0,1] | Protection / priority / queue position. Higher q = stronger protection, lower expected degradation. |
| d(q) | Expected violation / degradation severity at protection q. Decreasing, convex. |
| p(q) | Price schedule for protection q. Increasing, convex. |
| theta | Client/workflow sensitivity to degradation. The scalar type is the **affine special case** of a general private workflow loss L_w(v): under L_w(v) = theta_w*v, expected loss is theta_w*d(q) and theta_w is a sufficient statistic. No incentive-compatibility result is claimed for arbitrary nonlinear L_w (see A §7, B §3.1, C §3.1). |
| h(theta,q) | Per-unit burden: p(q) + theta*d(q). |
| q*(theta) | Type-specific optimal protection, from p'(q*) + theta*d'(q*) = 0. |
| F(theta) | Type distribution over sensitivities. |
| U_i | Workload volume of client/workflow i. |
| gamma, delta | Exponential severity-curve parameters: d(q) = gamma*exp(-delta*q). |
| alpha | Price-curve exponent: p(q) = A*exp(alpha*q) (in B, C). |

## COLLISION: eta (the most important one to watch)

`eta` has three distinct meanings across the family. Always disambiguate in cross-references.

| Paper | eta means |
|---|---|
| A | Cost-reduction rate from deferral in the workflow cost function c_i(q,tau,m) = a_i[c_0 + c_1(exp(kappa q)-1)exp(-eta*tau)](1+rho*m_i). |
| B | Client adjustment speed in the gradient/plan-switching dynamics dq_i/dt = -eta[p'(q_i)+theta_i d'(q_i)]. |
| C | (not a primary control variable; uses A/B objects by reference) |
| D | (friction-update gain is written k_phi, not eta — no collision) |
| F | Defender adaptation speed (controller gain) in df/dt = eta[b_obs(x(t-tau)) - b_target]. Primary bifurcation parameter. |

**B <-> F mapping (used by the B3 / F3 cross-references):** B's client adjustment speed eta and F's defender adaptation speed eta play analogous roles — the gain of the adapting side of a delayed feedback loop. B's Nyquist thresholds (eta_c, k_z,c, tau_c) and F's Hopf boundary tau_c(eta) are the same destabilization phenomenon in cooperative (B) vs adversarial (F) settings.

## COLLISION: tau

| Paper | tau means |
|---|---|
| A | Delay tolerance / deadline window (a workflow attribute the client chooses). Higher tau = more deferrable. |
| B | Reporting / measurement / provisioning delay in the feedback loop (a lag, not a tolerance). |
| C | Delay tolerance (follows A's usage in the deferral discussion). |
| D | Measurement / reaction delay in botness estimation and friction update (a lag). |
| E | Delay tolerance / freshness window for content access (higher tau = staler, cheaper). |
| F | Information lag (a lag): tau_d defender measurement lag, tau_b bot observation lag; tau = symmetric delay. Secondary bifurcation parameter. |

**Rule of thumb:** in A, C, E, tau is a *tolerance* (chosen, higher = cheaper/slower). In B, D, F, tau is a *lag* (imposed, higher = more destabilizing). State which sense is meant whenever tau crosses a paper boundary.

## COLLISION: alpha

| Paper | alpha means |
|---|---|
| A | (uses kappa for cost-curve exponent; alpha not primary) |
| B, C | Price-curve exponent in p(q) = A*exp(alpha*q). |
| F | Congestion coefficients alpha_V, alpha_E, alpha_M in the frequency-dependent payoffs (intra-strategy crowding). Unrelated to a price curve. |

## COLLISION: gamma

`gamma` carries two unrelated meanings; always disambiguate in cross-references.

| Paper | gamma means |
|---|---|
| 0, B, C, F | Severity-curve prefactor: d(q) = gamma*exp(-delta*q); gamma = d(0), the worst-case (q=0) expected severity. |
| A | **Metered workload intensity** — effective compute units (ECU) per billable interaction. A's type is the pair (theta, gamma); intensity is observable/metered, multiplies both invoice and loss, and cancels out of plan choice. A's own severity curve is written d(q) = exp(-beta*q), so A does NOT use the prefactor sense — the symbol is fully repurposed. |

**Rule of thumb:** in A, gamma is an *observable cost driver* (ECU/interaction); everywhere else it is the *severity intercept* d(0). State which sense whenever gamma crosses a paper boundary.

## COLLISION: beta (ARC_REVIEW_20260710)

`beta` carries three unrelated meanings; always disambiguate in cross-references.

| Paper | beta means |
|---|---|
| A | **Severity-decay exponent**: A writes its degradation curve d(q) = exp(-beta*q) (beta = 2 in the §7 calibration) — A's beta plays the role the family key calls delta. A does not use the gamma prefactor (gamma is repurposed as intensity there). |
| B, C | Scheduler aggressiveness / WFQ weight slope: w(q) = exp(beta*q). By C §4.2's heavy-traffic identification, this beta EQUALS the induced severity exponent delta (delta = beta) — same object as the family delta, approached from the engineering side. |
| F | Detection rates beta_V, beta_E, beta_M for the three bot strategies (asymmetric observation). Unrelated to severity or scheduling. |

**Rule of thumb:** map A's beta and B/C's beta to the family delta when assembling chapters; F's beta_i never crosses into the menu papers.

## COLLISION: rho (ARC_REVIEW_20260710)

`rho` carries four unrelated meanings.

| Paper | rho means |
|---|---|
| A | Workflow dependency-overhead coefficient in the cost function c_i(q,tau,m). |
| D | Residual-exposure factor of evasion: g(f) = rho*f (the share of friction growth an evader still absorbs; enters kappa via c_2*rho/c_1). |
| E | Publisher-posted royalty rho_j(a) for content-use type a. |
| F | Mimicry throughput penalty: pi_M's value term is v_A*rho/(1+alpha_M*x_M), rho in (0,1). NOT D's residual-exposure rho, despite F inheriting D's other cost symbols. |

**Rule of thumb:** D's rho and F's rho look adjacent (both adversarial-cost parameters) but are different objects; cite them as "residual-exposure rho (D)" and "mimicry throughput rho (F)" when they appear within one chapter.

## Convention: the contractual unit and the three mechanisms (ARC_SYNC_20260710)

- The contractual client is a **workflow class**, not a person or account: one organization can hold several workflow types simultaneously on the same menu.
- Three distinct allocation mechanisms recur and must not be conflated:
  - **voluntary q** — a cooperative workflow chooses protection q from a published price–severity menu (A, B, C, E user-side);
  - **assigned f = phi(b)** — a defender imposes friction f as a function of an inferred botness score b; not a voluntary choice (D, F);
  - **posted rho_j(a)** — a publisher posts a royalty rho_j for content-use type a (E).
- They share graded, machine-readable treatment and feedback constraints, but they are **not applications of one theorem**.

## Paper-specific symbols (no collision, listed for assembly)

| Symbol | Paper | Meaning |
|---|---|---|
| beta | B | Scheduler aggressiveness (WFQ weight slope in w(q)=exp(beta*q)). Distinct from F's beta_V/E/M detection rates. See COLLISION: beta below. |
| mu_i | B | Type masses (B avoids m_i: m is B's curvature tolerance, and m_i is A's dependency depth). |
| S_max | B | Severity cap: d(0) = gamma <= S_max, the fifth viability condition (makes V bounded). |
| tau_z | B | Provider smoothing time constant (NOT the reporting lag tau). |
| V_L | B | Lyapunov total-burden gap (a value; distinct from region V). |
| Phi, Psi | B | Loop linearization maps: Phi = client deviations -> congestion/severity error; Psi = provider state -> client marginal burden (A stays reserved for the price prefactor). |
| K_W | B | Lipschitz constant 1/((alpha+delta)*theta_min) in the client-entry robustness bound (Theorem 5). |
| theta_ref | B | Menu-defined mid-queue type: (A*alpha/(gamma*delta))*exp((alpha+delta)/2); population-centered only by explicit hypothesis. |
| PLI | A, B | Pooling Loss Index (welfare loss, not a transfer). |
| CSI_XS | A, B | Monetary Cross-Subsidy Index. |
| XS_i | A, D, E | Monetary cross-subsidy for client i = U_i[r_i - c_i]. |
| V (region) | B | Viability region in (gamma, delta) space. NOT a value. |
| V_E | E | Economic-robots viability region. |
| B_F | D | Adversarial friction band [f_min, f_switch]. |
| b in [0,1] | D, F | Botness / distrust score. |
| f in [0,1] | D, F | Friction intensity. |
| beta_V, beta_E, beta_M | F | Detection rates for visible/evasive/mimicry strategies (asymmetric observation). |
| x_V, x_E, x_M | F | Population fractions of bot strategies (replicator state). |
| k_phi | D | Friction-update gain. |
| k_z | B | Provider engineering-adaptation gain. |
| rho_j(a) | E | Publisher-posted royalty for content-use type a. See COLLISION: rho below. |
| m_i | A | Dependency depth / critical-path length of workflow i. |

## Cross-references between papers (the program's joints)

- **0 -> all (A–F):** Paper 0 is the foundation. It establishes `d(q)`, `p(q)`, `h(θ,q)=p(q)+θ·d(q)`, `q*(θ)`, and the Welfare-Loss-Minimizing Self-Selection Theorem (§4.4). A expands §4.4/§6.3 into the monetary cross-subsidy indices (PLI, CSI_XS, XS_i); B expands §6.4 (temporal heterogeneity / dynamic plan selection); C expands §7.2 (clustered-violation incident model) and §7.4 (multidimensional severity); D's friction menu and E's pricing specialize Paper 0's published d(q) curve (§5); and §7.7 (empirical validation) is realized by D's AGWA study. Cite as Lucz & Forstner (2026), *Contractible Reliability Curves*.
- **A -> B, C:** A develops Paper 0's pooling result into the monetary cross-subsidy accounting; B asks if a fixed separating menu stays viable dynamically; C shows engineering controls induce implicit d(q) menus. B1 (menu bandwidth theorem) explains A's segmentation/finite-tier results structurally.
- **B <-> F:** same delayed-feedback destabilization, cooperative vs adversarial (see eta mapping above).
- **C <-> E:** C publishes machine-readable *quality* (severity-curve endpoint, C3); E publishes machine-readable *prices* (ERF, RSL). Design symmetry.
- **C, B -> regulatory:** B2 positions V as a DORA-auditable envelope; C3's published severity curve is the artifact an auditor checks.
- **D -> F:** D owns the AGWA bot-strategy decomposition pipeline (feature extraction -> GMM -> botness score -> temporal dynamics). F inherits cluster proportions, detection-rate asymmetry, and reaction-lag tau by citation.
- **D <-> E:** D's friction menu is E's enforcement layer for unverified/non-paying access; E's contract token raises D's detection probability p_detect.
