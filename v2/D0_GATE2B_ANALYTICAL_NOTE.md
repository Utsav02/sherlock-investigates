# D0 mechanics: non-adaptation and adaptive-dominance proofs

**Status:** historical analytical support for an unexecuted Gate 2B draft; no
performance result. The active-investigation extension stopped on 2026-08-29.

## 1. Why Gate 2A BED cannot adapt

Let `T` be binary identity, `p = P(T=AI)`, and `h(x)` binary entropy in nats. In
Gate 2A, family `f` has one neutral mass `n_f` shared by every question. A
question has signed discrimination `d_q`. Conditional on a non-neutral response,
encode `Y=1` for `ai_cue`. Then

```text
P(Y=1 | AI, q)    = (1+d_q)/2
P(Y=1 | human, q) = (1-d_q)/2.
```

The predictive non-neutral response probability is

```text
P(Y=1 | q) = [1 + (2p-1)d_q]/2.
```

The neutral response is an erasure independent of identity. Exact EIG is

```text
I_q(p) = (1-n_f) * {
    h([1+(2p-1)d_q]/2) - h([1+d_q]/2)
}.
```

Because `h(x)=h(1-x)`, the sign of `d_q` cannot affect information value. For
`0 <= |d_1| <= |d_2|`, the weaker binary symmetric channel can be constructed by
passing the stronger channel through an additional independent bit-flip channel
with correlation `|d_1|/|d_2|`. The data-processing inequality therefore gives

```text
I_1(p) <= I_2(p) for every p in [0,1].
```

The common `(1-n_f)` factor preserves the same ordering. Since every configured
probability is strict, Bayesian updates keep the posterior in `(0,1)`. Thus Gate
2A BED must select unused questions in descending `|d_q|`, with only the frozen
question-ID tie break remaining. Answers change posterior confidence but cannot
change the question ranking. The observed one-sequence-per-family diagnostic is
therefore a theorem of the model, not a chance simulation outcome.

Conditional independence supplies a further result: for any fixed selected
question subset, final posterior and expected terminal log loss do not depend on
the order in which that subset is asked. An answer-dependent order that always
ends with the same subset has zero adaptive terminal value.

## 2. Exact expected log loss equals expected posterior entropy

For a correctly specified Bayesian model and log score, conditional expected
terminal loss after history `H_B` is

```text
E[-log P(T | H_B) | H_B] = H(T | H_B).
```

Taking the outer expectation yields

```text
R(pi) = E_pi[-log P(T | H_B)] = E_pi[H(T | H_B)].
```

Gate 2B can therefore integrate each policy tree exactly; sampled episodes are
not needed to estimate policy risk.

## 3. Adaptive dominance over open-loop policies

Let `Pi_OL` contain policies that choose their complete four-question subset
before observing any response. Let `Pi_AD` contain all history-dependent policies
under the same family knowledge, question bank, and budget. Every open-loop
policy is an adaptive policy that ignores its observations, so

```text
Pi_OL is a subset of Pi_AD.
```

Define

```text
pi_OL* = argmin over Pi_OL of R(pi)
pi_AD* = argmin over Pi_AD of R(pi).
```

Minimization over a superset cannot have a larger optimum:

```text
R(pi_AD*) <= R(pi_OL*).
```

Hence Gate 2B adaptation value

```text
Delta_adapt = R(pi_OL*) - R(pi_AD*)
```

is non-negative by construction. A negative value beyond numerical tolerance is
an implementation, optimization, or exact-integration error. A positive value is
not an independent discovery that “BED wins”; it says how much value the
environment assigns to the extra ability to condition future actions on answers.

The same set-inclusion logic gives two additional dominance checks:

- finite-horizon BED cannot be worse than the one-step EIG policy, because the
  latter is one member of the adaptive policy class;
- a family-aware open-loop oracle cannot be worse on average than a global
  open-loop oracle, because it can choose the global subset in every family.

For expected random selection, the global open-loop optimum also cannot be worse:
random is a mixture over deterministic subsets and no mixture can beat the
minimum component under a linear expected-risk objective.

## 4. Strict advantage requires a reachable consequential branch

The dominance inequality can be equality. A strict advantage requires a
positive-probability history at which:

1. the optimal continuation action depends on the observed response; and
2. forcing the wrong branch action increases exact continuation risk.

If both child histories have positive probability and at least one has strictly
positive forced-action regret, integrating over that branch contributes a
strictly positive amount to `Delta_adapt`. Conversely, sequence diversity alone
does not imply strict terminal advantage.

### Concrete asymmetric example

A router moves the posterior from 0.5 to either 0.8 or 0.2 with equal predictive
probability. Two binary specialist questions have

```text
q1: P(+|AI)=0.9, P(+|human)=0.5
q2: P(+|AI)=0.5, P(+|human)=0.1.
```

At posterior 0.8 their EIGs are approximately `0.07270` and `0.06076` nats. At
posterior 0.2 the ranking reverses. With one question remaining, conditioning on
the router answer gains about `0.00597` expected nats over committing to either
specialist. With two remaining slots and both specialists eventually asked, the
policy may reverse their order but obtain exactly zero terminal adaptive value.

This example motivates separate requirements for unique actions, reachable child
histories, forced-action regret, and a changed final subset.

## 5. Threshold units

A log-loss difference `delta` multiplies the geometric probability assigned to
the true class by `exp(delta)`. Thus `0.005`, `0.01`, and `0.02` nats correspond
to approximately `0.5%`, `1.0%`, and `2.0%` multiplicative changes respectively.
They are practical conventions, not natural constants.

Likewise, an absolute history probability of `0.005`, `0.01`, or `0.025` means
roughly one occurrence per 200, 100, or 40 matched episodes. Under exact tree
integration this is a reachability interpretation, not a Monte Carlo precision
requirement.

## 6. What the proofs do not establish

They do not show that the generated tables resemble real humans or AI systems,
that a model can infer the likelihoods, that an SFT model can imitate the policy,
or that synthetic policy value transfers to transcripts. They establish only
which comparisons are mathematical invariants and which remaining quantities can
carry empirical or design information.
