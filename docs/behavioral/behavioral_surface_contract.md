# Behavioral Surface Contract

**Repository Interface between BSVE/MSML and MPML**

---

## Purpose

This document defines the public contract between BSVE/MSML and MPML.

Behavioral Surfaces are produced by BSVE/MSML and consumed by MPML.

MPML intentionally does **not** know how Behavioral Surfaces are calibrated,
constructed or validated. Those responsibilities remain within BSVE/MSML.

Instead, MPML depends only upon the public metadata described here.

This separation allows both repositories to evolve independently while
maintaining compatibility.

---

# Ownership

Repository responsibilities are intentionally separated.

| Repository | Responsibility                                               |
| ---------- | ------------------------------------------------------------ |
| BSVE/MSML  | Construct Behavioral Surfaces                                |
| BSVE/MSML  | Calibrate Behavioral States                                  |
| BSVE/MSML  | Validate Behavioral Surface quality                          |
| MPML       | Consume Behavioral Surfaces                                  |
| MPML       | Evaluate strategies conditioned on Behavioral States         |
| MPML       | Produce strategy recommendations                             |
| MRML       | Consume MPML recommendations and perform execution and risk management |

MPML should never duplicate Behavioral Surface construction logic.

---

# Behavioral Surface

Every Behavioral Surface should expose the following public metadata.

| Field           | Description                        |
| --------------- | ---------------------------------- |
| surface_id      | Stable machine-readable identifier |
| surface_version | Semantic version                   |
| display_name    | Human-readable name                |
| states          | Collection of Behavioral States    |
| metadata        | Optional extensible metadata       |

The internal implementation is not part of the public contract.

---

# Behavioral Prediction Artifact

Behavioral Surfaces classify market behaviour.

Predictive models generate forecasts conditioned on those Behavioral States.

These are intentionally separate artifacts.

Behavioral Surfaces answer

> "What market state are we currently observing?"

Prediction artifacts answer

> "Given this Behavioral State, what does the trained model predict?"

MPML consumes both artifacts.

Behavioral Surface metadata determines which prediction artifacts are
applicable, while prediction artifacts provide the model outputs used during
strategy evaluation.

Behavioral prediction artifacts should expose, at minimum,

| Field                | Description                          |
| -------------------- | ------------------------------------ |
| surface_id           | Behavioral Surface identifier        |
| surface_version      | Surface version used during training |
| state_id             | Behavioral State used for training   |
| model                | Model family (MLP, LSTM, …)          |
| target_horizon       | Prediction horizon                   |
| feature_set          | Training feature set                 |
| prediction_timestamp | Artifact creation timestamp          |
| metadata             | Optional extensible metadata         |

Prediction-specific fields (such as probabilities, confidence scores or signal
strength) remain model-dependent and are outside the scope of this contract.

---

# Behavioral State

Every Behavioral State should expose the following public metadata.

| Field        | Description                        |
| ------------ | ---------------------------------- |
| state_id     | Stable machine-readable identifier |
| display_name | Human-readable name                |
| description  | Short description                  |
| metadata     | Optional extensible metadata       |

Behavioral States are immutable value objects.

---

# Current Behavioral Surfaces

The following Behavioral Surfaces currently form part of the public contract.

## Trend × Volatility

Surface ID

```
trend_vol
```

Canonical states

```
LVTF
HVTF
LVR
HVR
```

Legacy aliases accepted for compatibility

```
LVMR → LVR
HVMR → HVR
```

---

## Reactive JPY

Surface ID

```
reactive_jpy
```

Canonical states

```
JPY_NON_EXTREME

JPY_CONSENSUS_YOUNG

JPY_CONSENSUS_MATURING

JPY_CONSENSUS_MATURE
```

---

# Compatibility

Behavioral State identifiers constitute stable external interfaces.

Future releases should preserve canonical identifiers whenever practical.

Where historical names require migration, compatibility aliases may be
provided, but only one canonical identifier should exist for each state.

---

This contract intentionally specifies repository interfaces rather than model
implementations.

Behavioral Surface artifacts and Behavioral Prediction artifacts constitute the
public interface between MSML and MPML.

How those artifacts are produced remains an internal implementation detail of
BSVE/MSML.

---

# Architectural Principle

Behavioral Surfaces should be treated as immutable research artifacts.

MPML consumes Behavioral Surface metadata but should remain agnostic to the
algorithms used to generate those artifacts.

Future Behavioral Surfaces should become available to MPML by registering new
metadata rather than modifying MPML algorithms.

---

# Relationship to the Behavioral Surface Registry

This document defines the repository interface between BSVE/MSML and MPML.

Implementation details of the MPML registry are documented separately in

```
docs/behavioral/behavioral_surface_registry.md
```

The registry is one possible implementation of this contract.

The contract itself is implementation-independent.