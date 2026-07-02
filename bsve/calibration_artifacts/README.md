Versioned BSVE calibration outputs live in this directory.

Each calibration artifact must be immutable and referenced by `calibration_id`
from BSVE validation artifacts (alongside `spec_id`).

### Calibration Episode Semantics

Calibration artifacts summarize **consensus episodes used for hazard estimation**, not every consensus episode identified by the Behavioral Surface.

During calibration, one-bar consensus episodes are intentionally excluded (`min_episode_bars = 2`) before hazard and survival statistics are computed. This reduces sensitivity to transient threshold crossings that provide little information about persistence.

Consequently, calibration diagnostics such as `episode_count`, median duration and survival statistics refer to the filtered calibration population rather than all Behavioral Surface episode identifiers.

Behavioral Surface inspection utilities report all deterministic state segments. These counts are therefore expected to be larger than the calibration episode count while remaining consistent for longer-lived episodes (e.g. survival beyond the Young and Mature boundaries).

Calibration artifacts should always be interpreted together with this distinction in mind. 
