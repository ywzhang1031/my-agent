---
name: boundary-review
description: Review tests for missing boundary cases around intervals, thresholds and transitions.
---
Translate each specified transition into the last value on one side and the first on the other.
Compare those exact inputs to the existing tests; a passing interior example does not cover an endpoint.
Return only the requested candidate IDs. Do not select an interior value merely because it is untested.
