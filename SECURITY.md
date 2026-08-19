# Security Policy

## Reporting a vulnerability

Please report security issues through GitHub's private vulnerability
reporting rather than a public issue:

**[Report a vulnerability](https://github.com/scotCW/metal4Foam/security/advisories/new)**

This keeps the report private between you and the maintainer until a
fix is available, instead of disclosing it via a public issue first.

This is a single-maintainer project without a formal SLA — reports will
be acknowledged and looked at as soon as reasonably possible, but there's
no guaranteed response time.

## Scope

In scope: the metal4Foam plugin itself — the solver code, the Metal
backend, and the build integration in this repository.

Out of scope: OpenFOAM core and [metal-cpp](https://github.com/scotCW/metal4Foam/tree/master/thirdParty/metal-cpp)
(vendored, Apache-2.0) — please report issues in those upstream. Metal
and macOS platform issues belong with Apple.

## What "security" realistically means here

metal4Foam is a linear-solver plugin run locally as part of an OpenFOAM
simulation — it doesn't listen on a network, parse untrusted input over
the wire, or handle credentials or secrets. The realistic risk surface
is memory-safety bugs in the C++/Metal backend (e.g. an out-of-bounds
read on a malformed or corrupted matrix), not the usual web-application
vulnerability classes. Report anything that looks like memory corruption,
undefined behavior, or a crash triggerable by mesh/matrix input as
described above.

## Supported versions

There are no versioned releases yet — only the current `master` branch
is supported.
